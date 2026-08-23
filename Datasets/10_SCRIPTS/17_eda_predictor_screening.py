# -*- coding: utf-8 -*-
"""
EDA STAGE 5 - screen the candidate predictors before they reach the Quantile Regression.

A quantile regression will happily accept thirty collinear, non-stationary regressors and
return coefficients for all of them. The damage shows up out of sample, so the screening has
to happen here.

FOUR THINGS ARE ESTABLISHED, IN THIS ORDER

  1. STATIONARITY. A predictor in levels that contains a unit root produces a spurious fit
     against a stationary target. Yields, index levels, the dollar index and commodity prices
     are all suspect. Each candidate is tested and, where it fails, the differenced or
     log-differenced form is tested as the replacement. Anything that survives only in
     differences must enter the model in differences.

  2. STRICT NO-LOOK-AHEAD. Every predictor is measured at t and every target at t+1. This is
     enforced mechanically here by shifting the target, not by convention, so the screening
     numbers are honest out-of-sample-style relevance rather than contemporaneous
     correlation. Contemporaneous correlation between RV and the vol index is near 0.8 and
     means nothing for forecasting.

  3. MULTICOLLINEARITY. Variance inflation factors across the surviving set. The realized
     measures are mechanically related to one another - RV = RS_pos + RS_neg exactly, and
     ContVar = RV - Jump exactly - so some pairs cannot both enter. These identities are
     checked numerically rather than assumed.

  4. TAIL RELEVANCE, not mean relevance. A predictor useful for the CENTRE of the
     distribution need not be useful in the 1% tail, and it is the tail we are modelling. So
     alongside the usual predictive R-squared for log-RV, each candidate is scored by the
     pseudo-R-squared of a quantile regression of r(t+1) at tau = 0.05 and tau = 0.01. That
     is the quantity the paper actually cares about.

Outputs: _validation/eda5_stationarity.csv
         _validation/eda5_predictive.csv
         _validation/eda5_vif.csv
         _validation/eda5_identities.csv
         _validation/eda5_cross_index.csv
"""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
VAL = os.path.join(ROOT, '08_VALIDATION')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

# candidate predictors, all dated t
CANDIDATES = [
    # realized-measure family
    'LogRV', 'LogRV_w', 'LogRV_m', 'RV', 'RV_w', 'RV_m',
    'LogRS_neg', 'RSneg_w', 'SignedJump', 'Jump', 'ContVar', 'JumpShare',
    'RSkew', 'RKurt', 'RSV_Ratio', 'RV_RelSE',
    # implied volatility
    'VolIdx', 'LogIV', 'IV_DailyVar', 'VRP',
    # return-side, available on every day including intraday-defect days
    'Return', 'AbsReturn', 'NegReturn', 'Return_Sq', 'Overnight_LogRet',
    # range-based, also available every day
    'ParkinsonVar', 'GarmanKlassVar', 'RangePct',
    # macro / risk factors
    'US10Y_pct', 'TermSpread_pct', 'CreditStress', 'DXY_ret', 'WTI_ret', 'GOLD_ret',
]
# predictors whose LEVEL is suspect and which have a natural differenced counterpart
LEVEL_SUSPECT = ['VolIdx', 'IV_DailyVar', 'US10Y_pct', 'TermSpread_pct']


def adf_p(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 100:
        return np.nan
    try:
        return float(adfuller(x, autolag='AIC')[1])
    except Exception:
        return np.nan


def nw_reg(y, x, lags=10):
    """OLS with Newey-West standard errors. Returns R2 and the HAC t-stat on the slope."""
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 200:
        return np.nan, np.nan, int(m.sum())
    X = sm.add_constant(x[m])
    r = sm.OLS(y[m], X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return float(r.rsquared), float(r.tvalues[1]), int(m.sum())


def qr_pseudo_r2(y, x, tau):
    """Koenker-Machado pseudo R-squared: 1 - (fitted check loss / null check loss)."""
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 300:
        return np.nan
    yy, xx = y[m], x[m]
    try:
        f = QuantReg(yy, sm.add_constant(xx)).fit(q=tau)
        res = yy - f.fittedvalues
        loss1 = np.sum(np.where(res >= 0, tau * res, (tau - 1) * res))
        r0 = yy - np.quantile(yy, tau)
        loss0 = np.sum(np.where(r0 >= 0, tau * r0, (tau - 1) * r0))
        return float(1 - loss1 / loss0)
    except Exception:
        return np.nan


def main():
    stat_rows, pred_rows, vif_rows, ident_rows = [], [], [], []
    logrv_panel = {}
    ret_panel = {}

    for c in CODES:
        a = pd.read_csv(os.path.join(ANA, f'{c}_analysis.csv'), parse_dates=['Date'])
        b = a[a['InSample_B']].copy().reset_index(drop=True)
        logrv_panel[c] = b.set_index('Date')['LogRV']
        ret_panel[c] = b.set_index('Date')['Return']

        # ---------- 1. stationarity ----------
        for col in CANDIDATES:
            if col not in b.columns:
                continue
            p_lvl = adf_p(b[col])
            row = dict(Code=c, Predictor=col, ADF_p_level=p_lvl,
                       Stationary_level=(p_lvl < 0.05) if np.isfinite(p_lvl) else np.nan)
            if col in LEVEL_SUSPECT:
                row['ADF_p_diff'] = adf_p(b[col].diff())
            stat_rows.append(row)

        # ---------- 2. predictive relevance, target at t+1 ----------
        tgt_lrv = b['LogRV'].shift(-1).values      # next-day log realized variance
        tgt_ret = b['Return'].shift(-1).values     # next-day return, for the quantile scores
        for col in CANDIDATES:
            if col not in b.columns:
                continue
            x = b[col].values.astype(float)
            r2, t, n = nw_reg(tgt_lrv, x)
            q05 = qr_pseudo_r2(tgt_ret, x, 0.05)
            q01 = qr_pseudo_r2(tgt_ret, x, 0.01)
            q95 = qr_pseudo_r2(tgt_ret, x, 0.95)
            pred_rows.append(dict(Code=c, Predictor=col, N=n,
                                  R2_LogRV_next=r2, HAC_t=t,
                                  PseudoR2_q05=q05, PseudoR2_q01=q01, PseudoR2_q95=q95))

        # ---------- 3. mechanical identities ----------
        v = b[b['RV_Valid']]
        if len(v) > 100:
            id1 = float((v['RS_pos'] + v['RS_neg'] - v['RV']).abs().max())
            id2 = float((v['ContVar'] + v['Jump'] - v['RV']).abs().max())
            id3 = float((v['IV_DailyVar'] - v['RV'] - v['VRP']).abs().max())
            ident_rows.append(dict(Code=c,
                                   Max_Abs_Err_RSpos_plus_RSneg_minus_RV=id1,
                                   Max_Abs_Err_ContVar_plus_Jump_minus_RV=id2,
                                   Max_Abs_Err_VRP_identity=id3,
                                   Median_RV=float(v['RV'].median())))

        # ---------- 4. VIF on a non-redundant working set ----------
        working = ['LogRV', 'LogRV_w', 'LogRV_m', 'LogRS_neg', 'JumpShare', 'RSkew',
                   'LogIV', 'VRP', 'NegReturn', 'Return', 'RangePct',
                   'TermSpread_pct', 'CreditStress', 'DXY_ret']
        working = [w for w in working if w in b.columns]
        W = b[working].replace([np.inf, -np.inf], np.nan).dropna()
        if len(W) > 300:
            Xv = sm.add_constant(W.values)
            for i, name in enumerate(working):
                vif_rows.append(dict(Code=c, Predictor=name,
                                     VIF=float(variance_inflation_factor(Xv, i + 1))))
        print(f"  [{c}] screened")

    pd.DataFrame(stat_rows).to_csv(os.path.join(VAL, 'eda5_stationarity.csv'), index=False)
    P = pd.DataFrame(pred_rows)
    P.to_csv(os.path.join(VAL, 'eda5_predictive.csv'), index=False)
    V = pd.DataFrame(vif_rows)
    V.to_csv(os.path.join(VAL, 'eda5_vif.csv'), index=False)
    pd.DataFrame(ident_rows).to_csv(os.path.join(VAL, 'eda5_identities.csv'), index=False)

    # ---------- cross-index structure ----------
    L = pd.DataFrame(logrv_panel).dropna()
    R = pd.DataFrame(ret_panel).dropna()
    cross = []
    cl = L.corr()
    cr = R.corr()
    for i in CODES:
        for j in CODES:
            cross.append(dict(A=i, B=j, Corr_LogRV=cl.loc[i, j], Corr_Return=cr.loc[i, j]))
    pd.DataFrame(cross).to_csv(os.path.join(VAL, 'eda5_cross_index.csv'), index=False)

    pd.set_option('display.width', 250)
    print()
    print("=" * 100)
    print("NON-STATIONARY PREDICTORS (ADF fails to reject a unit root at 5%)")
    print("=" * 100)
    S = pd.DataFrame(stat_rows)
    ns = S[S['Stationary_level'] == False]
    if len(ns):
        piv = ns.pivot_table(index='Predictor', columns='Code', values='ADF_p_level')
        print(piv.round(3).to_string())
        print()
        print("  differenced form of the level-suspect ones:")
        d = S[S['ADF_p_diff'].notna()].pivot_table(index='Predictor', columns='Code',
                                                   values='ADF_p_diff')
        print(d.round(4).to_string())
    else:
        print("  none")

    print()
    print("=" * 100)
    print("TOP PREDICTORS of next-day log RV (mean R2 across the six indices)")
    print("=" * 100)
    g = P.groupby('Predictor').agg(R2=('R2_LogRV_next', 'mean'),
                                   min_t=('HAC_t', lambda s: s.abs().min()),
                                   q05=('PseudoR2_q05', 'mean'),
                                   q01=('PseudoR2_q01', 'mean'))
    print(g.sort_values('R2', ascending=False).head(18).round(4).to_string())

    print()
    print("=" * 100)
    print("TOP PREDICTORS for the 1% LEFT TAIL of next-day returns (mean pseudo-R2)")
    print("=" * 100)
    print(g.sort_values('q01', ascending=False).head(15).round(5).to_string())

    print()
    print("=" * 100)
    print("VARIANCE INFLATION FACTORS (mean across indices; >10 is a problem)")
    print("=" * 100)
    print(V.groupby('Predictor')['VIF'].mean().sort_values(ascending=False).round(2).to_string())

    print()
    print("=" * 100)
    print("MECHANICAL IDENTITIES - should be ~0 relative to median RV")
    print("=" * 100)
    print(pd.DataFrame(ident_rows).to_string(index=False))

    print()
    print("=" * 100)
    print("CROSS-INDEX CORRELATION of log RV (upper) and returns (lower), balanced days")
    print("=" * 100)
    print(f"  balanced log-RV days = {len(L)}, balanced return days = {len(R)}")
    print(cl.round(3).to_string())
    print()
    print(cr.round(3).to_string())
    ev = np.linalg.eigvalsh(cl.values)[::-1]
    print()
    print("  PCA on the log-RV correlation matrix, variance share:")
    print("   " + "  ".join(f"PC{i+1}={100*e/ev.sum():.1f}%" for i, e in enumerate(ev)))


if __name__ == "__main__":
    main()
