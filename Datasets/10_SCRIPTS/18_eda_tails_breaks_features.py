# -*- coding: utf-8 -*-
"""
EDA STAGE 6 - EVT threshold diagnostics, volatility regimes, and the final feature set.

THREE JOBS

  A. RESOLVE THE COLLINEARITY FOUND IN STAGE 5.
     LogRV and LogRS_neg came back with VIFs near 95. That is not a coincidence to be
     tuned away, it is an identity: RS_pos + RS_neg = RV exactly, and the downside share sits
     tightly around 0.50 for every index, so log RS_neg is log RV plus an almost constant.
     The fix is to replace the redundant pair with an orthogonal parameterisation - the LEVEL
     (log RV) and the SHARE (RSV_Ratio, or the signed jump) - rather than to drop a variable
     and pretend the information is gone. Both parameterisations are scored here.

  B. EVT THRESHOLD DIAGNOSTICS.
     GARCH-EVT stands or falls on the peaks-over-threshold choice. Too low a threshold and
     the GPD asymptotics do not hold; too high and the shape parameter is estimated from
     twenty points. The standard diagnostic is a stability plot: fit the GPD across a range
     of thresholds and look for the region where the shape parameter xi stops drifting.
     That is done here numerically rather than by eye, together with the exceedance counts
     that determine whether each threshold is even estimable.

     NOTE ON WHAT THIS IS AND IS NOT. McNeil-Frey applies EVT to the STANDARDISED RESIDUALS
     of a fitted GARCH, not to raw returns, because raw returns are not iid and the GPD
     limit theory assumes they are. The GARCH fit does not exist yet at the EDA stage, so
     these diagnostics are run on raw returns AND on returns standardised by a simple
     rolling-window volatility. The rolling-standardised version is the closer analogue and
     is the one to read; the raw version is shown so the difference is visible.

  C. VOLATILITY REGIMES AND CRISIS COVERAGE.
     An EVT threshold estimated on a sample containing one crisis is fragile. This reports
     the realised volatility timeline so the sample-B window can be judged on whether it
     contains enough distinct stress episodes to identify a tail.

Outputs: _validation/eda6_gpd_threshold.csv
         _validation/eda6_vif_final.csv
         _validation/eda6_regimes.csv
         _validation/eda6_extremes.csv
"""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
VAL = os.path.join(ROOT, '08_VALIDATION')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

# two competing parameterisations of the realized-measure block
SET_REDUNDANT = ['LogRV', 'LogRS_neg', 'LogRV_w', 'LogRV_m', 'LogIV', 'VRP',
                 'NegReturn', 'RangePct', 'CreditStress']
SET_ORTHOGONAL = ['LogRV', 'RSV_Ratio', 'LogRV_w', 'LogRV_m', 'JumpShare', 'RSkew',
                  'LogIV', 'VRP', 'NegReturn', 'RangePct', 'TermSpread_diff',
                  'CreditStress', 'DXY_ret']


def vif_table(b, cols, code, tag):
    cols = [c for c in cols if c in b.columns]
    W = b[cols].replace([np.inf, -np.inf], np.nan).dropna()
    out = []
    if len(W) < 300:
        return out
    X = sm.add_constant(W.values)
    for i, name in enumerate(cols):
        out.append(dict(Code=code, Set=tag, Predictor=name,
                        VIF=float(variance_inflation_factor(X, i + 1))))
    return out


def gpd_stability(x, quantiles):
    """Fit a GPD to exceedances over a range of thresholds and report the shape parameter.

    x is the LOSS series (positive numbers = losses), so the left tail of returns is passed
    in as -r. The threshold region to use is where xi is flat and the exceedance count is
    still large enough for the estimate to mean anything - conventionally at least 100.
    """
    out = []
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    for q in quantiles:
        u = np.quantile(x, q)
        exc = x[x > u] - u
        n = len(exc)
        if n < 30:
            out.append(dict(q=q, u=u, n_exc=n, xi=np.nan, beta=np.nan, xi_se=np.nan))
            continue
        try:
            xi, loc, beta = stats.genpareto.fit(exc, floc=0)
            # asymptotic standard error of xi under the GPD MLE
            xi_se = np.sqrt((1 + xi) ** 2 / n)
            out.append(dict(q=q, u=u, n_exc=n, xi=xi, beta=beta, xi_se=xi_se))
        except Exception:
            out.append(dict(q=q, u=u, n_exc=n, xi=np.nan, beta=np.nan, xi_se=np.nan))
    return out


def main():
    vif_rows, gpd_rows, reg_rows, ext_rows = [], [], [], []
    QS = [0.80, 0.85, 0.90, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.99]

    for c in CODES:
        a = pd.read_csv(os.path.join(ANA, f'{c}_analysis.csv'), parse_dates=['Date'])
        a['TermSpread_diff'] = a['TermSpread_pct'].diff()
        b = a[a['InSample_B']].copy().reset_index(drop=True)

        # ---------- A. collinearity ----------
        vif_rows += vif_table(b, SET_REDUNDANT, c, 'redundant')
        vif_rows += vif_table(b, SET_ORTHOGONAL, c, 'orthogonal')

        # ---------- B. GPD threshold stability ----------
        r_full = a['Return'].dropna().values
        # rolling-standardised returns: the closer analogue to GARCH residuals
        s = a['Return'].rolling(252, min_periods=100).std().shift(1)
        z = (a['Return'] / s).replace([np.inf, -np.inf], np.nan).dropna().values
        for label, series in (('raw_return_full', r_full),
                              ('rolling_std_resid_full', z)):
            for tail, sgn in (('left', -1.0), ('right', 1.0)):
                for row in gpd_stability(sgn * series, QS):
                    gpd_rows.append(dict(Code=c, Series=label, Tail=tail, **row))

        # ---------- C. volatility regimes ----------
        a['RollVol_Ann'] = 100 * np.sqrt(252) * a['Return'].rolling(63, min_periods=40).std()
        yb = b.copy()
        yb['Year'] = yb['Date'].dt.year
        for y, g in yb.groupby('Year'):
            reg_rows.append(dict(
                Code=c, Year=int(y), N=len(g),
                Vol_Ann_Pct=round(100 * np.sqrt(252) * g['Return'].std(), 2),
                Median_RVol_Ann=round(float(g['RVol_Ann_Pct'].median()), 2)
                if g['RVol_Ann_Pct'].notna().any() else np.nan,
                Median_VolIdx=round(float(g['VolIdx'].median()), 2),
                Worst_Return_Pct=round(100 * float(g['Return'].min()), 2),
                N_Exceed_m3pct=int((g['Return'] < -0.03).sum())))

        # ---------- worst days, for the record ----------
        w = a.dropna(subset=['Return']).nsmallest(10, 'Return')
        for _, r in w.iterrows():
            ext_rows.append(dict(Code=c, Date=r['Date'].date(),
                                 Return_Pct=round(100 * r['Return'], 2),
                                 VolIdx=r['VolIdx'],
                                 RVol_Ann_Pct=round(r['RVol_Ann_Pct'], 1)
                                 if np.isfinite(r['RVol_Ann_Pct']) else np.nan,
                                 InSampleB=bool(r['InSample_B'])))
        print(f"  [{c}] done")

    V = pd.DataFrame(vif_rows)
    V.to_csv(os.path.join(VAL, 'eda6_vif_final.csv'), index=False)
    G = pd.DataFrame(gpd_rows)
    G.to_csv(os.path.join(VAL, 'eda6_gpd_threshold.csv'), index=False)
    R = pd.DataFrame(reg_rows)
    R.to_csv(os.path.join(VAL, 'eda6_regimes.csv'), index=False)
    E = pd.DataFrame(ext_rows)
    E.to_csv(os.path.join(VAL, 'eda6_extremes.csv'), index=False)

    pd.set_option('display.width', 250)
    print()
    print("=" * 92)
    print("A. VIF - redundant vs orthogonal parameterisation (mean across indices)")
    print("=" * 92)
    for tag in ('redundant', 'orthogonal'):
        sub = V[V['Set'] == tag].groupby('Predictor')['VIF'].mean().sort_values(ascending=False)
        print(f"  --- {tag} ---")
        print("    " + sub.round(2).to_string().replace("\n", "\n    "))
        print(f"    MAX VIF = {sub.max():.1f}")

    print()
    print("=" * 92)
    print("B. GPD shape parameter xi by threshold - LEFT tail, rolling-standardised returns")
    print("   (look for the region where xi stops drifting and n_exc is still >= 100)")
    print("=" * 92)
    sub = G[(G['Series'] == 'rolling_std_resid_full') & (G['Tail'] == 'left')]
    piv = sub.pivot_table(index='q', columns='Code', values='xi')
    cnt = sub.pivot_table(index='q', columns='Code', values='n_exc')
    print("  xi:")
    print(piv.round(3).to_string())
    print()
    print("  exceedance counts:")
    print(cnt.astype(int).to_string())

    print()
    print("  Same for RAW returns (shown for contrast - not the series EVT is applied to):")
    sub2 = G[(G['Series'] == 'raw_return_full') & (G['Tail'] == 'left')]
    print(sub2.pivot_table(index='q', columns='Code', values='xi').round(3).to_string())

    print()
    print("=" * 92)
    print("C. Volatility by year inside sample B (annualised %, from daily returns)")
    print("=" * 92)
    print(R.pivot_table(index='Year', columns='Code', values='Vol_Ann_Pct').round(1).to_string())
    print()
    print("  days worse than -3% per year (tail events available to the estimator):")
    print(R.pivot_table(index='Year', columns='Code',
                        values='N_Exceed_m3pct').fillna(0).astype(int).to_string())

    print()
    print("=" * 92)
    print("D. Ten worst days per index (full history)")
    print("=" * 92)
    for c in CODES:
        s = E[E['Code'] == c].head(5)
        print(f"  {c}: " + " | ".join(
            f"{r.Date} {r.Return_Pct:+.1f}%{'*' if r.InSampleB else ''}"
            for r in s.itertuples()))
    print("  (* = inside sample B)")


if __name__ == "__main__":
    main()
