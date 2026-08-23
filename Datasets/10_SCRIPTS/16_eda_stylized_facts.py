# -*- coding: utf-8 -*-
"""
EDA STAGE 4 - distributional properties and the stylized facts of financial returns.

Every test here exists to justify, or to refuse to justify, a specific modelling choice
later. An EDA that reports moments without connecting them to the model is decoration; this
one is organised the other way round, by the decision each statistic informs.

  WHAT IS TESTED                       WHAT IT DECIDES
  moments, Jarque-Bera                 whether a Gaussian innovation is defensible at all
  Hill tail index, tail quantiles      whether EVT is needed, and where to put the POT
                                       threshold. A Hill alpha near 3-4 means the fourth
                                       moment may not exist, which is the whole case for
                                       GARCH-EVT over Gaussian or even Student-t GARCH.
  ADF / KPSS                           whether returns can be modelled in levels (they must
                                       be stationary) and whether log-RV is I(0) or needs
                                       fractional differencing
  Ljung-Box on r                       whether a conditional-mean term (AR) is needed before
                                       the variance model, or whether zero-mean is adequate
  Ljung-Box on r^2 and |r|             volatility clustering - the precondition for GARCH
  Engle ARCH-LM                        formal test that a conditional-variance model is
                                       warranted
  Engle-Ng sign-bias tests             whether asymmetry is present, i.e. whether GJR/EGARCH
                                       is required rather than plain GARCH. This is the test
                                       that decides the leverage specification.
  GPH long memory on log-RV            whether the realized-measure dynamics are long-range
                                       dependent, which is the case for HAR/Realized GARCH
                                       over a short-memory ARMA
  overnight variance share             how much of daily risk the session RV cannot see

Returns are tested on the FULL history (1990+) because the daily-only models estimate there.
Realized measures are tested on sample B where RV_Valid holds.

Outputs: _validation/eda4_moments.csv
         _validation/eda4_tests.csv
         _validation/eda4_tail.csv
"""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
VAL = os.path.join(ROOT, '08_VALIDATION')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]


def hill(x, tail='left', k_frac=0.05):
    """Hill estimator of the tail index alpha. Larger alpha = thinner tail.

    alpha <= 2 means infinite variance; alpha <= 4 means infinite kurtosis, which is the
    usual empirical finding for daily equity returns and the reason a Gaussian GARCH
    understates VaR at the 1% level.
    """
    v = np.sort(np.abs(x[x < 0])) if tail == 'left' else np.sort(x[x > 0])
    v = v[v > 0]
    n = len(v)
    if n < 50:
        return np.nan, np.nan, 0
    k = max(20, int(k_frac * n))
    k = min(k, n - 1)
    top = v[-(k + 1):]
    thr = top[0]
    alpha = 1.0 / np.mean(np.log(top[1:] / thr))
    se = alpha / np.sqrt(k)
    return alpha, se, k


def gph(x, power=0.6):
    """Geweke-Porter-Hudak semiparametric estimate of the fractional integration order d.

    d in (0, 0.5) is stationary long memory. Realized volatility is famously in that band
    (around 0.35-0.45), which is what the HAR cascade approximates.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 200:
        return np.nan, np.nan
    m = int(n ** power)
    xd = x - x.mean()
    per = np.abs(np.fft.fft(xd)) ** 2 / (2 * np.pi * n)
    lam = 2 * np.pi * np.arange(1, m + 1) / n
    y = np.log(per[1:m + 1])
    xr = np.log(4 * np.sin(lam / 2) ** 2)
    X = sm.add_constant(-xr)
    res = sm.OLS(y, X).fit()
    return float(res.params[1]), float(res.bse[1])


def engle_ng(r):
    """Engle-Ng (1993) joint sign-bias test on squared standardised residuals.

    Regresses z^2 on a negative-return dummy, and on that dummy interacted with the lagged
    return in both directions. A significant joint statistic says negative and positive
    shocks move variance differently - i.e. plain GARCH is misspecified and GJR or EGARCH
    is required. This is a specification decision, not a descriptive statistic.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    # crude standardisation by a rolling sd so the test is about ASYMMETRY, not level
    s = pd.Series(r).rolling(22, min_periods=10).std().shift(1)
    z = (pd.Series(r) / s).dropna()
    z2 = (z ** 2).values[1:]
    lag = z.values[:-1]
    neg = (lag < 0).astype(float)
    X = np.column_stack([np.ones_like(z2), neg, neg * lag, (1 - neg) * lag])
    res = sm.OLS(z2, X).fit()
    f = res.f_test(np.eye(4)[1:])
    return float(f.fvalue), float(f.pvalue)


def describe(x, label, code, scope):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return None
    q = np.percentile(x, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
    jb, jbp = stats.jarque_bera(x)
    return dict(Code=code, Series=label, Scope=scope, N=len(x),
                Mean=x.mean(), SD=x.std(ddof=1),
                Skew=float(stats.skew(x)), ExcessKurt=float(stats.kurtosis(x)),
                Min=x.min(), P0_1=q[0], P1=q[1], P5=q[2], P25=q[3], Median=q[4],
                P75=q[5], P95=q[6], P99=q[7], P99_9=q[8], Max=x.max(),
                JarqueBera=float(jb), JB_p=float(jbp))


def main():
    moments, tests, tails = [], [], []
    for c in CODES:
        a = pd.read_csv(os.path.join(ANA, f'{c}_analysis.csv'), parse_dates=['Date'])
        rall = a['Return'].dropna()
        b = a[a['InSample_B']].copy()
        rb = b['Return'].dropna()
        rv = b.loc[b['RV_Valid'], 'RV'].dropna()
        lrv = b.loc[b['RV_Valid'], 'LogRV'].dropna()

        for lab, s, sc in (('Return', rall, 'full 1990+'),
                           ('Return', rb, 'sample B'),
                           ('RV', rv, 'sample B valid'),
                           ('LogRV', lrv, 'sample B valid'),
                           ('VolIdx', b['VolIdx'].dropna(), 'sample B'),
                           ('RSV_Ratio', b['RSV_Ratio'].dropna(), 'sample B valid'),
                           ('JumpShare', b['JumpShare'].dropna(), 'sample B valid')):
            d = describe(s, lab, c, sc)
            if d:
                moments.append(d)

        # ---------- tail behaviour ----------
        for scope, s in (('full 1990+', rall), ('sample B', rb)):
            for side in ('left', 'right'):
                for kf in (0.05, 0.10):
                    al, se, k = hill(s.values, side, kf)
                    tails.append(dict(Code=c, Scope=scope, Tail=side, k_frac=kf, k=k,
                                      Hill_Alpha=al, SE=se,
                                      Implied_MaxFiniteMoment=np.floor(al) if np.isfinite(al) else np.nan))

        # ---------- tests on returns ----------
        row = dict(Code=c)
        rv_full = rall.values
        row['ADF_Return_stat'], row['ADF_Return_p'] = adfuller(rv_full, autolag='AIC')[:2]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            k_stat, k_p = kpss(rv_full, regression='c', nlags='auto')[:2]
        row['KPSS_Return_stat'], row['KPSS_Return_p'] = k_stat, k_p
        lb = acorr_ljungbox(rv_full, lags=[10, 22], return_df=True)
        row['LB10_Return_p'] = float(lb['lb_pvalue'].iloc[0])
        row['LB22_Return_p'] = float(lb['lb_pvalue'].iloc[1])
        lb2 = acorr_ljungbox(rv_full ** 2, lags=[10, 22], return_df=True)
        row['LB10_RetSq_p'] = float(lb2['lb_pvalue'].iloc[0])
        row['LB22_RetSq_p'] = float(lb2['lb_pvalue'].iloc[1])
        lba = acorr_ljungbox(np.abs(rv_full), lags=[22], return_df=True)
        row['LB22_AbsRet_p'] = float(lba['lb_pvalue'].iloc[0])
        arch_stat, arch_p = het_arch(rv_full, nlags=10)[:2]
        row['ARCH_LM10_stat'], row['ARCH_LM10_p'] = float(arch_stat), float(arch_p)
        f, p = engle_ng(rv_full)
        row['EngleNg_F'], row['EngleNg_p'] = f, p

        # ---------- tests on the realized measure ----------
        if len(lrv) > 300:
            row['ADF_LogRV_p'] = adfuller(lrv.values, autolag='AIC')[1]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                row['KPSS_LogRV_p'] = kpss(lrv.values, regression='c', nlags='auto')[1]
            d_gph, d_se = gph(lrv.values)
            row['GPH_d_LogRV'], row['GPH_d_SE'] = d_gph, d_se
            row['AC1_LogRV'] = float(pd.Series(lrv.values).autocorr(1))
            row['AC22_LogRV'] = float(pd.Series(lrv.values).autocorr(22))
            row['AC66_LogRV'] = float(pd.Series(lrv.values).autocorr(66))

        # ---------- leverage / asymmetry in the realized measure ----------
        v = b[b['RV_Valid']].copy()
        v['LogRV_next'] = v['LogRV'].shift(-1)
        vv = v.dropna(subset=['Return', 'LogRV_next'])
        if len(vv) > 200:
            row['Leverage_corr_r_LogRVnext'] = float(np.corrcoef(vv['Return'], vv['LogRV_next'])[0, 1])
            row['Corr_RSneg_LogRVnext'] = float(np.corrcoef(
                np.log(vv['RS_neg'].where(vv['RS_neg'] > 0).fillna(np.nan).dropna()),
                vv.loc[vv['RS_neg'] > 0, 'LogRV_next'])[0, 1]) if (vv['RS_neg'] > 0).sum() > 200 else np.nan

        # ---------- overnight variance share ----------
        w = b.dropna(subset=['Return', 'RV'])
        if len(w) > 200:
            row['VarShare_Session_Pct'] = 100 * float(w['RV'].mean() / w['Return_Sq'].mean())
            row['ScaleFactor_HL'] = float(a['ScaleFactor_HL'].iloc[0])
        tests.append(row)
        print(f"  [{c}] done")

    pd.DataFrame(moments).to_csv(os.path.join(VAL, 'eda4_moments.csv'), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(VAL, 'eda4_tests.csv'), index=False)
    pd.DataFrame(tails).to_csv(os.path.join(VAL, 'eda4_tail.csv'), index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 60)
    M = pd.DataFrame(moments)
    print()
    print("=" * 110)
    print("MOMENTS - daily log returns, full history")
    print("=" * 110)
    show = ['Code', 'N', 'Mean', 'SD', 'Skew', 'ExcessKurt', 'Min', 'P1', 'P99', 'Max', 'JB_p']
    print(M[(M.Series == 'Return') & (M.Scope == 'full 1990+')][show].round(5).to_string(index=False))
    print()
    print("=" * 110)
    print("MOMENTS - log realized variance, sample B")
    print("=" * 110)
    print(M[(M.Series == 'LogRV')][['Code', 'N', 'Mean', 'SD', 'Skew', 'ExcessKurt',
                                    'JB_p']].round(4).to_string(index=False))
    T = pd.DataFrame(tests)
    print()
    print("=" * 110)
    print("SPECIFICATION TESTS on returns (p-values)")
    print("=" * 110)
    print(T[['Code', 'ADF_Return_p', 'KPSS_Return_p', 'LB10_Return_p', 'LB22_RetSq_p',
             'LB22_AbsRet_p', 'ARCH_LM10_p', 'EngleNg_p']].round(5).to_string(index=False))
    print()
    print("=" * 110)
    print("REALIZED MEASURE dynamics")
    print("=" * 110)
    print(T[['Code', 'ADF_LogRV_p', 'KPSS_LogRV_p', 'GPH_d_LogRV', 'GPH_d_SE',
             'AC1_LogRV', 'AC22_LogRV', 'AC66_LogRV',
             'Leverage_corr_r_LogRVnext', 'VarShare_Session_Pct']].round(4).to_string(index=False))
    print()
    print("=" * 110)
    print("TAIL INDEX (Hill), k = 5% of tail observations")
    print("=" * 110)
    Tl = pd.DataFrame(tails)
    print(Tl[(Tl.k_frac == 0.05)][['Code', 'Scope', 'Tail', 'k', 'Hill_Alpha', 'SE']]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
