# -*- coding: utf-8 -*-
"""
Phase 8 / Risk HIGH #1: is the Dukascopy CFD a faithful proxy for the exchange index?

Intraday RV is built from a broker-quoted CFD; daily returns come from the actual index.
If those two instruments disagree, Realized GARCH is being fed a realized measure for a
DIFFERENT asset than the one whose returns it models. That is a fatal, and completely
avoidable, methodological hole - so we measure it explicitly.

-------------------------------------------------------------------------------------
THE ALIGNMENT RULE (added 2026-08-23 after a false alarm - read this before editing)
-------------------------------------------------------------------------------------
The CFD close-to-close return on day t is close(t)/close(t-1) where t-1 is the previous
session-day PRESENT IN THE CFD FILE. The index return on day t is a strict one-day return.
Whenever the CFD file is missing a day that the exchange traded (a Dukascopy feed hole, a
day dropped by the stale-bar filter, a half-day), the CFD "02_RAW_DAILY" return silently becomes a
TWO- or THREE-day return while the index return stays one day. Those observations are not
comparable and they dominate the variance of the residual.

Measured effect of ignoring this (BID side, 2011-2026):
        naive            prev-day aligned
  SPX   R2 0.9785   ->   R2 0.9898
  NDX   R2 0.9173   ->   R2 0.9966     <- would have been wrongly demoted
  UKX   R2 0.9087   ->   R2 0.9701     <- would have been wrongly demoted

So the headline verdict is computed on the ALIGNED sample: keep day t only when the CFD's
previous session-day equals the index's previous trading day. The naive number is still
reported alongside it, and Pct_Aligned tells you how much of the sample survived - a low
Pct_Aligned is itself a data-completeness warning even when R2 looks fine.

Pass criterion (aligned R2): > 0.99 PASS. 0.95-0.99 usable, must be disclosed in the paper.
Below 0.95 = demote the index.

Writes Datasets/08_VALIDATION/cfd_vs_index_check.csv
"""
import os, glob, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = os.path.join(ROOT, '08_VALIDATION')
os.makedirs(VAL, exist_ok=True)

CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]


def fit(x, y):
    """Return corr, slope, intercept, R2, annualised tracking error."""
    corr = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = float(1 - resid.var() / y.var())
    te = float(resid.std() * np.sqrt(252))
    return corr, float(slope), float(intercept), r2, te


rows = []
for code in CODES:
    rvp = os.path.join(ROOT, '06_REALIZED_MEASURES', f'{code}_RV_daily.csv')
    dly = glob.glob(os.path.join(ROOT, '02_RAW_DAILY', code, f'{code}_daily_*.csv'))
    if not os.path.exists(rvp) or not dly:
        rows.append(dict(Code=code, Status="SKIPPED - intraday or daily file missing"))
        continue

    rv = pd.read_csv(rvp, parse_dates=['Date']).sort_values('Date')
    dd = pd.read_csv(dly[0], parse_dates=['Date']).sort_values('Date')

    # previous available day in each series - the basis of the alignment rule
    rv['prev_cfd'] = rv['Date'].shift(1)
    dd['prev_idx'] = dd['Date'].shift(1)

    a = rv[['Date', 'CloseToClose_LogRet', 'prev_cfd']].rename(columns={'CloseToClose_LogRet': 'cfd_ret'})
    b = dd[['Date', 'LogReturn', 'prev_idx']].rename(columns={'LogReturn': 'idx_ret'})
    m = a.merge(b, on='Date', how='inner').dropna(subset=['cfd_ret', 'idx_ret'])
    # guard against a stray outlier from a half-day / data hole dominating the fit
    m = m[(m['cfd_ret'].abs() < 0.5) & (m['idx_ret'].abs() < 0.5)]

    if len(m) < 100:
        rows.append(dict(Code=code, N_Naive=len(m), Status="INSUFFICIENT OVERLAP"))
        continue

    al = m[m['prev_cfd'] == m['prev_idx']]
    if len(al) < 100:
        rows.append(dict(Code=code, N_Naive=len(m), N_Aligned=len(al),
                         Status="INSUFFICIENT ALIGNED OVERLAP - intraday file has too many holes"))
        continue

    _, _, _, r2_naive, _ = fit(m['idx_ret'].values, m['cfd_ret'].values)
    x, y = al['idx_ret'].values, al['cfd_ret'].values
    corr, slope, intercept, r2, te = fit(x, y)

    verdict = ("PASS" if r2 > 0.99 else
               "USABLE - disclose" if r2 > 0.95 else
               "FAIL - demote this index")

    rows.append(dict(Code=code,
                     N_Aligned=len(al), N_Naive=len(m),
                     Pct_Aligned=round(100.0 * len(al) / len(m), 2),
                     First=str(al['Date'].min().date()), Last=str(al['Date'].max().date()),
                     Correlation=round(corr, 5),
                     R2_Aligned=round(r2, 5), R2_Naive_DoNotUse=round(r2_naive, 5),
                     OLS_Slope=round(slope, 4),
                     OLS_Intercept_bps=round(intercept * 1e4, 3),
                     Tracking_Error_annual_pct=round(te * 100, 3),
                     Resid_SD_daily_bps=round(float(np.std(y - (slope * x + intercept)) * 1e4), 1),
                     CFD_vol_annual_pct=round(float(y.std() * np.sqrt(252) * 100), 2),
                     Index_vol_annual_pct=round(float(x.std() * np.sqrt(252) * 100), 2),
                     Status=verdict))
    print(f"{code}: n={len(al)} ({100.0*len(al)/len(m):.1f}% aligned) corr={corr:.4f} "
          f"R2={r2:.4f} (naive {r2_naive:.4f}) slope={slope:.3f} TE={te*100:.2f}%/yr -> {verdict}")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(VAL, 'cfd_vs_index_check.csv'), index=False)
print("\nwrote _validation/cfd_vs_index_check.csv")
if len(out):
    print(out.to_string(index=False))
