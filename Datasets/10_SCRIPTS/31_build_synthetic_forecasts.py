# -*- coding: utf-8 -*-
"""
SYNTHETIC PLACEHOLDER FORECASTS — contract-format files with no modelling content.

WHY THIS EXISTS
  Researcher B's evaluation code (QLIKE, MSE, VaR backtests, DM, MCS) can and should be
  written and unit-tested BEFORE Researcher A's real GARCH-family output exists, so the two
  workstreams run in parallel rather than in sequence. These files exist only so B's code has
  something in the exact 26_forecast_io.py contract shape to develop against.

  The numbers are a naive constant-volatility placeholder (rolling 21-day realised std of
  Return, no asymmetry, no fitted parameters) - deliberately unsophisticated, so nobody
  mistakes a synthetic file for a real result. `Spec` is stamped "SYNTHETIC_PLACEHOLDER" and
  every row's `Reason` says so.

DO NOT cite, evaluate, or report numbers from these files. Delete or ignore them once the
real forecast files (GJR-skewt__*, RealGARCH__*, ...) exist for the same (model, code) pair.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
fio = importlib.import_module('26_forecast_io')

SYN_DIR = os.path.join(fio.FCDIR, '_SYNTHETIC')


def build(code):
    a = pd.read_csv(os.path.join(ANA, f'{code}_analysis.csv'), parse_dates=['Date'], low_memory=False)
    a = a.sort_values('Date').reset_index(drop=True)
    a = a[a['InSample_B']].reset_index(drop=True)

    sigma = a['Return'].rolling(21, min_periods=10).std().shift(1)  # lagged, no look-ahead
    out = pd.DataFrame({'Date': a['Date'], 'OriginDate': a['Date'].shift(1)})
    out['SigmaHat'] = sigma
    out['VarHat'] = sigma ** 2
    nu = 6.0  # arbitrary fixed placeholder
    c = np.sqrt(nu / (nu - 2.0))
    for k, tau in fio.LEVELS:
        q = stats.t.ppf(tau, df=nu) / c
        out[f'VaR_{k}'] = sigma * q
    out['Realized'] = a['Return']
    out['RVProxy'] = np.where(a['RV_Valid'].astype(bool), a['RV_Scaled'], np.nan)
    out['Valid'] = out['SigmaHat'].notna() & out['OriginDate'].notna()
    out['Reason'] = np.where(out['Valid'], 'SYNTHETIC_PLACEHOLDER_do_not_cite', 'insufficient_history')
    return out


def main():
    os.makedirs(SYN_DIR, exist_ok=True)
    for code in CODES:
        df = build(code)
        path = fio.write_forecasts(df, model="PLACEHOLDER", code=code,
                                    base=SYN_DIR, spec="SYNTHETIC_PLACEHOLDER_21d_rolling_std")
        print(f"[{code}] wrote {path}  ({int(df['Valid'].sum())} valid rows)")
    readme = os.path.join(SYN_DIR, 'README.txt')
    with open(readme, 'w') as f:
        f.write(
            "These files are NOT model output. They exist so Researcher B's evaluation code\n"
            "(QLIKE, MSE, VaR backtests, DM, MCS) can be written and unit-tested against the\n"
            "exact forecast-file contract (see 10_SCRIPTS/26_forecast_io.py) before Researcher\n"
            "A's real GARCH-family forecasts exist. Do not cite or evaluate these numbers -\n"
            "SigmaHat here is nothing more than a 21-day trailing return std, lagged one day.\n"
            "Swap in 20_FORECASTS/<REAL_MODEL>__<CODE>_forecasts.csv as soon as they exist.\n")
    print(f"wrote {readme}")


if __name__ == "__main__":
    main()
