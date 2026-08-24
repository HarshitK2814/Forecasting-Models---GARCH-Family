# -*- coding: utf-8 -*-
"""
WINDOW-LENGTH AND FORECAST-HORIZON ROBUSTNESS — closing two items the Executive Summary names
explicitly and that were not yet built:

  Section 4.1: "Re-fit each model on data up to t-1 (WITH FIXED WINDOW OR EXPANDING WINDOW)."
    29_rolling_forecast_engine.py only ever ran expanding. This script adds the fixed-window
    alternative and compares it against expanding, so the choice is demonstrated, not asserted.

  "Evaluation & Robustness": "Window length: try different rolling window sizes (e.g. 2 vs 5
    years)... Horizon: if applicable, extend forecasts to 5-day ahead and compare results."

Both checks reuse 29_rolling_forecast_engine.run_index_spec(), which now accepts window_size
(None=expanding, int=fixed trailing window) and horizon (default 1, cumulative H-day forecast
for horizon>1) - see that module's docstring for exactly what "H-day forecast" means and the
approximation it makes (same fitted shape parameters, scaled by the H-day aggregate sigma).

PART A - WINDOW LENGTH (SPX only, matching the cost-bounded precedent set by the refit-cadence
  check in 30_robustness_checks.py - a full 6-index x 3-window re-run is not warranted for a
  robustness demonstration).
  Three regimes: expanding (production default), fixed 2-year (504 trading days), fixed 5-year
  (1260 trading days) - the doc's own "2 vs 5 years" suggestion.

PART B - FORECAST HORIZON (all six indices, GJR-skewt, horizon=5).
  Written as genuine contract-format forecast files (Model="GJR-skewt-h5") alongside the
  production horizon=1 files, so they are usable by B's evaluation code exactly like any other
  forecast file - not just a diagnostic side-table.

OUTPUTS
  08_VALIDATION/robustness_window_length_sensitivity.csv
  08_VALIDATION/robustness_horizon_extension.csv
  20_FORECASTS/GJR-skewt-h5__<CODE>_forecasts.csv   (6 files, Horizon=5)
  09_FIGURES/18_window_length_sensitivity.png
  09_FIGURES/19_horizon_extension.png
  11_LOGS/phase25_window_horizon.log
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = os.path.join(ROOT, '08_VALIDATION')
FIG = os.path.join(ROOT, '09_FIGURES')
LOG = os.path.join(ROOT, '11_LOGS')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
engine = importlib.import_module('29_rolling_forecast_engine')
fio = importlib.import_module('26_forecast_io')

WINDOWS = [("expanding", None), ("fixed_2y", 504), ("fixed_5y", 1260)]


# ---------------------------------------------------------------------------
# Part A - window length
# ---------------------------------------------------------------------------
def check_window_length(code="SPX", spec="GJR-skewt"):
    series = {}
    rows = []
    for label, wsize in WINDOWS:
        t0 = time.time()
        df, refits, elapsed = engine.run_index_spec(code, spec, window_size=wsize)
        df['Date'] = pd.to_datetime(df['Date'])
        series[label] = df.set_index('Date')['SigmaHat']
        rows.append(dict(Code=code, Window=label, WindowDays=wsize if wsize else -1,
                          N=len(df), MeanSigmaHat=float(df['SigmaHat'].mean()), Elapsed_s=elapsed))
        print(f"    {label:12s} N={len(df)}  mean sigma={df['SigmaHat'].mean():.5f}  {elapsed:.1f}s")

    joined = pd.concat(series, axis=1, join='inner')
    base = joined['expanding']
    for label, _ in WINDOWS[1:]:
        rel = (joined[label] - base).abs() / base
        for r in rows:
            if r['Window'] == label:
                r['MeanAbsRelDiff_vs_expanding'] = float(rel.mean())
                r['MaxAbsRelDiff_vs_expanding'] = float(rel.max())
                r['Corr_vs_expanding'] = float(base.corr(joined[label]))
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(VAL, 'robustness_window_length_sensitivity.csv'), index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    colors = {'expanding': 'steelblue', 'fixed_2y': 'darkorange', 'fixed_5y': 'seagreen'}
    for label, _ in WINDOWS:
        ax1.plot(joined.index, joined[label] * 100, color=colors[label], linewidth=0.8,
                  label=label, alpha=0.85)
    ax1.set_ylabel('Daily sigma-hat, %')
    ax1.set_title(f'{code} {spec}: expanding vs fixed 2y vs fixed 5y rolling window')
    ax1.legend()
    diff2y = (joined['fixed_2y'] - joined['expanding']) * 100
    diff5y = (joined['fixed_5y'] - joined['expanding']) * 100
    ax2.plot(joined.index, diff2y, color=colors['fixed_2y'], linewidth=0.6, label='fixed_2y - expanding')
    ax2.plot(joined.index, diff5y, color=colors['fixed_5y'], linewidth=0.6, label='fixed_5y - expanding')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('pp')
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, '18_window_length_sensitivity.png'), dpi=140)
    plt.close(fig)
    print("  wrote 09_FIGURES/18_window_length_sensitivity.png")
    return df_out


# ---------------------------------------------------------------------------
# Part B - horizon extension
# ---------------------------------------------------------------------------
def build_horizon_forecasts(spec="GJR-skewt", horizon=5):
    rows = []
    series_1d, series_5d = {}, {}
    for code in CODES:
        print(f"  [{code}] horizon={horizon} walk-forward ...")
        df1, _, e1 = engine.run_index_spec(code, spec, horizon=1)
        df5, _, e5 = engine.run_index_spec(code, spec, horizon=horizon)
        path = fio.write_forecasts(df5, model=f"{spec}-h{horizon}", code=code,
                                    spec=f"{spec}-AR1-expanding-refit21-horizon{horizon}")
        print(f"    -> wrote {path} ({len(df5)} rows, contract-validated)")

        ev1 = fio.eval_frame(fio.read_forecasts(
            os.path.join(fio.FCDIR, f"{spec}__{code}_forecasts.csv")))
        ev5 = fio.eval_frame(fio.read_forecasts(path))
        rows.append(dict(Code=code, N_1d=len(ev1), N_5d=len(ev5),
                          QLIKE_1d=float(ev1['QLIKE'].mean()), QLIKE_5d=float(ev5['QLIKE'].mean()),
                          MeanSigma_1d_ann=float(df1['SigmaHat'].mean() * np.sqrt(252) * 100),
                          MeanSigma_5d_ann=float(df5['SigmaHat'].mean() * np.sqrt(252 / horizon) * 100)))

        df1['Date'] = pd.to_datetime(df1['Date'])
        df5['Date'] = pd.to_datetime(df5['Date'])
        series_1d[code] = df1.set_index('Date')['SigmaHat'] * np.sqrt(252) * 100
        series_5d[code] = df5.set_index('Date')['SigmaHat'] * np.sqrt(252 / horizon) * 100

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(VAL, 'robustness_horizon_extension.csv'), index=False)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, code in zip(axes.flat, CODES):
        ax.plot(series_1d[code].index, series_1d[code].values, color='steelblue',
                 linewidth=0.6, label='1-day-ahead (annualised)')
        ax.plot(series_5d[code].index, series_5d[code].values, color='darkorange',
                 linewidth=0.6, alpha=0.8, label='5-day cumulative (annualised)')
        ax.set_title(code, fontsize=10, fontweight='bold')
        ax.set_ylabel('Annualised vol, %')
        ax.legend(fontsize=6.5)
    fig.suptitle(f'{spec}: 1-day vs {horizon}-day-ahead forecasts, annualised for comparability', y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, '19_horizon_extension.png'), dpi=140)
    plt.close(fig)
    print("  wrote 09_FIGURES/19_horizon_extension.png")
    return df_out


def main():
    os.makedirs(VAL, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(LOG, exist_ok=True)
    t0 = time.time()
    log_lines = [f"phase25 window/horizon robustness — {pd.Timestamp.now()}", ""]

    print("PART A: window-length sensitivity (SPX, GJR-skewt) ...")
    a = check_window_length()
    print(a.to_string(index=False))
    log_lines.append("window length:\n" + a.to_string(index=False))

    print("\nPART B: horizon extension, all six indices ...")
    b = build_horizon_forecasts()
    print(b.to_string(index=False))
    log_lines.append("\nhorizon extension:\n" + b.to_string(index=False))

    with open(os.path.join(LOG, 'phase25_window_horizon.log'), 'w') as f:
        f.write("\n".join(log_lines) + f"\n\ndone in {time.time()-t0:.1f}s\n")

    print(f"\nDone in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
