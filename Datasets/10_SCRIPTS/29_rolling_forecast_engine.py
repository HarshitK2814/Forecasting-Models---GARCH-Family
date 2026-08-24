# -*- coding: utf-8 -*-
"""
ROLLING OUT-OF-SAMPLE FORECAST ENGINE — GARCH-family models.

Researcher A, plan item "Rolling out-of-sample engine", 24h.

DESIGN
  Two things are conflated in naive "rolling forecast" code and kept separate here:
    (a) RE-ESTIMATION of parameters  - expensive (numerical optimisation)
    (b) STATE UPDATE of the variance recursion h_t - cheap (one recursion step)
  Refitting parameters every trading day is standard in small demos but is not what the
  literature does at scale (window-selection studies use periodic refits - see e.g. the
  `rugarch::ugarchroll` `refit.every` argument, and Feng (2024, J. Forecasting) on rolling vs
  expanding windows). Parameters are re-estimated every REFIT_EVERY trading days on an
  EXPANDING window (chosen over a fixed rolling window because the EDA found GPH d=0.50-0.63,
  i.e. long memory in volatility - a fixed window forgets exactly the information a
  long-memory process still needs). Between refits, the conditional-variance recursion is
  still updated every day using the REAL, newly observed return - so every forecast is a
  genuine 1-step-ahead forecast, using only information available the prior close. Only the
  parameter VALUES are frozen between refit dates, not the state.

  This is implemented with `arch`'s `.fix(params)`: build the model on data through day t-1,
  fix it at the last re-estimated parameter vector, and call `.forecast(horizon=1)`. No
  optimisation runs on the daily step, so this is fast (~1-2 ms/day measured), and the
  refit itself (~50-150 ms) runs only every REFIT_EVERY days.

WHAT IS FORECAST, AND WHERE THE EVALUATION WINDOW STARTS
  Burn-in: all data from the index's own return-history start up to SAMPLE_B_START
  (2013-09-30, see RESEARCHER_A_DECISIONS.md section 3) is used ONLY to produce the first
  fit; no forecast row is written for it. From SAMPLE_B_START onward, every trading day gets
  a genuine walk-forward 1-step-ahead forecast - this is what "only the forecast window must
  match the other models" (FEATURE_SETS.csv) means in practice.

SPECS RUN
  Only PRIMARY_SPEC (GJR-skewt, matching FEATURE_SETS.csv's "AR(1)-GJR-GARCH" and
  27_baseline_garch.py's spec selection) is run by default, to keep the walk-forward cost
  bounded within one script invocation. Add strings to RUN_SPECS to widen the comparison -
  everything from the fit and quantile-extraction machinery is spec-agnostic.

REALIZED GARCH IS NOT WALK-FORWARD RE-ESTIMATED HERE
  The Realized GARCH optimiser in 28_realized_garch.py takes ~85s per full-sample fit (custom
  quasi-MLE, no closed-form gradient, Nelder-Mead). A walk-forward re-estimation at the same
  monthly cadence used here would be ~130 refits x 6 indices x 85s =~ 18 hours - not run as
  part of this delivery. 28_realized_garch.py instead reports the full-sample-parameter,
  daily-recursive 1-step-ahead series, which is look-ahead-free in the conditional-variance
  STATE (h_t only uses x_{t-1}, h_{t-1}) but uses PARAMETERS estimated on the whole sample -
  the standard "in-sample GARCH" convention, identical to how 27_baseline_garch.py's output is
  produced, and clearly weaker than genuine walk-forward re-estimation. This gap is recorded
  as an open item for the robustness-check budget (or an overnight batch job with a coarser,
  e.g. quarterly or annual, refit cadence) rather than silently presented as equivalent.

OUTPUT
  20_FORECASTS/<SPEC>__<CODE>_forecasts.csv     contract-format (26_forecast_io.py)
  08_VALIDATION/rolling_engine_refit_log.csv    every refit: date, params, N used
  11_LOGS/phase21_rolling_engine.log
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from arch import arch_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
VAL = os.path.join(ROOT, '08_VALIDATION')
LOG = os.path.join(ROOT, '11_LOGS')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
fio = importlib.import_module('26_forecast_io')
LEVELS = fio.LEVELS

SAMPLE_B_START = pd.Timestamp('2013-09-30')
REFIT_EVERY = 21   # ~1 trading month; see docstring for the rationale

SPEC_DEFS = {
    "GARCH-t":      dict(vol="GARCH",  p=1, o=0, q=1, dist="t"),
    "GJR-skewt":    dict(vol="GARCH",  p=1, o=1, q=1, dist="skewt"),
    "EGARCH-skewt": dict(vol="EGARCH", p=1, o=1, q=1, dist="skewt"),
}
RUN_SPECS = ["GJR-skewt"]   # widen here if compute budget allows; see docstring


def run_index_spec(code, spec_name, window_size=None, horizon=1):
    """Walk-forward one (index, spec).

    window_size: None (default) = EXPANDING window, the production choice (see docstring for
        why: GPH d=0.50-0.63 long memory means a fixed window forgets what it still needs).
        An integer = FIXED ROLLING window of that many trading days - the doc's Section 4.1
        names "fixed window or expanding window" explicitly as the two options; this parameter
        makes the alternative buildable and comparable rather than only asserted in prose.

    horizon: 1 (default) = the production 1-step-ahead forecast used everywhere else in the
        pipeline. An integer >1 produces a CUMULATIVE H-day forecast: SigmaHat becomes the
        conditional SD of the H-day-ahead cumulative log return (sqrt of the sum of the H
        one-step-ahead variances - the standard GARCH multi-step aggregation, since returns are
        conditionally uncorrelated in this mean specification), and mu is the sum of the H
        one-step-ahead conditional means. VaR/ES quantiles reuse the SAME fitted shape
        parameters (nu, lambda) scaled by the H-day sigma - an approximation (the true H-day
        return distribution is a convolution of H marginals with time-varying variance, not
        itself exactly skew-t), stated explicitly rather than left implicit. This is the
        doc's Section "Evaluation & Robustness" horizon-extension item.
    """
    spec = SPEC_DEFS[spec_name]
    a = pd.read_csv(os.path.join(ANA, f'{code}_analysis.csv'), parse_dates=['Date'], low_memory=False)
    a = a.sort_values('Date').reset_index(drop=True)
    r = a.set_index('Date')['Return'].dropna() * 100.0   # percent scale, matches 27_*.py
    dates = r.index

    start_i = int(np.searchsorted(dates.values, np.datetime64(SAMPLE_B_START)))
    start_i = max(start_i, 250)   # never fit on fewer than ~1 year of data
    if window_size is not None:
        start_i = max(start_i, window_size)   # need a full window before the first fit
    if start_i >= len(dates) - horizon:
        raise RuntimeError(f"{code}: not enough post-burn-in history for a rolling run")

    theta = None
    dist_names = None
    rows = []
    refit_log = []
    t0 = time.time()

    for i in range(start_i, len(dates) - horizon + 1):
        need_refit = theta is None or (i - start_i) % REFIT_EVERY == 0
        # everything strictly before date i -> OriginDate = dates[i-1]. Fixed window: only the
        # trailing `window_size` observations; expanding (window_size=None): everything to date.
        train = r.iloc[max(0, i - window_size):i] if window_size is not None else r.iloc[:i]
        am = arch_model(train, mean="AR", lags=1, vol=spec["vol"],
                         p=spec["p"], o=spec["o"], q=spec["q"], dist=spec["dist"])
        if need_refit:
            res = am.fit(disp="off", show_warning=False)
            theta = res.params
            dist_names = res.model.distribution.parameter_names()
            refit_log.append(dict(Code=code, Spec=spec_name, RefitDate=dates[i - 1],
                                   N=len(train), Converged=res.convergence_flag == 0,
                                   **{f"param_{k}": v for k, v in theta.items()}))
            fixed = res
        else:
            fixed = am.fix(theta)

        f = fixed.forecast(horizon=horizon, reindex=False)
        var_h = float(f.variance.values[-1, :].sum())     # cumulative H-day variance, percent^2
        mean_h = float(f.mean.values[-1, :].sum())         # cumulative H-day mean, percent
        sigma = np.sqrt(var_h) / 100.0              # decimal
        mu = mean_h / 100.0

        dp = [theta[n] for n in dist_names] if dist_names else []
        dist_obj = am.distribution
        taus = [t for _, t in LEVELS]
        q = dist_obj.ppf(taus, dp) if dp else dist_obj.ppf(taus, None)
        target_date = dates[i + horizon - 1] if horizon > 1 else dates[i]
        row = dict(Date=target_date, OriginDate=dates[i - 1], SigmaHat=sigma, VarHat=sigma ** 2)
        for (k, tau), qi in zip(LEVELS, q):
            row[f"VaR_{k}"] = mu + sigma * qi
        rows.append(row)

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    df["ES_01"] = np.nan   # arch's Distribution API exposes ppf but not a closed-form ES for
    df["ES_025"] = np.nan  # every dist; left NaN here rather than approximated silently -
                            # 28_realized_garch.py's analytic Student-t ES is the reference
                            # implementation where ES is required.
    act = a.set_index('Date')
    if horizon == 1:
        df["Realized"] = act.loc[df["Date"], "Return"].values
        rv_valid = act.loc[df["Date"], "RV_Valid"].astype(bool).values
        df["RVProxy"] = np.where(rv_valid, act.loc[df["Date"], "RV_Scaled"].values, np.nan)
    else:
        # H-day cumulative actual: log returns are additive, so Realized = sum of the daily
        # log returns strictly after OriginDate through Date (inclusive). RVProxy likewise sums
        # RV_Scaled over the same window, but only when every day in it has RV_Valid - a single
        # defect day (e.g. NKY) invalidates the whole H-day realized-variance comparison, not
        # just that one day, since the sum would otherwise silently understate it.
        realized, rvproxy = [], []
        for _, rr in df.iterrows():
            window = act.loc[(act.index > rr["OriginDate"]) & (act.index <= rr["Date"])]
            realized.append(float(window["Return"].sum()))
            if window["RV_Valid"].astype(bool).all() and len(window) == horizon:
                rvproxy.append(float(window["RV_Scaled"].sum()))
            else:
                rvproxy.append(np.nan)
        df["Realized"] = realized
        df["RVProxy"] = rvproxy
    df["Valid"] = True
    df["Reason"] = ""
    return df, pd.DataFrame(refit_log), elapsed


def main():
    os.makedirs(VAL, exist_ok=True)
    os.makedirs(LOG, exist_ok=True)
    os.makedirs(fio.FCDIR, exist_ok=True)
    all_refits = []
    log_lines = [f"phase21 rolling forecast engine — {pd.Timestamp.now()}",
                 f"REFIT_EVERY={REFIT_EVERY} trading days, expanding window, RUN_SPECS={RUN_SPECS}", ""]
    t0 = time.time()

    for spec_name in RUN_SPECS:
        for code in CODES:
            print(f"[{spec_name}] [{code}] rolling walk-forward ...")
            df, refits, elapsed = run_index_spec(code, spec_name)
            n_refits = refits["RefitDate"].nunique() if len(refits) else 0
            print(f"    {len(df)} forecasts, {n_refits} refits, {elapsed:.1f}s "
                  f"({df['Date'].min().date()} .. {df['Date'].max().date()})")
            log_lines.append(f"[{spec_name}][{code}] n={len(df)} refits={n_refits} time={elapsed:.1f}s")
            path = fio.write_forecasts(df, model=spec_name, code=code,
                                        spec=f"{spec_name}-AR1-expanding-refit{REFIT_EVERY}")
            print(f"    -> wrote {path} (contract-validated)")
            all_refits.append(refits)

    refit_df = pd.concat(all_refits, ignore_index=True) if all_refits else pd.DataFrame()
    refit_df.to_csv(os.path.join(VAL, 'rolling_engine_refit_log.csv'), index=False)

    with open(os.path.join(LOG, 'phase21_rolling_engine.log'), 'w') as f:
        f.write("\n".join(log_lines) + f"\n\ntotal {time.time()-t0:.1f}s\n")

    print(f"\nDone in {time.time()-t0:.1f}s total. Refit log: {len(refit_df)} refit events.")


if __name__ == "__main__":
    main()
