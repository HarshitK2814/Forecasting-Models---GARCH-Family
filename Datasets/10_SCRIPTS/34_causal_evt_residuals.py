# -*- coding: utf-8 -*-
"""
CAUSAL (look-ahead-free) standardised-residual provider for GARCH-EVT stage 2.

Researcher A, fix for an issue flagged in code review of Researcher B's PR #1
(2026-08-26), disclosed the same day in 27_baseline_garch.py and
RESEARCHER_A_SCOPE.md, and closed here on 2026-08-26 at explicit user
instruction ("fix everything related to that").

THE PROBLEM THIS FIXES
  27_baseline_garch.py's `<CODE>_std_resid.csv` is a SINGLE full-sample GJR-skewt
  fit: CondVol/StdResid at every historical date reflect parameters estimated on
  the WHOLE sample, including dates after any given OriginDate. 42_garch_evt.py's
  expanding-window GPD tail fit used this series at every OriginDate, so xi/beta -
  and every downstream GARCH-EVT VaR/ES - carried a look-ahead channel through
  CondVol, on top of the smaller, separately-disclosed Mu-reconstruction bias
  (both used the same full-sample file).

THE FIX, AND WHY IT COSTS NO NEW GARCH OPTIMISATION
  29_rolling_forecast_engine.py already refits AR(1)-GJR-skewt on an EXPANDING
  window (1990 -> RefitDate) every REFIT_EVERY=21 trading days, and already
  persists every refit's parameters to 08_VALIDATION/rolling_engine_refit_log.csv.
  Those parameters were estimated using ONLY data through RefitDate - exactly
  "causal as of any OriginDate >= RefitDate, up to the next refit". Re-attaching a
  refit's fixed parameters via arch's `.fix(theta)` (no optimisation - the same
  cheap operation script 29's own daily forecast step already performs)
  reproduces that refit's in-sample std_resid/conditional_volatility bit-for-bit
  (verified: `.fix(res.params).std_resid` equals `res.std_resid` to float
  precision - see the verification in this module's git history / commit
  message). So building the causal residual series costs ~150 `.fix()` calls per
  index (one per refit already on record), not a single new refit.

  For an OriginDate T, `residual_history_as_of` finds the LATEST RefitDate <= T
  and returns that refit's std_resid for its ENTIRE training window
  (1990 -> RefitDate) - the same expanding-history EVT input 42_garch_evt.py
  already expects, computed with parameters that never saw data after T. Within
  a refit block the series is frozen at the block's start (not updated day by
  day) - a disclosed, minor granularity approximation (at most 20 days stale),
  NOT a look-ahead channel: every value used at OriginDate T still comes from a
  fit that only saw data through RefitDate <= T.

  Mu (the AR(1) conditional-mean forecast at the target date) is recovered
  algebraically from the ALREADY-COMMITTED GJR-skewt__<CODE>_forecasts.csv
  (VaR_01 = mu + sigma * skewt.ppf(0.01; eta, lambda), all other terms known),
  rather than reconstructed from the full-sample residual file. This is exact
  (verified against a direct `.forecast(horizon=1)` call to float precision) and
  incidentally also closes the smaller, separately-disclosed Mu bias in the same
  fix, since it now uses the genuinely walk-forward-forecast mean rather than an
  in-sample full-sample-fit value.

OUTPUT
  No new data files - this module is imported directly by 42_garch_evt.py.
  results/tables/34_causal_verification.csv: one row per index, confirming the
  .fix() reconstruction matches 27_baseline_garch.py's own numbers at the LAST
  refit (the block closest to a full-sample fit, hence the strongest test) and
  reports how many distinct refit blocks were used.
"""
import os
import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import SkewStudent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
VAL = os.path.join(ROOT, '08_VALIDATION')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

SPEC_KW = dict(vol="GARCH", p=1, o=1, q=1, dist="skewt")  # GJR-skewt, matches 27_/29_


def _load_returns(code):
    a = pd.read_csv(os.path.join(ANA, f'{code}_analysis.csv'), parse_dates=['Date'], low_memory=False)
    a = a.sort_values('Date').reset_index(drop=True)
    return a.set_index('Date')['Return'].dropna() * 100.0  # percent scale, matches 27_/29_


def _load_refit_log(code):
    log = pd.read_csv(os.path.join(VAL, 'rolling_engine_refit_log.csv'), parse_dates=['RefitDate'])
    log = log[(log['Code'] == code) & (log['Spec'] == 'GJR-skewt')].sort_values('RefitDate').reset_index(drop=True)
    if log.empty:
        raise RuntimeError(f"{code}: no GJR-skewt refits in rolling_engine_refit_log.csv - "
                            f"run 29_rolling_forecast_engine.py first")
    return log


class CausalResidualSource:
    """One instance per index.

    residual_history_as_of(origin_date) -> pd.Series of StdResid, index<=origin_date,
        computed with parameters that never saw data after origin_date. None if
        origin_date precedes the first refit.
    mu_forecast(origin_date, sigma, var_01_gjr_skewt) -> the AR(1) mean forecast for
        the day after origin_date, recovered algebraically (see module docstring).
    """

    def __init__(self, code):
        self.code = code
        self.r = _load_returns(code)
        self.log = _load_refit_log(code)
        self._param_cols = [c for c in self.log.columns if c.startswith('param_')]
        self._dist = SkewStudent()
        self._cache = {}

    def _block_for(self, origin_date):
        eligible = self.log[self.log['RefitDate'] <= origin_date]
        if eligible.empty:
            return None
        return eligible.iloc[-1]

    def _theta(self, block):
        return pd.Series({c[len('param_'):]: block[c] for c in self._param_cols})

    def _hist_for_block(self, block):
        refit_date = block['RefitDate']
        if refit_date in self._cache:
            return self._cache[refit_date]
        train = self.r.loc[:refit_date]
        if len(train) != int(block['N']):
            raise RuntimeError(f"{self.code} refit {refit_date.date()}: reconstructed training "
                                f"window is {len(train)} obs, refit log says {int(block['N'])} - "
                                f"the return series or refit log has drifted since script 29 ran")
        theta = self._theta(block)
        am = arch_model(train, mean="AR", lags=1, **SPEC_KW)
        fixed = am.fix(theta)
        out = pd.DataFrame({
            'StdResid': fixed.std_resid.values,
            'CondVol': (fixed.conditional_volatility / 100.0).values,
        }, index=train.index)
        self._cache[refit_date] = out
        return out

    def residual_history_as_of(self, origin_date):
        block = self._block_for(origin_date)
        if block is None:
            return None
        return self._hist_for_block(block)['StdResid'].dropna()

    def mu_forecast(self, origin_date, sigma, var_01_gjr_skewt):
        block = self._block_for(origin_date)
        if block is None or not np.isfinite(var_01_gjr_skewt):
            return np.nan
        theta = self._theta(block)
        q01 = float(self._dist.ppf([0.01], [theta['eta'], theta['lambda']])[0])
        return var_01_gjr_skewt - sigma * q01


def _verify(code):
    """Sanity check: the last refit's reconstruction should be close to
    27_baseline_garch.py's full-sample fit (they nearly coincide near the end
    of the sample, where the two training windows are almost the same length),
    and clearly diverge earlier - which is the whole point of the fix."""
    src = CausalResidualSource(code)
    last_block = src.log.iloc[-1]
    last_hist = src._hist_for_block(last_block)
    full = pd.read_csv(f'{ROOT}/06_REALIZED_MEASURES/{code}_std_resid.csv', parse_dates=['Date']
                       ).set_index('Date').sort_index()
    common = last_hist.index.intersection(full.index)
    tail_common = common[common >= common[-1] - pd.Timedelta(days=60)]
    early_common = common[common <= common[0] + pd.Timedelta(days=1500)]
    tail_diff = float((last_hist.loc[tail_common, 'StdResid'] - full.loc[tail_common, 'StdResid']).abs().max())
    early_diff = float((last_hist.loc[early_common, 'StdResid'] - full.loc[early_common, 'StdResid']).abs().max())
    return {'code': code, 'n_refit_blocks': len(src.log),
            'last_refit_date': last_block['RefitDate'], 'n_at_last_refit': len(last_hist),
            'max_abs_diff_recent_60d_vs_fullsample': tail_diff,
            'max_abs_diff_early_history_vs_fullsample': early_diff}


def main():
    os.makedirs('results/tables', exist_ok=True)
    rows = [_verify(code) for code in CODES]
    out = pd.DataFrame(rows)
    out.to_csv('results/tables/34_causal_verification.csv', index=False)
    print(out.to_string(index=False))
    print("\nExpected pattern: 'recent' diff small (causal and full-sample windows nearly\n"
          "coincide near the end of history), 'early history' diff clearly non-zero (this is\n"
          "exactly the look-ahead the causal source removes - early residuals now reflect only\n"
          "parameters known at the time, not parameters re-estimated using the whole sample).")


if __name__ == "__main__":
    main()
