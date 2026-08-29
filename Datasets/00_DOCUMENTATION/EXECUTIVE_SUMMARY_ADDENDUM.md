# Addendum to Executive Summary — deviations and status

The original `Executive Summary.pdf` (root of the project) is the governing plan and is not
rewritten here. This addendum reconciles four places where what was actually built differs
from what the plan describes, plus a status check against every row explicitly assigned to
Researcher A. Read alongside `RESEARCHER_A_DECISIONS.md` and `RESEARCHER_A_SCOPE.md`.

---

## 1. Data sourcing — free sources, not the plan's paid ones

The Executive Summary's "Data Sources & Preprocessing" section names **NYSE TAQ via WRDS**,
**Bloomberg**, **Refinitiv**, and **Quandl** as the primary intraday/daily sources, with Yahoo
Finance mentioned only as a fallback. **All data actually used is free**: Yahoo Finance
(daily), Dukascopy (intraday tick/1-min), CBOE and STOXX (implied volatility), plus 22 keyless
macro series. This was a standing constraint set at the start of the project, not a shortcut
taken later. Consequence: no institutional/WRDS access is required to reproduce any part of
this work — a genuine advantage for a student project — but it does mean the plan's data
table (page 2 of the PDF) no longer describes the actual sources and should not be quoted
directly in the paper's data section without updating it.

## 2. Scope exceeded — six indices and 36 years, not "1-3" and "10+ years"

The plan calls for "1-3 major indices... at least 10+ years." The delivered dataset covers
**six Tier-1 indices** (SPX, NDX, UKX, DAX, NKY, HSI) from **1990-01-02 to 2026-08-21** (~36
years daily; intraday from 2011-09, DAX from 2013-09). This was a deliberate scope expansion,
not scope creep without purpose: cross-index results (PC1 = 62% of log-RV variation,
Asia-US timing artefacts, index-specific realized-measure biases) are only visible with more
than one or two indices, and several of the EDA's most useful findings (the NKY 2016-17 gap,
the monotone-in-session-length Hansen-Lunde scale factors) depend on having six.

## 3. Realized GARCH — the delivered specification is more complete than the plan's equation

The plan's Section 3.3 states the model as:
```
r_t = mu + eps_t,  eps_t = sigma_t * z_t
log(sigma_t^2) = omega + beta*log(sigma_{t-1}^2) + alpha*log(RV_{t-1})
```
This is the GARCH recursion only, with no measurement equation and no leverage term. The
actual Hansen, Huang & Shek (2012) paper the plan cites specifies a *third* equation (the
measurement equation) with an explicit leverage function:
```
log(x_t) = xi + phi*log(h_t) + tau1*z_t + tau2*(z_t^2 - 1) + sigma_u * eps_t
```
**The delivered implementation uses the full three-equation form**, including the leverage
function - matching the cited paper itself rather than the plan's own simplification of it.
This was necessary, not optional: without the leverage term the model cannot capture the
asymmetry the EDA's Engle-Ng test found in every index (worst-case p=7e-5). One consequence
worth flagging for the paper: `tau1` is negative in all six fits (-0.05 to -0.24), consistent
with the leverage-effect literature, and is itself a citable result.

## 4. Two Section-4 robustness items the plan named but the first robustness pass skipped

`30_robustness_checks.py` (session 2026-08-24) covered sub-sample stability, innovation
distribution, sampling-frequency sensitivity, and refit-cadence sensitivity — four checks, but
not all of what Section 4.1 and the "Evaluation & Robustness" section name explicitly:

- **Window length** ("try different rolling window sizes, e.g. 2 vs 5 years, and expanding
  windows"). The rolling engine (`29_rolling_forecast_engine.py`) only ever ran expanding.
- **Horizon** ("if applicable, extend forecasts to 5-day ahead and compare results"). Only
  1-step-ahead forecasts existed.

**Both are now closed** — `10_SCRIPTS/33_window_and_horizon_robustness.py`. Full numbers and
interpretation: `08_VALIDATION/ROBUSTNESS_SUMMARY.md` sections 5-6. Summary:

- **Window length** (SPX GJR-skewt, expanding vs fixed 2y vs fixed 5y): unlike refit-cadence
  (correlation 0.99998, barely matters), window length is a **real** choice - correlation with
  expanding drops to ~0.954, mean absolute relative difference ~10%. The 2020 COVID window
  makes the mechanism visible directly in `18_window_length_sensitivity.png`: fixed 5-year
  overshoots expanding by up to +2pp at the peak, then both fixed windows undershoot for months
  after as COVID ages out of their lookback. This corroborates the original design choice
  (expanding, motivated by the EDA's GPH d=0.50-0.63 long-memory finding) with evidence rather
  than leaving it as an assertion.
- **Horizon extension** (all six indices, GJR-skewt, 5-day cumulative): genuine walk-forward
  5-day forecasts written as real contract-format files
  (`20_FORECASTS/GJR-skewt-h5__<CODE>_forecasts.csv`, `Horizon=5`), usable by B's evaluation
  code exactly like the 1-day files. Mean annualised volatility agrees between 1-day and 5-day
  forecasts to within 0.1-0.7 percentage points on every index - the model's long-run vol
  estimate is stable across aggregation horizons. One caveat documented explicitly: QLIKE is
  lower at 5-day than 1-day on every index, but this is a **scale artifact of QLIKE on a
  coarser target**, not evidence the model forecasts better at 5 days - QLIKE_1d and QLIKE_5d
  must never be compared to each other directly, only within the same horizon across models.

---

## Status against every row explicitly labelled "Lead: A" in the plan's task table

| Task | Plan hours | Status |
|---|---|---|
| Data acquisition & cleaning | 40h | Done, scope exceeded (see §1, §2) |
| Realized volatility construction | 16h | Done, scope exceeded (semivariance, MedRV, RQ/TQ, subsampled RV added beyond plain RV) |
| Baseline GARCH & GJR/EGARCH | 24h | Done, 6 specs x 6 indices |
| Realized GARCH | 24h | Done, exceeds the plan's own equation (see §3) |
| Rolling out-of-sample engine | 24h | Done — expanding (production) **and now fixed-window comparison** (see §4) |
| Robustness checks (thresholds/windows/horizons) | 16h | Done — 4 checks (2026-08-24) **plus window-length and horizon-extension** (see §4) |

**All six of Researcher A's lead rows are now complete, including the two items previously
open.** Remaining shared (A/B) rows not yet complete: Docker/environment reproducibility audit,
unit tests for model-fitting functions, and B's half of figures/tables (crisis-regime and
evaluation visualisations, which cannot exist before B's models do).

*Owner: Researcher A. Written 2026-08-24, updated same day after closing the window-length and
horizon-extension items.*

---

**2026-08-26 follow-up.** B's PR #1 (all remaining lead rows: GARCH-EVT, quantile regression,
evaluation, crisis/regime, statistical comparison, reproducibility audit) merged the same day,
closing every remaining coding deliverable in this table. Of the two shared rows still open
above: unit tests are done (`tests/`, 39 pytest tests). A Dockerfile was built, reviewed, and
then dropped — never build-verified in this environment and not worth maintaining unverified;
environment reproducibility instead rests on `requirements.txt` / `requirements_B.txt`'s exact
pins plus `50_reproducibility_audit.py`'s environment check. Also fixed the same day: a
look-ahead channel in GARCH-EVT's tail-fit input (full-sample residuals, not walk-forward —
see `RESEARCHER_A_SCOPE.md` §1 and `34_causal_evt_residuals.py`), found in code review of PR #1.

---

**2026-08-29 follow-up — the execution-plan review and its fixes.** A separate AI-assisted
review of the execution plan (`Forecasting_Volatility_Research_Paper_Execution_Plan.pdf`)
identified two P0 look-ahead issues still open after the 08-26 EVT fix, plus several P1/P2
items. All are now closed:

**P0 — the two that could move the paper's central conclusion:**
1. **Causal Hansen-Lunde scaling.** `RV_Scaled` (the realized-measure scale factor used
   throughout) was a single constant estimated over the *entire* evaluation window - an
   observation dated 2014 was scaled using information partly from 2026. Replaced with
   `RV_Scaled_Causal`, an expanding factor using only observations strictly before the target
   date (`15_build_analysis_dataset.py` DECISION 5). Every model's QLIKE now scores against
   this causal series (`26_forecast_io.py load_actuals`), not just Realized GARCH's own.
2. **Walk-forward Realized GARCH.** Previously fit ONCE on the full sample (a genuine
   look-ahead in the parameter values, though the recursion itself was causal). Now
   re-estimated annually on an expanding window (`28_realized_garch.py`, ~13 refits/index,
   `08_VALIDATION/realized_garch_refit_log.csv`), mirroring the GJR engine's design. **This
   materially changed the headline result**: under the fair walk-forward refit, Realized
   GARCH's 99% VaR breach rate got dramatically worse (now RED on the Basel traffic light for
   4 of 6 markets, vs GARCH-EVT green on 5/6), while its QLIKE (volatility-accuracy) advantage
   over GJR-skewt survived intact and remains statistically significant on 5/6 indices under
   proper HAC inference. The "variance accuracy ≠ tail calibration" story is now demonstrated
   honestly rather than resting on an in-sample parameter advantage - see
   `results/tables/47a_volatility_losses.csv`, `47b_var_backtests.csv`, `49_basel.csv`.

**P1:**
- Skew-t Realized GARCH robustness variant added (`RealGARCH_ST` in every forecast/results
  table), per Watanabe (2012)'s finding that a symmetric-t innovation can confound realized-
  information quality with tail-density misspecification.
- NKY session-close fix: the Tokyo Stock Exchange extended its cash-session close from 15:00
  to 15:30 JST effective 2024-11-05; the session-window constant used to build every NKY
  realized measure was date-independent and missed the extra half hour on ~450 trading days.
  Fixed across `05_build_intraday_and_RV.py`, `12_extended_realized_measures.py`,
  `14_session_classification.py`; full data pipeline rebuilt for NKY.
- HAC/Newey-West Diebold-Mariano fix: the serial-correlation correction in `40_b_common.py`
  was a no-op whenever h=1 (i.e. every DM call actually made in this project), silently using
  the plain single-period variance. Now applies a proper Bartlett-kernel long-run variance with
  the Newey-West (1994) plug-in bandwidth regardless of horizon.
- EVT exceedance-dependence diagnostic added (Ljung-Box, Wald-Wolfowitz runs test,
  Ferro-Segers extremal index) - `results/tables/41_exceedance_dependence.csv`. Found genuine
  tail clustering in SPX and HSI (p<0.05), a citable limitation for the discussion section.
- NKY missing-RV robustness table added (`robustness_nky_missing_rv.csv`) - confirms NKY does
  not materially drive the six-market headline QLIKE result.
- 5-day horizon metadata bug fixed: the `Horizon` contract field was silently defaulting to 1
  on every `GJR-skewt-h5` file too.

**Two bugs found only during this rebuild, unrelated to the causal-scaling work:**
- `44_qr.py` (quantile regression) was silently broken - a required predictor column
  (`TermSpread_diff`) was never created by the canonical data builder. Fixed in
  `15_build_analysis_dataset.py`; QR now runs and its forecast files are current.
- Pipeline sequencing gap: `20_finalise_and_document.py` (adds `CommonDate_B`/`BalancedRV_B`)
  must run after `15_build_analysis_dataset.py` and before any evaluation script - was missed
  once during this rebuild, causing a transient KeyError, now corrected in the run order.

Full regeneration verified: 28/29 reproducibility-audit checks pass (`results/tables/50_audit.csv`;
the one failure is expected local Python-version drift, not a real issue), all 45 unit tests
pass. Drive was swept folder-by-folder on 2026-08-29 and every file found to contain now-wrong
numbers (stale RealGARCH/GJR forecast files, stale NKY analysis/session-class files, stale
`06_MODEL_FITS`, stale robustness tables) was removed; large binary/CSV files could not be
re-uploaded through this tooling and need manual replacement from the local
`Datasets/20_FORECASTS/`, `09_FIGURES/`, `06_MODEL_FITS/`, and `01_ANALYSIS_READY/` folders.
