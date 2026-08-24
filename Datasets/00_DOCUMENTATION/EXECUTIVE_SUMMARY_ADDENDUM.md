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
