# Researcher A — modelling deliverables, session 2026-08-24

**Status: all seven items on A's list complete.** Data acquisition and cleaning (session
2026-08-22/23) plus crisis labels were already done. This session adds the four modelling
deliverables and the three cross-index decisions from the Executive Summary's A-side plan.
Total: **6 of 6 module-level deliverables complete, 88 h of planned scope delivered.**

Read `RESEARCHER_A_DECISIONS.md` first — it is the shorter document and it is what actually
unblocks Researcher B (schema, session convention, crisis-coverage statement). This document
is the fuller record of what was built and why.

`EXECUTIVE_SUMMARY_ADDENDUM.md` reconciles four places where delivery differs from the
Executive Summary's own text (free data sources instead of paid, 6 indices instead of 1-3, a
fuller Realized GARCH spec than the doc's own simplified equation, and the window-length /
horizon-extension robustness items closed after the doc flagged them as outstanding) — read it
before citing the Exec Summary's data-source or model-equation sections directly in the paper.

---

## 1. Baseline GARCH, GJR-GARCH, EGARCH — `10_SCRIPTS/27_baseline_garch.py`

Six specs per index (Normal/t/skew-t × symmetric/asymmetric), AR(1) mean equation, full
history from 1990. AR(1) was added after a constant-mean first pass left HSI's standardised
residuals autocorrelated (Ljung-Box p=5e-5 at 10 lags) — fixed, and it also matches
`FEATURE_SETS.csv`'s explicit "AR(1)-GJR-GARCH" spec for GARCH-EVT stage 1.

| Code | GJR-skewt LogLik | AIC | Persistence |
|---|---|---|---|
| SPX | -11857.8 | 23731.7 | 0.985 |
| NDX | -15590.4 | 31196.8 | 0.992 |
| UKX | -11909.5 | 23835.0 | 0.980 |
| DAX | -14225.3 | 28466.5 | 0.986 |
| NKY | -15064.5 | 30145.0 | 0.976 |
| HSI | -15167.7 | 30351.4 | 0.987 |

**GJR-skewt is the primary spec** (matches the plan's stage-1 GARCH-EVT model and the EDA's
own asymmetry/heavy-tail findings), though **EGARCH-skewt has marginally lower AIC on every
index** — reported as a comparator, not silently substituted, because swapping the plan's
named model is B's call, not an implicit one made by an AIC ranking. See
`08_VALIDATION/garch_baseline_params.csv` for every spec × index cell.

**Output**: `06_REALIZED_MEASURES/<CODE>_std_resid.csv` — Date, Return, CondVol, StdResid from
the GJR-skewt full-sample fit. Used for parameter tables, diagnostics, the summary figure, and
`41_evt_threshold.py`'s one-time global threshold calibration (a fixed hyperparameter chosen
once from the whole history, not a per-forecast quantity).

**No longer the input to GARCH-EVT stage 2.** Flagged 2026-08-25 (code review of B's GARCH-EVT
PR): stage 2's expanding-window GPD tail parameters (xi, beta) — which set every GARCH-EVT
VaR/ES — were fit at each OriginDate on this file's residuals, whose scale already reflected
parameters estimated on the full sample, including future dates. A look-ahead channel into the
tail shape, beyond the smaller Mu-reconstruction bias B originally disclosed in PR #1. **Fixed
2026-08-26**: `42_garch_evt.py` now sources both StdResid and Mu from
`34_causal_evt_residuals.py`, which reuses `29_rolling_forecast_engine.py`'s own walk-forward
refits (already computed and on record in `rolling_engine_refit_log.csv` — no new GARCH
optimisation) via `.fix(theta)` at the latest refit at or before each OriginDate, so nothing
feeding the tail fit was ever estimated on data after it. Effect on GARCH-EVT's output: breach
counts unchanged on five of six indices; NDX moved from 46 to 49 (see
`results/tables/34_causal_verification.csv` and `34_causal_evt_residuals.py`'s docstring for
why the effect is this small — recent history barely changes, only the distant past does).

---

## 2. Realized GARCH — `10_SCRIPTS/28_realized_garch.py`

Not available in the `arch` package (confirmed by checking its supported model list) —
implemented directly via quasi-MLE (scipy.optimize), following Hansen, Huang & Shek (2012,
*Journal of Applied Econometrics*), log-linear specification: three equations (return, GARCH
recursion on log h_t, measurement equation on log x_t with a leverage function
τ1·z_t + τ2·(z_t²−1)). Verified against the paper's own equations before coding.

Fitted on `RV_Scaled` (Hansen–Lunde-corrected to the close-to-close scale), not raw session
`RV` — feeding session-only RV into a model of the close-to-close variance would force `xi`
to silently absorb a scale mismatch that differs by index (1.71×–3.04×).

| Code | N | LogLik | AIC | β | γ | φ | τ1 | ν | Persistence (β+γφ) | RV imputed days |
|---|---|---|---|---|---|---|---|---|---|---|
| SPX | 3670 | -7062.7 | 14145.4 | 0.407 | 0.560 | 0.934 | -0.243 | 8.56 | 0.930 | 375 |
| NDX | 3753 | -8225.2 | 16470.3 | 0.461 | 0.528 | 0.903 | -0.233 | 9.19 | 0.938 | 259 |
| UKX | 3770 | -6167.0 | 12353.9 | 0.555 | 0.435 | 0.918 | -0.123 | 7.20 | 0.954 | 173 |
| DAX | 3266 | -6757.1 | 13534.2 | 0.549 | 0.433 | 0.935 | -0.161 | 6.54 | 0.953 | 23 |
| NKY | 3558 | -7811.7 | 15643.4 | 0.556 | 0.387 | 0.992 | -0.135 | 6.41 | 0.940 | **842** |
| HSI | 3670 | -8055.5 | 16131.1 | 0.647 | 0.314 | 1.016 | -0.053 | 8.95 | 0.966 | 177 |

All six converged; φ close to 1 everywhere (as theory predicts — the realized measure should
be an unbiased-up-to-leverage signal of h_t); τ1 negative everywhere (leverage effect,
consistent with the EDA's −0.12 to −0.20 correlation finding).

**The NKY gap, handled explicitly, not silently.** 842 of 3,558 days (24%) have no valid
realized measure. On those days the GARCH recursion substitutes `h_{t-1}` for the missing
`x_{t-1}` (self-consistent — the measurement equation says E[x_t|h_t]=h_t up to the leverage
correction) — but the **likelihood never sees the imputed value**, only the recursion does.
Every affected forecast row carries `Reason="RV_imputed_in_recursion"` so B can isolate them.

**Output**: `06_REALIZED_MEASURES/<CODE>_realized_garch_fit.csv` (full diagnostic series) and
`20_FORECASTS/RealGARCH__<CODE>_forecasts.csv` (contract-format, ready for B's evaluation
code). VaR and ES use the closed-form Student-t formulas (ES formula checked against the
McNeil–Frey–Embrechts analytic result before coding).

---

## 3. Rolling out-of-sample forecast engine — `10_SCRIPTS/29_rolling_forecast_engine.py`

Genuine walk-forward: parameters re-estimated every 21 trading days on an **expanding**
window (chosen over fixed rolling because the EDA found long memory, GPH d=0.50–0.63 — a
fixed window forgets exactly what a long-memory process still needs), but the conditional-
variance **state** updates every day using the real, just-observed return via `arch`'s
`.fix(params).forecast(horizon=1)` — so every row is a true 1-step-ahead forecast, not a
multi-step projection held over the refit gap.

Run for the primary spec (GJR-skewt) across all six indices, covering the full sample-B
window (2013-09-30 onward):

| Code | Forecasts | Refits |
|---|---|---|
| SPX | 3,243 | 155 |
| NDX | 3,243 | 155 |
| UKX | 3,258 | 156 |
| DAX | 3,267 | 156 |
| NKY | 3,150 | 150 |
| HSI | 3,172 | 152 |

**Realized GARCH is NOT walk-forward re-estimated** — its custom optimiser takes ~85s per
full-sample fit; a monthly walk-forward at that cost would be ~130 refits × 6 indices × 85s ≈
18 hours, not run in this session. Its forecast file instead uses full-sample parameters with
a daily-recursive (look-ahead-free in *state*, not in *parameters*) 1-step series — flagged
explicitly in the script docstring as weaker than genuine walk-forward, and left as an
overnight-batch candidate at a coarser (quarterly/annual) cadence.

**Output**: `20_FORECASTS/GJR-skewt__<CODE>_forecasts.csv` — contract-validated.

**Also feeds `10_SCRIPTS/34_causal_evt_residuals.py` (added 2026-08-26).** That module reuses
`08_VALIDATION/rolling_engine_refit_log.csv` — every refit's parameters, already persisted here
— to give GARCH-EVT stage 2 a look-ahead-free residual/Mu source (see section 1's caveat,
above). No new refits: it re-attaches this engine's own already-fitted parameters via
`.fix(theta)`.

---

## 4. Robustness checks — `10_SCRIPTS/30_robustness_checks.py`, `33_window_and_horizon_robustness.py`

Full results and interpretation notes: `08_VALIDATION/ROBUSTNESS_SUMMARY.md`. Six checks — the
first four cover sub-sample/distribution/frequency/cadence; the last two close the Executive
Summary's window-length and horizon-extension items, which the first robustness pass had not
yet addressed (see `EXECUTIVE_SUMMARY_ADDENDUM.md` §4 for the reconciliation):

**(a) Sub-sample stability (pre/post-COVID, GJR-skewt).** Persistence *drops* post-COVID for
every index (e.g. SPX 0.987→0.973, NKY 0.981→0.921, HSI 0.988→0.942) and the skew parameter
becomes more negative for five of six (leverage effect strengthens) — a genuine finding, not
noise: report it, don't average over it.

**(b) Innovation distribution.** Normal→t is decisive everywhere (ΔAIC 252–466); t→GJR-skewt
is also decisive for five indices (ΔAIC 178–326) but weak for HSI (ΔAIC 63) — HSI's asymmetry
is the least pronounced of the six, consistent with its comparatively small Engle-Ng effect
in the EDA.

**(c) Sampling-frequency sensitivity.** The Hansen-Lunde scale factor moves 3–6% from 5-min
to 10-min RV, but up to 17–28% by 30-min — the 5-min choice is doing real work, not an
arbitrary convention. UKX is the one exception, moving slightly *down* then up — flagged, not
explained away.

**(d) Refit-cadence sensitivity (SPX, 21-day vs 63-day).** Correlation 0.99998, mean absolute
relative difference 0.14% — confirms REFIT_EVERY=21 in the rolling engine is a compute-cost
choice, not a result-changing one; 63-day would have been equally valid and ~40% cheaper.

**(e) Window-length sensitivity (SPX, expanding vs fixed 2-year vs fixed 5-year).** Unlike
refit-cadence, this one **matters**: correlation with expanding drops to ~0.954, mean absolute
relative difference ~10%. The 2020 COVID window shows why — fixed 5-year overshoots expanding
by up to +2pp at the peak, then both fixed windows undershoot for months after as COVID ages
out of their lookback. Corroborates the original expanding-window choice (motivated by the
EDA's GPH d=0.50–0.63 long-memory finding) with evidence, not just the original argument.

**(f) Horizon extension (all six indices, 1-day vs 5-day cumulative).** Mean annualised
volatility agrees to within 0.1–0.7pp between 1-day and 5-day forecasts on every index — the
long-run volatility estimate is stable across aggregation horizons. **Caveat, stated plainly
because it is easy to misread**: QLIKE is lower at 5-day than 1-day on every index, but that
is a scale artifact of QLIKE on a coarser target, not evidence of better 5-day forecasting —
QLIKE_1d and QLIKE_5d must never be compared to each other, only within the same horizon.

---

## Figures — `10_SCRIPTS/32_modelling_figures.py`, `33_window_and_horizon_robustness.py`, output in `09_FIGURES/`

Nine figures, each backing a specific number above rather than illustrating in the abstract:

| File | What it shows |
|---|---|
| `11_VaR_breach.png` | Return vs 1% VaR, breaches marked, all six indices. Empirical breach rates land at 0.8–1.5% against a 1% nominal target. |
| `12_forecast_vs_realized.png` | GJR-skewt and Realized GARCH sigma-hat vs sqrt(RV_Scaled), all six. The NKY panel visibly shows Realized GARCH flatlining through the 2016–17 gap while GJR-skewt (returns-only) keeps tracking — the same fact `17_nky_gap.png` makes with shading, seen from the other model's side. |
| `13_residual_diagnostics.png` | QQ vs fitted skew-t + ACF(resid²) with Ljung-Box p-values, all six. Center fits well; a handful of tail points (SPX, DAX) sit visibly off the skew-t line — the visual case for EVT beyond skew-t alone. |
| `14_subsample_stability.png` | Persistence and skew-t lambda, pre- vs post-COVID bars, all six — the picture behind check (a). |
| `15_frequency_sensitivity.png` | Hansen-Lunde scale factor vs sampling interval, all six lines — the picture behind check (c). |
| `16_refit_cadence_overlay.png` | SPX sigma-hat, 21-day vs 63-day refit, overlaid + difference panel — the picture behind check (d). |
| `17_nky_gap.png` | NKY Realized GARCH conditional volatility with the imputed-recursion window shaded (842 days) — the picture behind Phase 20.3. |
| `18_window_length_sensitivity.png` | SPX sigma-hat, expanding vs fixed 2y vs fixed 5y, overlaid + difference panel — the picture behind check (e), with the 2020 divergence directly visible. |
| `19_horizon_extension.png` | 1-day vs 5-day-ahead annualised volatility, all six indices — the picture behind check (f). |

---

## Three decisions — see `RESEARCHER_A_DECISIONS.md` for the full statements

5. **Non-synchronous sessions** — each index keeps its own local calendar; pooled/cross-index
   work must lag Asia by one session relative to US/Europe.
6. **Forecast-file schema** — `10_SCRIPTS/26_forecast_io.py`, enforced in code
   (`validate()`/`read_forecasts()`/`write_forecasts()`/`eval_frame()`), not just described.
   Synthetic placeholder files in `20_FORECASTS/_SYNTHETIC/` let B develop against the format
   before real output existed (now superseded by the real files above).
7. **Sample and crisis-coverage statement** — sample B (2,685 days, 2013-09-30 onward) misses
   the Asian Crisis, DotCom, GFC and Euro Sovereign Debt; out-of-sample evaluation only ever
   sees four post-2013 stress episodes (~17% of days). Must be stated in the paper.

---

## What B should do first

1. Read `RESEARCHER_A_DECISIONS.md` in full (10 minutes).
2. Point the GARCH-EVT stage-2 code at `06_REALIZED_MEASURES/<CODE>_std_resid.csv`.
3. Point evaluation/backtest code at `20_FORECASTS/{GJR-skewt,RealGARCH}__<CODE>_forecasts.csv`
   via `fio.read_forecasts()` / `fio.eval_frame()` — both already validated end-to-end (QLIKE,
   squared error and VaR breach indicators tested and confirmed correct on SPX for both
   models before this handoff).
4. Delete or ignore `20_FORECASTS/_SYNTHETIC/` — real files exist for every (model, index)
   pair now.

*Owner: Researcher A. Written 2026-08-24.*
