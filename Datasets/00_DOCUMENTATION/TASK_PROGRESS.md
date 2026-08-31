# Data Acquisition — Task Progress Tracker
Researcher A | GARCH-EVT vs Realized GARCH vs Quantile Regression
Sessions: 2026-08-22, 2026-08-23

Legend: `[x]` done+verified · `[~]` running · `[!]` blocked · `[ ]` pending

## STATUS: acquisition COMPLETE + cleaning/EDA COMPLETE.
Acquisition: 260 CSVs, 1.19 GB, 1,195 checks, 0 failures.
Cleaning/EDA: analysis dataset built and validated, 158 checks, 0 failures. See `EDA_REPORT.md`.

---

## Phase 0 — Setup
- [x] 0.1 Diagnosed and fixed expired CA bundle (certifi 2024.08.30 → 2026.7.22). Every HTTPS fetch was failing before this.
- [x] 0.2 Folder structure created
- [x] 0.3 Tracker created

## Phase 1 — Source verification (probe BEFORE downloading)
- [x] 1.1 CBOE volatility endpoints — 13 tested, 13 working
- [x] 1.2 Yahoo daily index tickers — 6/6 working
- [x] 1.3 STOXX VSTOXX (V2TX) + VDAX-NEW (V1X) — found, current to 2026-08-21
- [x] 1.4 Dukascopy binary candle format reverse-engineered and verified: `>Iiiiif`, months ZERO-INDEXED, prices int32 ÷1000
- [x] 1.5 FRED — website unreachable (4/4 timeouts, re-confirmed 2026-08-23); API host reachable, needs free key
- [x] 1.6 Rejected sources documented with evidence: Stooq (JS anti-bot), Alpha Vantage (month= now paid), FRED SP500 (10y limit + no redistribution), Oxford-Man (discontinued)

## Phase 2 — Daily index data ✅ COMPLETE
- [x] 2.1 All 6 indices, 1990-01-02 → 2026-08-21 (8,990–9,270 rows each)
- [x] 2.2 Each accepted only after **two identical consecutive fetches**
- [x] 2.3 CSVs written with LogReturn precomputed

## Phase 3 — Volatility indices ✅ COMPLETE
- [x] 3.1 CBOE ×13: VIX, VXN, VXEFA, VXEEM, VVIX, SKEW, VIX9D, VIX3M, VIX6M, RVX, VXD, OVX, GVZ
- [x] 3.2 STOXX ×3: V2TX (VSTOXX, from 1999), V1X (VDAX-NEW, from 2005), V6I1
- [x] 3.3 Yahoo ×1: NKVI (Nikkei VI, 2018+)
- [x] 3.4 All 17 validated and written

## Phase 4 — Intraday (Dukascopy 1-min, BID) ✅ COMPLETE
- [x] 4.1 Format confirmed against the 2020-03-16 COVID crash day
- [x] 4.2 True intraday start discovered: **2011-09-19** (not 2013 as the vendor pages imply).
      DAX is the exception at **2013-09-30** — verified by probing earlier years and getting empties.
- [x] 4.3 All six downloaded:

      SPX 4080/4080  UKX 4080/4080  NKY 4080/4080  HSI 4080/4080
      NDX 4074/4080  DAX 3551/3559

      The 14 missing days survived **three independent retry passes** — absent at source.
- [x] 4.4 Lesson recorded: Dukascopy throttles per-IP. Extra parallel processes made throughput
      **worse** (4.9 → 1.9 req/s) and triggered a lasting penalty. ONE process, 6 threads,
      keep-alive sessions. Without keep-alive it is ~0.05 req/s — a 100× penalty.

## Phase 5 — Realized volatility ✅ COMPLETE

| | SPX | NDX | UKX | DAX | NKY | HSI |
|---|---|---|---|---|---|---|
| session-days | 3,755 | 3,733 | 3,728 | 3,307 | 3,543 | 3,576 |
| median 5-min bars | 78 | 78 | 102 | 102 | 60 | 66 |
| expected | 78 | 78 | 102 | 102 | 60 | 66 |

- [x] 5.1 Bar counts match every exchange calendar **exactly** — independent confirmation that
      the session windows and DST handling are correct.
- [x] 5.2 RV_{1,5,10,15,30}min + BPV + jumps + overnight return computed for all six
- [x] 5.3 Exchange-holiday / feed-outage guard added (4 bp session-range threshold).
      Removes 15 sessions, all logged to `08_VALIDATION/frozen_sessions_dropped.csv`.
      Caught a **Dukascopy feed outage on SPX 2013-02-25→28 and NDX 2013-02-26→28** —
      days the exchange was open, which were SPX's two largest CFD residuals.

## Phase 6 — Macro ✅ COMPLETE (keyless route)
- [x] 6.1 **22 keyless series downloaded** (script 09): rates curve, FX, commodities, credit
      ETFs, 6 regional equity ETFs. Removes the FRED key dependency for everything we use.
- [!] 6.2 5 remaining FRED series (CPIAUCSL, UNRATE, INDPRO, NFCI, USREC) have no honest market
      analogue. Optional weekly/monthly covariates. Script 04 ready; needs your free key.

## Phase 7 — Validation ✅
- [x] 7.1 **1,195 checks across 239 files — 0 failures.** 19 warnings, every one investigated
      individually and explained (see README §3). None left open.
- [x] 7.2 CFD-vs-index check, **all six pass**:

      SPX R²=0.9928 PASS      NDX R²=0.9971 PASS      DAX R²=0.9825 disclose
      HSI R²=0.9873 disclose  NKY R²=0.9856 disclose  UKX R²=0.9700 disclose

      **The alignment rule was decisive.** Comparing CFD returns that span a feed gap against
      one-day index returns wrecks the fit. Naive vs aligned: NDX 0.972→0.997, NKY 0.927→0.986,
      HSI 0.943→0.987, UKX 0.912→0.970. On the naive numbers UKX would have been discarded.
- [x] 7.3 Variance decomposition: SPX session RV 12.51% + overnight 10.82% → 16.54% vs actual
      daily 16.98%. A 2.6% gap — near exact.
- [x] 7.4 Yahoo FX/futures OHLC inconsistency diagnosed: **Close is reliable, O/H/L are
      placeholders.** Proven on GOLD — Close sits 32.9 bp from next-day Open vs 61.5 bp for the
      H/L midpoint. Every macro row now carries `OHLC_Consistent`.
- [x] 7.5 Yield-unit bug caught and fixed: Yahoo no longer scales ^TNX by 10. Dividing produced
      a 10-year yield of 0.05–0.91%.

## Phase 8 — Panel ✅ COMPLETE
- [x] 8.1 `07_PANEL_INTERMEDIATE/<CODE>_panel_daily.csv` — 36–38 columns, exchange daily file as the spine,
      RV coverage 91.8–100% since each index's RV start date.

## Phase 9 — Documentation ✅
- [x] 9.1 `DATA_ACQUISITION_REPORT.xlsx` — **28 sheets**
- [x] 9.2 `README.md` — standalone hand-off doc: provenance, rejected sources with evidence,
      every limitation, reproduction commands, and the two operational traps (certifi, throttling)
- [x] 9.3 Regenerate any time with `python 10_SCRIPTS/07_build_excel_report.py`

---

## Remaining optional work (nothing is blocking)
- [ ] Dukascopy **ASK** side, for mid-price RV as a microstructure robustness check.
      `set DUKA_SIDE=ASK` then re-run script 03. ~15 min per index at the measured 4.8 req/s.
- [ ] Alpaca SPY/QQQ 1-min as an independent cross-check on Dukascopy RV. Free, needs an account key.
- [ ] The 5 FRED series above.

## Full rebuild
```
pip install -r requirements.txt
python 10_SCRIPTS/01_download_daily.py
python 10_SCRIPTS/02_download_volatility.py
python 10_SCRIPTS/03_download_intraday.py SPX NDX UKX DAX NKY HSI
python 10_SCRIPTS/09_download_macro_yahoo.py
python 10_SCRIPTS/05_build_intraday_and_RV.py
python 10_SCRIPTS/10_build_master_panel.py
python 10_SCRIPTS/06_validate.py
python 10_SCRIPTS/08_cfd_vs_index_check.py
python 10_SCRIPTS/07_build_excel_report.py
```
Step 3 is resumable — cached days are skipped, so a repeat run only fetches what is missing.


---

# PART 2 — Cleaning and EDA (session 2026-08-23)

## Phase 10 — Sample definition ✅
- [x] 10.1 Three samples frozen in code (`11_define_samples.py`). **B is primary**: all six
      indices, 2013-09-30 → 2026-08-21, NKY on the declared VXEFA proxy.
      A = 3,056 days / 5 indices (no euro area). B = 2,685. C = 1,892.
- [x] 10.2 `InSample_*`, `CommonDate_B` and `BalancedRV_B` flags written onto every panel.

## Phase 11 — Extended realized measures ✅
- [x] 11.1 Recomputed from the 1-min cache: realized **semivariance** (RS+/RS−), **MedRV**,
      **RQ/TQ** quarticity, **subsampled RV**, realized skew/kurtosis. 39 columns per index.
- [x] 11.2 Alignment against the existing RV files asserted, not assumed — max relative
      difference 1.5e-4 (float formatting only), zero row mismatches.
- [x] 11.3 Caught my own error: the RV standard error was `sqrt(2/3·RQ)`, overstating it by
      `sqrt(n/3)` — a factor of five on a 77-return session. Correct form is `sqrt(2·RQ/n)`,
      giving a relative measurement error of 16–22%.

## Phase 12 — Session classification ✅ **(the key finding)**
- [x] 12.1 Every session classified FULL / HALFDAY / DEFECT on the **5-minute grid**.
      A first pass on the 1-minute grid was wrong — it called 674 SPX days "half-days"
      against the ~3/year the exchange schedules, because a quiet minute with no quote
      change is dropped upstream and is not missing data.
- [x] 12.2 **NKY 2016 and 2017 are 100% defective** — the entire 09:00 opening hour is absent
      from the Dukascopy feed. 672 defect sessions, 18.9% of its history.
- [x] 12.3 Bias **measured, not assumed**: RV/Parkinson on NKY defect days = 0.786 vs 1.030
      on full days → RV biased **24% low**. SPX control: 1.035 vs 1.135.
- [x] 12.4 Half-days confirmed genuine — they land on 07-03, 07-04, 12-24, 12-31.

## Phase 13 — Cleaned analysis dataset ✅
- [x] 13.1 `01_ANALYSIS_READY/<CODE>_analysis.csv`, 96 columns. Realized measures **nulled** on defect
      days; **rows kept**, because the exchange close is still valid.
- [x] 13.2 **Returns NOT winsorized.** Explicit decision — the tail is the object of study.
- [x] 13.3 Macro forward-filled ≤5 business days, forward only. Never backward.
- [x] 13.4 Hansen-Lunde scale factors: DAX 1.71, UKX 1.74, SPX/NDX 1.80, HSI 2.14, NKY 3.04 —
      **monotone in session length**, which is the internal check that they are real.
- [x] 13.5 Return re-derived from Close at full precision; the download had rounded it to 6 dp.

## Phase 14 — Statistical EDA ✅
- [x] 14.1 Hill tail index **2.6–3.9**, all below 4 → fourth moment does not exist → the
      quantitative case for EVT.
- [x] 14.2 **Engle-Ng sign-bias rejects on all six** (worst p = 7e-5) → plain GARCH(1,1) is
      misspecified, GJR/EGARCH required.
- [x] 14.3 ARCH-LM and Ljung-Box on r² reject at p < 1e-16 everywhere.
- [x] 14.4 GPH d on log RV = **0.50–0.63**, ADF and KPSS both reject → long memory → HAR.
- [x] 14.5 Leverage corr(r_t, logRV_{t+1}) = −0.12 to −0.20, negative for all six.
- [x] 14.6 Session variance share 33–58% — the overnight gap is 42–67% of daily variance.

## Phase 15 — Predictor screening ✅
- [x] 15.1 `US10Y_pct` and `TermSpread_pct` **non-stationary in levels** (ADF p = 0.87 / 0.33);
      differenced forms added and the levels flagged as unfit as regressors.
- [x] 15.2 **`LogRV` vs `LogRS_neg` VIF ≈ 95** — the same variable by identity. Re-parameterised
      as level + share: max VIF 21.9 → **8.1**.
- [x] 15.3 Mechanical identities verified to 1e-12 against a median RV of 3e-5.
- [x] 15.4 Predictive screening, all targets at t+1: best for log RV is `LogRV_w` (R²=0.567);
      **best for the 1% tail is the implied-vol index** (pseudo-R²=0.199). `RangePct` from the
      daily bar alone reaches R²=0.42 and is available on every day.
- [x] 15.5 Cross-index: PC1 = 62.2% of log-RV variation. SPX–NDX 0.96, UKX–DAX 0.85, Asia–US
      ~0.18 on returns — a **timing artefact**, not independence.

## Phase 16 — EVT threshold diagnostics ✅
- [x] 16.1 GPD ξ fitted across thresholds on **rolling-standardised** returns (the right
      analogue for McNeil-Frey, which applies EVT to GARCH residuals, not raw returns).
- [x] 16.2 ξ positive throughout → heavy tail survives volatility standardisation.
      **Recommended POT threshold: 95th–97.5th percentile**, 230–460 exceedances.

## Phase 17 — Figures and documentation ✅
- [x] 17.1 Ten diagnostic figures in `09_FIGURES/`.
- [x] 17.2 `EDA_REPORT.md` (29k chars) — every number read from the validation CSVs, not typed.
- [x] 17.3 `EDA_REPORT.xlsx`, 18 sheets. `DATA_DICTIONARY.csv` (37 entries),
      `FEATURE_SETS.csv` (17 entries).

## Phase 18 — Final validation ✅
- [x] 18.1 **158 checks, 0 failures** (`22_validate_analysis.py`).
- [x] 18.2 Look-ahead tested by **prefix stability** — every time-dependent column rebuilt from
      truncated data and compared at the cut. An earlier correlation-based test was invalid
      (ratio variables containing RV_t give false positives) and was replaced.
- [x] 18.3 `ScaleFactor_HL` flagged as a full-sample in-sample constant.

---

# PART 3 — Modelling deliverables (session 2026-08-24)

Full detail and results tables: `RESEARCHER_A_SCOPE.md`. Decisions Researcher B depends on:
`RESEARCHER_A_DECISIONS.md`. All four of A's remaining plan modules complete this session.

## Phase 19 — Baseline GARCH / GJR-GARCH / EGARCH ✅
- [x] 19.1 Six specs per index (Normal/t/skew-t × symmetric/asymmetric), AR(1) mean.
      AR(1) added after a constant-mean pass left HSI's residuals autocorrelated
      (Ljung-Box p=5e-5) - matches `FEATURE_SETS.csv`'s stated "AR(1)-GJR-GARCH".
- [x] 19.2 GJR-skewt selected as primary (matches the plan's GARCH-EVT stage-1 spec and the
      EDA's asymmetry findings); EGARCH-skewt has marginally lower AIC everywhere but is
      reported as a comparator, not substituted - swapping the plan's named model is B's call.
- [x] 19.3 Wrote `06_REALIZED_MEASURES/<CODE>_std_resid.csv` - **the file GARCH-EVT stage 2
      reads**. This was the critical-path deliverable for unblocking B.

## Phase 20 — Realized GARCH ✅
- [x] 20.1 Implemented directly (not in `arch`) per Hansen, Huang & Shek (2012) log-linear
      spec, verified against the paper's own equations before coding.
- [x] 20.2 Fitted on `RV_Scaled`, not raw session RV - avoids a by-index scale mismatch
      (1.71x-3.04x) contaminating the leverage/persistence parameters.
- [x] 20.3 NKY's 2016-17 gap (842 of 3,558 days, 24%) handled explicitly: `h_{t-1}` substitutes
      for missing `x_{t-1}` in the RECURSION only; the likelihood never sees the imputed
      value. Affected forecast rows carry `Reason="RV_imputed_in_recursion"`.
- [x] 20.4 All six converged; phi in [0.90, 1.02] (theory-consistent), tau1 negative
      everywhere (leverage, consistent with EDA's -0.12 to -0.20 finding).

## Phase 21 — Rolling out-of-sample forecast engine ✅
- [x] 21.1 Genuine walk-forward: expanding-window re-estimation every 21 trading days
      (chosen over fixed rolling given GPH d=0.50-0.63 long memory), daily state update via
      `arch`'s `.fix(params).forecast(horizon=1)` - real 1-step-ahead forecasts throughout,
      not multi-step projections held over the refit gap.
- [x] 21.2 Run for GJR-skewt across all six indices, full sample-B window: 3,150-3,267
      forecasts per index, 150-156 refits each.
- [x] 21.3 Realized GARCH NOT walk-forward re-estimated - custom optimiser cost (~85s/fit)
      makes a monthly refit ~18 hours; documented as an overnight-batch candidate, not
      silently presented as equivalent to the GARCH-family rolling output.

## Phase 22 — Robustness checks ✅
- [x] 22.1 Sub-sample stability: persistence drops post-COVID on every index (e.g. SPX
      0.987->0.973, NKY 0.981->0.921); skew strengthens on five of six.
- [x] 22.2 Innovation distribution: Normal->t decisive everywhere (dAIC 252-466); t->GJR-skewt
      decisive on five indices, weak on HSI (dAIC 63) - HSI has the least Engle-Ng asymmetry.
- [x] 22.3 Sampling-frequency sensitivity: Hansen-Lunde scale factor moves 3-6% at 10-min RV,
      17-28% at 30-min - the 5-min choice is doing real work, not an arbitrary convention.
- [x] 22.4 Refit-cadence sensitivity: 21-day vs 63-day correlation 0.99998 - confirms the
      rolling engine's cadence is a compute-cost choice, not a result-changing one.

## Phase 23 — Forecast-file contract and cross-index decisions ✅
- [x] 23.1 `10_SCRIPTS/26_forecast_io.py` - schema enforced in code (`validate()`,
      `read_forecasts()`, `write_forecasts()`, `eval_frame()`), not just documented. Both real
      model outputs pass validation and a full QLIKE/breach-rate join test.
- [x] 23.2 Synthetic placeholder forecast files built and superseded once real output existed.
- [x] 23.3 Non-synchronous-session convention and the crisis-coverage statement written to
      `RESEARCHER_A_DECISIONS.md`.

## Phase 24 — Modelling figures ✅
- [x] 24.1 Seven figures in `09_FIGURES/` (11-17), each backing a specific number from
      Phases 19-23 rather than illustrating in the abstract. See `RESEARCHER_A_SCOPE.md`
      "Figures" section for the full index.

## Phase 25 — Window-length and horizon-extension robustness ✅ (closes an Exec-Summary gap)
- [x] 25.1 `EXECUTIVE_SUMMARY_ADDENDUM.md` written - reconciles four places delivery differs
      from the Executive Summary's own text (free data sources, 6 indices not 1-3, a fuller
      Realized GARCH spec than the doc's simplified equation, and this phase closing the two
      robustness items the doc names but the first robustness pass had not yet covered).
- [x] 25.2 Window-length sensitivity (SPX, expanding vs fixed 2y vs fixed 5y): correlation with
      expanding drops to ~0.954, mean abs rel diff ~10% - unlike refit-cadence, this **matters**.
      2020 COVID window shows the mechanism: fixed 5y overshoots expanding by up to +2pp at the
      peak, then both fixed windows undershoot for months after as COVID ages out of lookback.
- [x] 25.3 Horizon extension (all six indices, 1-day vs 5-day cumulative GJR-skewt): mean
      annualised vol agrees to within 0.1-0.7pp across horizons - stable long-run estimate.
      Caveat documented: QLIKE_5d < QLIKE_1d on every index is a scale artifact of QLIKE on a
      coarser target, not evidence of better 5-day forecasting - never compare the two directly.
- [x] 25.4 `29_rolling_forecast_engine.py` extended with `window_size` and `horizon` params,
      backward-compatible (verified: refactored expanding/horizon=1 output matches the
      previously delivered production files to float-precision noise, 5e-12 max abs diff).
- [x] 25.5 6 new contract-format forecast files: `20_FORECASTS/GJR-skewt-h5__<CODE>_forecasts.csv`.
- [x] 25.6 2 new figures: `18_window_length_sensitivity.png`, `19_horizon_extension.png`.

**All six of Researcher A's Exec-Summary lead rows are now complete with no open items inside
them.** See `EXECUTIVE_SUMMARY_ADDENDUM.md` for the row-by-row reconciliation table.

---

## Open items for the modelling stage
- [x] ~~NKY Realized GARCH must handle the 2016-17 gap~~ - done, see Phase 20.3.
- [x] ~~Decide and state the treatment of non-synchronous sessions~~ - done, see Phase 23.3.
- [x] ~~Window-length and horizon-extension robustness~~ - done, see Phase 25.
- [x] ~~Re-estimate the POT threshold on the actual GARCH residuals once stage 1 is fitted~~ -
      done by B, `41_evt_threshold.py` (genuine GJR-GARCH residuals, not the EDA-stage
      rolling-standardised-return stand-in). xi narrower and still positive: 0.05-0.16 at
      q=0.95, vs the 0.15-0.25 stand-in band. See Phase 26.
- [x] ~~Realized GARCH walk-forward re-estimation at a coarser (quarterly/annual)
      cadence~~ - done 2026-08-29, `28_realized_garch.py` (annual expanding-window refit,
      ~13 refits/index). Materially changed the headline VaR result - see Phase 26.
- [x] ~~Unit tests for model-fitting functions~~ - done 2026-08-26, `tests/` (39 pytest tests:
      the EVT/backtest/loss-function library, the forecast-file contract, the Basel zone fn).
      Now 45 tests, all passing (2026-08-30).
- [x] ~~Environment reproducibility~~ - `requirements.txt` / `requirements_B.txt` exact pins +
      `50_reproducibility_audit.py`'s environment check. A Dockerfile was tried and dropped
      2026-08-26 (never build-verified; not worth maintaining unverified).
- [ ] Optional: Dukascopy ASK side for mid-price RV; the 5 remaining FRED series.

---

# PART 4 — Look-ahead fixes, strict common window, review closeout (2026-08-29 to 08-30)

Full detail: `EXECUTIVE_SUMMARY_ADDENDUM.md` (2026-08-29 and 2026-08-30 follow-ups). Merged via
GitHub PRs #2, #3, #4 on `main`. Summary only, so this tracker does not fall out of step again:

## Phase 26 — P0 look-ahead fixes and their headline consequence ✅
- [x] 26.1 Causal Hansen-Lunde scaling (`RV_Scaled_Causal`, expanding factor, strictly-prior
      observations only) replaces the full-sample-constant `RV_Scaled` everywhere a forecast
      or evaluation consumes it.
- [x] 26.2 Realized GARCH re-estimated walk-forward (annual expanding-window refit, ~13
      refits/index) instead of once on the full sample - closes the open item above. Headline
      consequence: RealGARCH's 99% VaR breach rate got materially worse under the fair refit
      (multiple markets moved to RED on the Basel traffic light) while its QLIKE advantage over
      GJR-skewt survived intact - the "variance accuracy != tail calibration" story is now
      demonstrated honestly rather than resting on an in-sample parameter advantage.
- [x] 26.3 NKY session-close fix (TSE extended its close 15:00->15:30 JST from 2024-11-05);
      full NKY realized-measure pipeline rebuilt.
- [x] 26.4 HAC/Newey-West Diebold-Mariano fix - the long-run-variance correction was a silent
      no-op at h=1, i.e. every DM call actually made in this project.
- [x] 26.5 EVT exceedance-dependence diagnostic added (Ljung-Box, runs test, Ferro-Segers
      extremal index) - genuine tail clustering found in HSI and UKX.

## Phase 27 — Strict common evaluation window ✅
- [x] 27.1 B flagged (from independent review of PR #2) that RealGARCH's walk-forward burn-in
      moved its valid-date window out of step with GJR-skewt/GARCH-EVT per index (DAX lost 502
      days), distorting pooled breach rates and crisis-window coverage.
- [x] 27.2 `strict_window()` added to `47_evaluation.py` (per-index intersection of the three
      variance models' valid dates, QR deliberately excluded); propagated by B into
      `48_crisis_regime.py` and `49_model_comparison.py`. Both unrestricted and strict tables
      are kept side by side - nothing was overwritten, only added alongside.
- [x] 27.3 `50_reproducibility_audit.py` extended with two new cross-script consistency checks
      for the strict window; audit now 31 checks (30/31 locally - only the expected
      Python-version environment mismatch fails; 31/31 on B's pinned environment).

## Phase 28 — Three issues closed from B's review of the merged PR ✅
- [x] 28.1 Exceedance-dependence console flag in `41_evt_threshold.py` was gated on
      Ljung-Box and a saturated Ferro-Segers estimate alone, ignoring the runs test's sign -
      it mislabelled SPX as clustered and missed UKX. Fixed to require the runs test to agree
      on direction and at least one test to be significant; reproduces {HSI, UKX}.
- [x] 28.2 Stale xi range (0.15-0.25, the EDA-stage stand-in) in
      `25_build_figure_and_handoff_docs.py` / `Handoff_to_Researcher_B.pdf` updated to cite
      both the stand-in and the confirmed genuine-residual range (0.05-0.16).
- [x] 28.3 `RealGARCH_FullSample_INSAMPLE__*` archive-comparison forecast files were
      discoverable in the live `20_FORECASTS` contract folder despite their Spec string saying
      NOT-A-RESULT. Moved to `20_FORECASTS/_ARCHIVE_NOT_A_RESULT/` (both locally and on Drive).

**Coding is complete against every plan deliverable as of 2026-08-30.** The only items left
open anywhere in this tracker are the two optional/no-honest-alternative ones above (Dukascopy
ASK side, the 5 remaining FRED series) and the write-up-only note (Bartlett-kernel DM change
needs a sentence in the methodology section - not a code task).

---

# PART 5 — Final output selection and release review (2026-08-31)

## Phase 29 — "Final outputs" folder for the writing team ✅

Arham asked for the four-folder `Output/` section to be replaced by a single folder holding
only the figures and tables that go in the paper (5-8 figures, 5-7 tables), with Claude
asked whether each result is worth including.

- [x] 29.1 All 16 figures and 42 tables reviewed individually. First cut was 7 figures and
      6 tables; `49_loss_metrics.png` was added as an eighth after review showed RMSE gives
      the opposite sign to QLIKE on SPX, UKX and DAX — a second instance of the paper's own
      "the metric determines the answer" argument.
- [x] 29.2 Selection rationale written to `results/FINAL_OUTPUTS.md`, including what was
      cut and why, and three results that belong in the text as a sentence rather than as a
      table (`49_mcs`, `49_dm_pinball`, `41_exceedance_dependence`).
- [x] 29.3 Drive restructured: `Output/` renamed `Final Output/`, holding only `figures/`
      and `tables/`. Raw sub-folders trashed after confirming every file is tracked in git
      or present locally (`06_MODEL_FITS` was not in git but exists in
      `Datasets/06_REALIZED_MEASURES/`; its six `realized_garch_fit` Drive copies were stale
      by ~15 hours).

## Phase 30 — Two figure bugs found by verifying the manifest against the CSVs ✅

- [x] 30.1 `49_loss_metrics.png` carried its caption as a hard-coded string literal while
      its bars were computed from the data. It claimed RMSE said RealGARCH was 0.6% *better*
      on UKX; it is 3.6% **worse** (QLIKE +17.9%, DM 4.33). The caption contradicted its own
      chart and inverted the sign. Title is now derived from the data, and the exhibit index
      is chosen from it rather than named in the string.
- [x] 30.2 `49_qlike_vs_breach.png` took its y-axis from the strict common window but its
      x-axis from `47a_volatility_losses.csv`, the unrestricted file — 8 of 18 plotted cells
      had a different sample on each axis (DAX 3172 vs 2755). Now loads the strict file.
      Visually negligible (QLIKE moves ≤1.4% relative, no ranking changes) but the figure is
      now internally consistent.

## Phase 31 — the writing team's release review (Arham and Absar) ✅

Fourteen points raised. Two were already closed by Phase 30 and the review predates them.
The rest were verified against the CSVs before acting:

- [x] 31.1 Pinball DM significant count corrected 3/60 → **4/60** (DAX RealGARCH vs
      QR-Range; NKY GARCH-EVT and GJR-skewt vs RealGARCH; HSI GARCH-EVT vs GJR-skewt).
- [x] 31.2 Tail-clustering attribution corrected to **UKX and HSI** (matching 28.1, which
      already had it right). SPX's Ljung-Box is marginal at 0.049 but its runs statistic
      points the other way (z=+0.25), so SPX is not classified as clustered.
- [x] 31.3 `43_var_breaches.png` demoted to the appendix list: its panel titles are the
      per-model GARCH-EVT rates (SPX 1.203%) not the strict ones (1.231%), so the claim that
      every selected figure was strict was false.
- [x] 31.4 `45_qr_calibration.png` had the same problem (DAX QR-Range 1.133% vs 1.049%
      strict). Regenerated on the strict window as `51_qr_calibration_strict.png`.
- [x] 31.5 **RealGARCH-t vs RealGARCH-skew-t robustness added** — the main gap in the
      shortlist. `51_final_release_exhibits.py`, paired dates. Skew-t cuts the 99% breach
      rate on all six (mean 0.40pp) at essentially unchanged QLIKE, so the tail failure is
      the innovation distribution rather than the realized measure. This is what licenses
      reading `VaR = mu + sigma*q` as "a better sigma does not buy a better q".
- [x] 31.6 Added a banner to `FINAL_OUTPUTS.md` telling readers to cite the CSVs, not the
      manifest — a prose summary is always one regeneration behind.

**Final selection: 8 figures, 7 tables, every exhibit on the strict common window.**

## Open — Researcher B's writer handoff, not a code task

`HANDOFF_TO_WRITERS.md` sits on Researcher B's (Maham's) account, outside this repo, so it
was not edited and the writing team should not cite it until B refreshes it. It still quotes
unrestricted GJR QLIKE (0.2695, 0.2517, 0.1567, 0.1893, 0.2447, 0.1849) where the strict
values are **0.2702, 0.2517, 0.1567, 0.1920, 0.2477, 0.1849**; still says 3 of 60 for the
pinball DM; and still carries the obsolete EVT full-sample-mu limitation. Refreshed strict
percentages: DAX QLIKE improvement **7.2%**, DAX RMSE disadvantage **-0.6%**, mean QLIKE
improvement across markets **11.7%**. The scientific interpretation does not change.
