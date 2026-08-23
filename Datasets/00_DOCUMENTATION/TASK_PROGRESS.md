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

## Open items for the modelling stage
- [ ] NKY Realized GARCH must handle the 2016-17 gap: rolling windows use the last *available*
      observations; no forecasts can be produced or evaluated inside it.
- [ ] Decide and state the treatment of non-synchronous sessions before any pooled analysis.
- [ ] Re-estimate the POT threshold on the actual GARCH residuals once stage 1 is fitted.
- [ ] Optional: Dukascopy ASK side for mid-price RV; the 5 remaining FRED series.
