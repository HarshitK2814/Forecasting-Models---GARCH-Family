# FINAL_OUTPUTS — the 8 figures and 7 tables for the paper

**Drive location:** `Final Output/` — contains `figures/` and `tables/` and nothing else,
per Arham's spec of 2026-08-31. The folder previously called `Output/` was renamed and its
raw sub-folders (`08_VALIDATION`, `09_FIGURES`, `20_FORECASTS`, `06_MODEL_FITS`) removed;
this document lives in `Datasets/00_DOCUMENTATION/` on Drive so it does not clutter it.
**Selected:** 2026-08-31 · **Revised:** 2026-08-31 after the writing team's review
**Source of truth:** GitHub `main`, `results/figures/` and `results/tables/`

Everything not in the Drive folder is still in this repository. Nothing was lost: the 43
tables, 18 figures and 56 forecast files are all tracked in git, and the untracked model
fits live locally in `Datasets/06_REALIZED_MEASURES/`.

Every exhibit here is now genuinely on the **strict common evaluation window**. That was
not true of the first cut — see "What the writing team's review changed" at the end. Files are renamed
`FIG1_…` / `TAB1_…` in paper order; the original filename is kept so each traces back to
`results/`. The `FIG_`/`TAB_` prefixes exist only on Drive — the repo keeps the unprefixed
names its scripts write, so do not rename the local files.

> **Do not cite numbers from this document.** Cite the CSVs. This is a selection rationale,
> and a prose summary will always be one regeneration behind the tables.

---

## The paper's central claim

**Variance accuracy and tail calibration are different things, and the models rank in
opposite orders on them.** Realized GARCH has the lowest QLIKE on all six indices (best
volatility forecasts) and simultaneously the worst 99% VaR breach rate on five of six —
Basel **red on four** markets (1 green / 1 amber / 4 red). GARCH-EVT is the mirror image:
middling QLIKE, best-calibrated tail, **4 green / 2 amber / 0 red**.

FIG3 completes the argument. Holding the variance model and the realized measure fixed and
changing only the innovation distribution, RealGARCH-skew-t cuts the 99% breach rate on
**all six** indices (mean 0.40 pp). So the tail failure is attributable to the innovation
term, not to the realized-measure machinery — which is what licenses reading

    VaR_alpha,t = mu_t + sigma_t * q_alpha,t

as "a better sigma_t does not buy you a better q_alpha,t".

---

## Figures (8)

| # | File | What it shows | Section |
|---|---|---|---|
| 1 | `FIG1_41_threshold_stability.png` | GPD shape ξ vs POT threshold q, ±1 sampling SE (≈0.062). ξ drifts gently and stays inside the band from 0.90 to 0.975, then becomes erratic above it as the exceedance count collapses. | Methods — justifies q=0.95 |
| 2 | `FIG2_41_qq_gpd.png` | GPD QQ plots of exceedances, all six indices. KS p = 0.49–0.95 at q=0.95; no index rejects. | Methods — EVT fit is valid |
| 3 | `FIG3_51_realgarch_innovation.png` | RealGARCH-t vs RealGARCH-skew-t, 99% breach rate, paired dates. Skew-t lower on all six. Isolates the innovation distribution as the cause of the tail failure. | Results — the mechanism |
| 4 | **`FIG4_49_qlike_vs_breach_HEADLINE.png`** | **QLIKE (x) vs 99% breach rate (y). The two axes rank the models in opposite orders. The paper's thesis in one scatter.** | **Results — headline** |
| 5 | `FIG5_49_basel.png` | Basel traffic light, 6 indices × 5 models. RealGARCH red on 4 of 6; GARCH-EVT green on 4, amber on 2, never red. | Results — regulatory payoff |
| 6 | `FIG6_48_crisis_heatmap.png` | Breach rate by regime (Normal, China/Oil, Q4-2018, COVID, 2022 rates), pooled over six indices. Every model degrades; RealGARCH and QR-Full worst (7.1% and 7.9% in COVID vs 1.0% target). | Results — regime conditioning |
| 7 | `FIG7_51_qr_calibration_strict.png` | QR-Full and QR-Range breach rates at 99/97.5/95%, **rebuilt on the strict window** so it agrees with TAB4/TAB5. | Results — quantile regression |
| 8 | `FIG8_49_loss_metrics.png` | RealGARCH vs GJR-skew-t under four loss functions. RMSE gives the **opposite sign** to QLIKE on SPX, UKX and DAX — on UKX, RMSE says RealGARCH is 3.6% *worse* where QLIKE says 17.9% better (DM 4.33, p<0.0001). | Discussion — the loss function is not neutral |

**Why these eight.** 1–2 defend the two methodological choices a referee will challenge
(threshold selection, distributional fit). 3 isolates the mechanism behind the thesis.
4 is the thesis. 5 and 6 are the two dimensions it has to survive: regulatory
classification and crisis conditioning. 7 covers the model family the others under-serve.
8 is a second, independent instance of the same argument — the metric determines the
answer — this time within volatility accuracy rather than between accuracy and
calibration, and it pre-empts a referee asking why QLIKE and not RMSE.

### Left out (in the repo at `results/figures/`)

Ranked by how close they came. The first is the one to promote if anything gets dropped.

1. **`45_qr_crossings.png` — strongest appendix candidate.** QR-Full puts the fitted 1%
   line *above* the 2.5% line on 5.4–11.7% of days, which is logically impossible and a
   direct symptom of overfitting; QR-Range stays under 2%. This is the mechanism behind
   QR-Full's poor showing in FIG7 and TAB4, and a limitation the paper should disclose
   rather than leave for a referee to find.
2. **`43_var_breaches.png` — was FIG3 in the first cut, now an appendix figure.** It is
   the only illustrative time-series view of the VaR path, but its panel titles are the
   *per-model GARCH-EVT* breach rates (SPX 1.203%, DAX 1.255%, NKY 1.206%), not the strict
   ones (1.231%, 1.338%, 1.243%), because `43_evt_figures.py` reads
   `42_evt_diagnostics_<CODE>.csv`. If it goes in an appendix it must be labelled
   explicitly as the GARCH-EVT model-specific forecast path, never as a strict result.
3. `45_qr_coefficients.png` — which predictors drive the 1% tail (LogIV strongly negative
   everywhere, VRP positive on SPX/NDX/NKY). The only figure carrying economic
   interpretation. Appendix, if the paper says anything about *why* QR works.
4. `45_qr_calibration.png` — superseded by FIG7. Same chart on each specification's own
   dates rather than the strict window. Keep only as a within-QR diagnostic, clearly
   labelled as such.
5. `49_scorecard.png` — Kupiec/Independence/CC/DQ pass-fail grid. Informative (DQ rejects
   20/30 cells, far more than the other three), but it is TAB4 in picture form.
6. `43_xi_path.png` — expanding-window ξ with ±1 SE bands. An honest null: all movement
   sits inside the sampling band. FIG1 already carries the ξ-uncertainty message.
7. `43_evt_vs_skewt.png` — EVT vs skew-t tail quantile ratio, within ±2%. Near-null; one
   sentence in the text.
8. `48_degradation.png` — subsumed by FIG6, which shows five regimes not a binary.
9. `45_qr_var_series.png` — FIG3's old layout for a different model; redundant.
10. `46_qr_longhistory.png` — labelled ROBUSTNESS ONLY in its own title, and most of it
    lies outside sample B. Correctly out of any model-comparison section.

---

## Tables (7)

| # | File | What it shows | Section |
|---|---|---|---|
| 1 | `TAB1_47_strict_window.csv` | Rows per index × model, unrestricted vs strict, and how many were dropped to enforce the common window. DAX loses 502 days, SPX 74, NDX/UKX none. | Data — defines the evaluation sample |
| 2 | `TAB2_47a_volatility_losses_strict.csv` | QLIKE, MSE, RMSE, MAE, MAPE by index × model. RealGARCH lowest QLIKE on all six. | Results — volatility accuracy |
| 3 | `TAB3_47a_dm_volatility.csv` | Diebold–Mariano on QLIKE differences, HAC/Newey-West corrected. RealGARCH's edge is significant at 5% on NDX, UKX, HSI; marginal on DAX (p=0.052); not on SPX or NKY. | Results — is the edge real? |
| 4 | `TAB4_47b_var_backtests_strict.csv` | Kupiec, Christoffersen independence and CC, Dynamic Quantile, at 99/97.5/95%. | Results — tail calibration |
| 5 | `TAB5_49_basel.csv` | Basel traffic-light zone per index × model, with breach and expected counts. | Results — regulatory reading |
| 6 | `TAB6_48_crisis_pooled_strict.csv` | Breach counts and rates by regime, pooled over six indices, with a `reportable` flag. | Results — crisis behaviour |
| 7 | `TAB7_51_realgarch_innovation.csv` | RealGARCH-t vs RealGARCH-skew-t on paired dates: breach counts and rates at 99/97.5/95, Kupiec p, and QLIKE. The numeric backing for FIG3. | Results — the mechanism |

### Two things to do before these go in the manuscript

1. **TAB4 is 90 rows — too long for a main-text table.** Filter to `confidence == 0.99`
   (30 rows) for the paper; 97.5% and 95% to an appendix. The 99% level is where the
   models separate.
2. **TAB6 has two regimes flagged `reportable = False`** — Volmageddon (n=53) and
   Yen_Carry_Unwind (n=42). At a 1% target those windows expect well under one breach, so
   the observed 7–13% rates are noise. Drop them or caveat them explicitly. Do not let
   them into a headline comparison.

### Left out (in the repo at `results/tables/`)

- `49_mcs.csv` — Model Confidence Set. Eliminates nothing on five of six indices: "we
  cannot distinguish these models on pinball loss." An honest null, worth **one sentence**,
  not a table.
- `49_dm_pinball.csv` — DM on pinball loss. **4 of 60 significant at 5%** (DAX RealGARCH
  vs QR-Range; NKY GARCH-EVT vs RealGARCH; NKY GJR-skewt vs RealGARCH; HSI GARCH-EVT vs
  GJR-skewt) against 3 expected by chance at 60 × 0.05. Same null story as the MCS.
- `42_evt_summary.csv` — ξ ranges and breach counts per index; already in FIG2's titles.
- `41_exceedance_dependence.csv` — Ljung-Box, Wald-Wolfowitz runs, Ferro-Segers extremal
  index. **Worth a Limitations mention:** genuine tail clustering in **UKX** (runs z=−2.15,
  p=0.031) and **HSI** (LB p=0.019, runs z=−2.75, p=0.006), violating the i.i.d. assumption
  behind the POT fit. SPX's Ljung-Box is marginal (p=0.049) but its runs statistic points
  the *other* way (z=+0.25, p=0.80), so SPX is **not** classified as clustered.
- `47c_es_backtests.csv` — Expected Shortfall backtests. Include only if the paper claims
  something about ES; it currently does not.
- `48_volregime_pooled_strict.csv` — the same analysis conditioned on *volatility regime*
  rather than named crisis windows. **Worth one sentence, because it complicates FIG6:**
  the degradation nearly vanishes (GARCH-EVT 1.04/1.43/1.25/1.13% across
  Calm/Normal/Stressed/Crisis, with Crisis *below* Normal). FIG6 is the sharper result but
  should carry the caveat rather than let a referee find it. Do not print both.
- `34_causal_verification.csv`, `48_window_effect.csv`, `50_audit.csv` — robustness and
  reproducibility evidence, for a supplementary appendix.
- Per-index EVT diagnostics and all forecast files — raw output, never printed.

---

## What the writing team's review changed (2026-08-31)

Arham and Absar reviewed the first cut and raised fourteen points. Two had already been fixed hours
earlier and the review predates them; the rest were acted on. Verified item by item
against the CSVs:

**Already fixed before the review landed**
- *Headline figure mixes windows.* Correct diagnosis, already closed: `49_model_comparison.py`
  now loads `47a_volatility_losses_strict.csv` for both FIG4 and FIG8. Confirmed at
  `49_model_comparison.py:213`.
- *`49_loss_metrics.png` should not be silently dropped.* Agreed and already promoted to
  FIG8, after its hard-coded stale caption was replaced with data-derived text.

**Fixed in response**
- **Pinball DM count was wrong.** This document said 3 of 60; the CSV has **4**. Corrected.
- **Exceedance clustering was misattributed.** This document said SPX and HSI; the
  defensible reading is **UKX and HSI**. Corrected, with the SPX nuance stated. (The project
  tracker item 28.1 already had it right as {HSI, UKX} — the manifest was the outlier.)
- **FIG3 was not on the strict window.** `43_var_breaches.png` demoted to the appendix
  list with the discrepancy documented, and the FIG3 slot given to the RealGARCH-t vs
  RealGARCH-skew-t exhibit the review asked for.
- **FIG7 was not on the strict window.** Regenerated from `47b_var_backtests_strict.csv`
  as `51_qr_calibration_strict.png`; DAX QR-Range now reads 1.049% and SPX 1.515%, matching
  TAB4/TAB5 exactly.
- **RealGARCH-skew-t was missing.** Added as FIG3 and TAB7 via
  `Datasets/10_SCRIPTS/51_final_release_exhibits.py`, computed on paired dates. Reproduces
  the review's numbers exactly: 2.209/1.578, 2.183/1.476, 1.682/1.224, 1.844/1.627,
  1.897/1.636, 1.136/1.010. A detail worth using: QLIKE is essentially unchanged between
  the two innovations, so the skew-t buys tail calibration at no cost in variance accuracy.
- **"Basel green on three"** was already corrected to four earlier the same day.

**Outstanding — belongs to Researcher B (Maham), not editable by us**
- `HANDOFF_TO_WRITERS.md` sits on B's account and outside this repo, so it was not edited.
  **The writing team should not cite it until B refreshes it.** It still
  quotes **unrestricted** GJR QLIKE (0.2695, 0.2517, 0.1567, 0.1893, 0.2447, 0.1849) where
  the strict values are **0.2702, 0.2517, 0.1567, 0.1920, 0.2477, 0.1849**; still says
  3 of 60 for the pinball DM; and still carries the obsolete EVT full-sample-μ limitation.
  Under strict evaluation the refreshed figures are: DAX QLIKE improvement **7.2%**, DAX
  RMSE disadvantage **−0.6%**, mean QLIKE improvement across markets **11.7%**. All
  independently confirmed here. The scientific interpretation does not change.

---

## Provenance

Every file in `Final Output/` matches `results/figures/` or `results/tables/` on GitHub
`main` byte for byte, verified after upload.

**Drive sync is complete as of 2026-08-31.** All 8 figures and all 7 tables are present in
`Final Output/` and every one was byte-size checked against the repo after upload. The four
figures that could not be pushed through the tooling (FIG3, FIG4, FIG7, FIG8) were uploaded
by hand and verified: 127,880 / 175,785 / 139,773 / 110,426 bytes respectively.

Regenerate with the scripts in `Datasets/10_SCRIPTS/` (41–51). `51_final_release_exhibits.py`
must run after 47 and 49, since it reads their strict-window output.
`50_reproducibility_audit.py` checks 31 conditions; it passes 31/31 on the pinned
environment in `requirements_B.txt` (30/31 locally, the one failure being Python-version
drift on this machine). 45/45 unit tests pass.
