# FINAL_OUTPUTS — the 8 figures and 6 tables for the paper

**Drive location:** `Output/FINAL_OUTPUTS/` (`figures/`, `tables/`)
**Selected:** 2026-08-31 · **Source of truth:** GitHub `main`, `results/figures/` and `results/tables/`

This is the shortlist for the manuscript, cut from 16 figures and 42 tables. All 16 figures
were opened and reviewed individually before the cut, not filtered by filename. Everything
here is on the **strict common evaluation window** unless noted, so every number in the
paper is comparable across models. Files are renamed `FIG1_…` / `TAB1_…` in paper order;
the original filename is kept in the name so it traces back to the full results folders.

---

## The paper's central claim

**Variance accuracy and tail calibration are different things, and the models rank in
opposite orders on them.** Realized GARCH has the lowest QLIKE on all six indices (best
volatility forecasts), and simultaneously the worst 99% VaR breach rate on five of six —
Basel *red* on four markets. GARCH-EVT is the mirror image: middling QLIKE, best-calibrated
tail, Basel *green* on four and never red.

Every selection below either sets that claim up, demonstrates it, or qualifies it, and
every number quoted was checked against the CSVs on 2026-08-31 rather than carried over
from an earlier draft.

---

## Figures (8)

| # | File | What it shows | Section |
|---|---|---|---|
| 1 | `FIG1_41_threshold_stability.png` | GPD shape ξ vs POT threshold q, ±1 sampling SE (≈0.062). ξ drifts gently and stays inside the band from 0.90 to 0.975, then becomes erratic above it as the exceedance count collapses. | Methods — justifies q=0.95 |
| 2 | `FIG2_41_qq_gpd.png` | GPD QQ plots of exceedances, all six indices. KS p = 0.49–0.95 at q=0.95; no index rejects. | Methods — EVT fit is valid |
| 3 | `FIG3_43_var_breaches.png` | GARCH-EVT 99% VaR vs realised returns, six panels, breaches marked, crisis windows shaded. | Results — what the model does |
| 4 | **`FIG4_49_qlike_vs_breach_HEADLINE.png`** | **QLIKE (x) vs 99% breach rate (y). The two axes rank the models in opposite orders. This is the paper's thesis in one scatter.** | **Results — headline** |
| 5 | `FIG5_49_basel.png` | Basel traffic light, 6 indices × 5 models. RealGARCH red on 4 of 6; GARCH-EVT green on 4, amber on 2, never red. | Results — regulatory payoff |
| 6 | `FIG6_48_crisis_heatmap.png` | Breach rate by regime (Normal, China/Oil, Q4-2018, COVID, 2022 rates), pooled over six indices. Every model degrades; RealGARCH and QR-Full worst (7.1% and 7.9% in COVID vs 1.0% target). | Results — regime conditioning |
| 7 | `FIG7_45_qr_calibration.png` | QR-Full and QR-Range breach rates at 99/97.5/95%. The fifth model family's own calibration result. | Results — quantile regression |
| 8 | `FIG8_49_loss_metrics.png` | RealGARCH vs GJR-skew-t under four loss functions. RMSE gives the **opposite sign** to QLIKE on SPX, UKX and DAX — on UKX, RMSE says RealGARCH is 3.6% *worse* where QLIKE says 17.9% better (DM 4.33, p<0.0001). | Discussion — the loss function is not neutral |

**Why these eight and not the others.** 1–2 defend the two methodological choices a
referee will actually challenge (threshold selection, distributional fit). 3 is the only
figure showing the raw series, which the reader needs once before being shown aggregates.
4 is the thesis. 5 and 6 are the two dimensions the thesis has to survive: regulatory
classification and crisis conditioning. 7 covers the model family the other figures
under-serve. 8 is a second, independent instance of the paper's own argument — the metric
you choose determines the answer — this time within volatility accuracy rather than
between accuracy and calibration. It also pre-empts a referee asking why QLIKE and not
RMSE: the honest answer is that the choice matters and QLIKE is the defensible one for a
latent-variance target.

### The eight left out (available in `Output/09_FIGURES/` if wanted)

Ranked by how close they came. The first is the one to promote if anything gets dropped.

1. **`45_qr_crossings.png` — strongest appendix candidate.** QR-Full puts the fitted 1%
   line *above* the 2.5% line on 5.4–11.7% of days, which is logically impossible and a
   direct symptom of overfitting; QR-Range stays under 2%. This is the mechanism behind
   QR-Full's poor showing in FIG7 and TAB4, and it is a limitation the paper should
   disclose rather than leave for a referee to find. Out only because eight is Arham's
   ceiling.
2. `45_qr_coefficients.png` — which predictors drive the 1% tail (LogIV strongly negative
   everywhere, VRP positive on SPX/NDX/NKY). The only figure carrying economic
   interpretation rather than model comparison. Belongs in an appendix if the paper says
   anything about *why* QR works.
3. `49_scorecard.png` — Kupiec/Independence/CC/DQ pass-fail grid. Genuinely informative
   (DQ rejects 20/30 cells, far more than the other three tests), but it is the same
   information as TAB4 in picture form. Keep the table. Promote the figure only if you
   want the DQ point made visually.
4. `43_xi_path.png` — expanding-window ξ over the forecast period, with ±1 SE bands. An
   honest null: all movement sits inside the sampling band. Good discipline, but FIG1
   already carries the ξ-uncertainty message.
5. `43_evt_vs_skewt.png` — ratio of EVT to skew-t tail quantile, within ±2% of each other.
   Near-null. One sentence in the text.
6. `48_degradation.png` — subsumed by FIG6, which shows five regimes instead of a
   normal/crisis binary.
7. `45_qr_var_series.png` — QR-Range VaR series. Same layout as FIG3 for a different
   model; redundant once FIG3 is in.
8. `46_qr_longhistory.png` — explicitly labelled ROBUSTNESS ONLY in its own title, and
   most of it lies outside sample B where no other model is evaluated. Correctly out of
   any model-comparison section.

---

## Tables (6)

| # | File | What it shows | Section |
|---|---|---|---|
| 1 | `TAB1_47_strict_window.csv` | Rows per index × model, unrestricted vs strict, and how many were dropped to enforce the common window. DAX loses 502 days, SPX 74, NDX/UKX none. | Data — defines the evaluation sample |
| 2 | `TAB2_47a_volatility_losses_strict.csv` | QLIKE, MSE, RMSE, MAE, MAPE by index × model. RealGARCH lowest QLIKE on all six. | Results — volatility accuracy |
| 3 | `TAB3_47a_dm_volatility.csv` | Diebold–Mariano on QLIKE differences, HAC/Newey-West corrected. RealGARCH's edge is significant at 5% on NDX, UKX, HSI; marginal on DAX (p=0.052); not on SPX or NKY. | Results — is the edge real? |
| 4 | `TAB4_47b_var_backtests_strict.csv` | Full backtest battery: Kupiec, Christoffersen independence and CC, Dynamic Quantile, at 99/97.5/95%. | Results — tail calibration |
| 5 | `TAB5_49_basel.csv` | Basel traffic-light zone per index × model, with breach counts and expected counts. | Results — regulatory reading |
| 6 | `TAB6_48_crisis_pooled_strict.csv` | Breach counts and rates by regime, pooled over six indices, with a `reportable` flag. | Results — crisis behaviour |

### Two things to do to these before they go in the manuscript

1. **TAB4 is 90 rows — too long for a main-text table.** Filter to `confidence == 0.99`
   (30 rows, one per index × model) for the paper and put the 97.5% and 95% rows in an
   appendix. The 99% level is where the models actually separate.
2. **TAB6 has two regimes flagged `reportable = False`** — Volmageddon (n=53) and
   Yen_Carry_Unwind (n=42). At a 1% target those windows expect well under one breach, so
   the observed 7–13% rates are noise, not evidence. Either drop those two rows or keep
   them with an explicit "too few observations to interpret" note. Do not let them into a
   headline comparison.

### Deliberately left out (available in `Output/08_VALIDATION/` if wanted)

- `49_mcs.csv` — Model Confidence Set. It eliminates nothing on five of six indices, so it
  reports "we cannot distinguish these models on pinball loss." That is an honest null and
  belongs as **one sentence in the text**, not a table.
- `49_dm_pinball.csv` — DM tests on pinball loss, 60 rows, three significant at 5%
  (DAX RealGARCH vs QR-Range, NKY GARCH-EVT/GJR vs RealGARCH). Same null story as the MCS.
  Appendix at most.
- `42_evt_summary.csv` — ξ ranges and breach counts per index. Its content is already
  printed in FIG2's panel titles.
- `41_exceedance_dependence.csv` — Ljung-Box / runs test / extremal index on exceedances.
  **Worth a mention in Limitations:** it finds genuine tail clustering in SPX (p=0.049) and
  HSI (p=0.019, runs p=0.006), which violates the i.i.d. assumption behind the POT fit.
  One or two sentences, cite the table, no need to print it.
- `47c_es_backtests.csv` — Expected Shortfall backtests. Include only if the paper claims
  anything about ES; it currently does not.
- `48_volregime_pooled_strict.csv` — the same breach analysis conditioned on *volatility
  regime* (Calm/Normal/Stressed/Crisis) instead of named crisis windows. **Worth one
  sentence, because it complicates FIG6:** under this conditioning the degradation nearly
  vanishes (GARCH-EVT runs 1.04/1.43/1.25/1.13% across the four regimes, with the "Crisis"
  vol regime actually *below* "Normal"). The named-window result in FIG6 is the sharper
  and more interpretable one, but a referee could reasonably ask whether it is an artefact
  of window selection, and this table is the answer. Do not print both.
- `34_causal_verification.csv`, `48_window_effect.csv`, `50_audit.csv` — robustness and
  reproducibility evidence. These belong in a supplementary/reproducibility appendix, not
  in the results.
- Per-index EVT diagnostics (`42_evt_diagnostics_*.csv`, ~420 KB each) and all forecast
  files — raw output, never printed.

---

## Two bugs found while verifying this selection (both fixed 2026-08-31)

Checking the manifest's numbers against the CSVs turned up two real defects in
`49_model_comparison.py`, not just documentation slips. Both are fixed and the affected
figures regenerated; `49_basel.csv`, `49_mcs.csv`, `49_dm_pinball.csv` and
`49_strict_window.csv` were byte-identical after the rerun, so no table changed.

1. **`49_loss_metrics.png` carried a hard-coded title.** The caption read "On UKX, RMSE
   says RealGARCH is 0.6% better; QLIKE says 19.3% (DM 6.46, p<0.0001)" as a frozen string
   literal, while the bars were computed from the data. All three numbers were stale —
   almost certainly from before the walk-forward RealGARCH and causal-scaling fixes. The
   correct values are RMSE **−3.6%** (RealGARCH *worse*), QLIKE **+17.9%**, DM **4.33**.
   The caption contradicted its own chart and would have put wrong numbers in the paper.
   The title is now computed from the data, and the exhibit index is chosen from it rather
   than hard-coded.
2. **The headline figure mixed two evaluation windows.** `49_qlike_vs_breach.png` took its
   y-axis (breach rate) from the strict common window but its x-axis (QLIKE) from
   `47a_volatility_losses.csv`, the *unrestricted* file — so 8 of the 18 plotted cells had
   a different sample on each axis, DAX worst at 3172 vs 2755 observations. The same `vol`
   frame fed FIG8. Now loads the strict file, matching the fallback pattern already used
   for the backtests. The effect on FIG4 is visually negligible (QLIKE moves at most 1.4%
   relative, no ranking changes) — the conclusion never depended on it — but the figure is
   now internally consistent and matches the tables beside it.

---

## Provenance

Every file here is a copy of the corresponding file in `Output/09_FIGURES/` or
`Output/08_VALIDATION/`, which in turn match `results/figures/` and `results/tables/` on
GitHub `main` byte for byte. Figures were copied server-side within Drive; tables were
uploaded from the local repo and each verified byte-exact after upload.

**One open item on Drive:** FIG4 and FIG8 were regenerated by the fixes above, and PNGs of
that size cannot be pushed through the Drive tooling in use. The superseded copies are
renamed `OUTDATED_DO_NOT_USE__…` in both `FINAL_OUTPUTS/figures/` and `09_FIGURES/`, and
`ACTION_REQUIRED_replace_two_figures.md` in the Drive folder has the four-file upload
instruction. The repo copies are already correct — this affects Drive only.

Regenerate everything from scratch with the scripts in `Code/10_SCRIPTS/` (41–50).
`50_reproducibility_audit.py` checks 31 conditions over the outputs; it passes 31/31 on the
pinned environment in `requirements_B.txt`.
