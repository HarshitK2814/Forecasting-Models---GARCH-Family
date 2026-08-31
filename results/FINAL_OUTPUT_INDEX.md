# Final Output — what is in this folder

The paper's exhibits, and nothing else. Eight figures in `figures/`, seven tables in
`tables/`, both numbered in the order they appear in the manuscript.

Everything here is computed on the **strict common evaluation window** — the per-index
intersection of the dates on which GARCH-EVT, GJR-skew-t and Realized GARCH all produced a
valid forecast. That is what makes the numbers comparable across models, and it is why
some counts here differ from earlier drafts.

> **Cite the CSVs, not this file.** This is an index. Any prose summary of a result will
> eventually fall behind the table it describes.
>
> Why each exhibit was chosen, what was cut and why, and the caveats worth a sentence in
> the text: `Datasets/00_DOCUMENTATION/README_FINAL_OUTPUTS.md`.

---

## figures/

| File | What it shows | Paper section | Written by |
|---|---|---|---|
| `FIG1_41_threshold_stability.png` | GPD shape ξ against POT threshold, with sampling-error bands. Justifies the q=0.95 choice. | Methods | `41_evt_threshold.py` |
| `FIG2_41_qq_gpd.png` | GPD QQ plots of exceedances, all six indices. Shows the tail fit is valid. | Methods | `41_evt_threshold.py` |
| `FIG3_51_realgarch_innovation.png` | RealGARCH-t against RealGARCH-skew-t at 99% VaR, same dates. Same variance model and realized measure, different innovation — isolates where the tail failure comes from. | Results | `51_final_release_exhibits.py` |
| `FIG4_49_qlike_vs_breach_HEADLINE.png` | **The headline.** Volatility accuracy on one axis, tail accuracy on the other; the models rank in opposite orders. | Results | `49_model_comparison.py` |
| `FIG5_49_basel.png` | Basel traffic-light zones, six indices by five models. | Results | `49_model_comparison.py` |
| `FIG6_48_crisis_heatmap.png` | Breach rate by named crisis regime, pooled across indices. | Results | `48_crisis_regime.py` |
| `FIG7_51_qr_calibration_strict.png` | Quantile-regression calibration at all three VaR levels, on the strict window. | Results | `51_final_release_exhibits.py` |
| `FIG8_49_loss_metrics.png` | The same model pair under four different volatility loss functions, which do not agree. | Discussion | `49_model_comparison.py` |

## tables/

| File | What it is | Paper section | Written by |
|---|---|---|---|
| `TAB1_47_strict_window.csv` | How many rows each index and model contributes, before and after the common-window restriction. Defines the evaluation sample. | Data | `47_evaluation.py` |
| `TAB2_47a_volatility_losses_strict.csv` | QLIKE, MSE, RMSE, MAE and MAPE per index and model. | Results | `47_evaluation.py` |
| `TAB3_47a_dm_volatility.csv` | Diebold–Mariano tests on QLIKE differences, HAC-corrected. Says which accuracy gaps are real. | Results | `47_evaluation.py` |
| `TAB4_47b_var_backtests_strict.csv` | Kupiec, Christoffersen independence and conditional coverage, and Dynamic Quantile, at 99 / 97.5 / 95%. | Results | `47_evaluation.py` |
| `TAB5_49_basel.csv` | Basel zone per index and model, with breach and expected counts. Numeric backing for FIG5. | Results | `49_model_comparison.py` |
| `TAB6_48_crisis_pooled_strict.csv` | Breach counts and rates by crisis regime. Numeric backing for FIG6. | Results | `48_crisis_regime.py` |
| `TAB7_51_realgarch_innovation.csv` | RealGARCH-t against RealGARCH-skew-t: breach counts, rates, Kupiec p and QLIKE. Numeric backing for FIG3. | Results | `51_final_release_exhibits.py` |

---

## Two things to handle before these go into the manuscript

1. **`TAB4` is too long for a main-text table.** Filter it to `confidence == 0.99` — that
   is one row per index and model, and the 99% level is where the models actually separate.
   Put the 97.5% and 95% rows in an appendix.
2. **`TAB6` has two regimes flagged `reportable = False`.** Those windows are far too short
   to estimate a 1% breach rate from, so their rates are noise rather than evidence. Drop
   those rows or label them explicitly; do not let them into a headline comparison.

## Notes

- The `FIG_` and `TAB_` prefixes exist only here, to fix the reading order. The repository
  uses the unprefixed filenames its scripts write. If you pull one of these files down to
  edit, **copy it rather than renaming in place**, or the prefix ends up in the codebase and
  the next pipeline run produces duplicates.
- Every file here is byte-identical to its counterpart in `results/figures/` or
  `results/tables/` on GitHub `main`.
- To regenerate the whole set: scripts 41 to 51 in `Datasets/10_SCRIPTS/`, in order.
  `51_final_release_exhibits.py` must run after 47 and 49 because it reads their output.
