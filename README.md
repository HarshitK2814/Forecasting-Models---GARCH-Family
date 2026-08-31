# Forecasting Models — GARCH Family

Volatility and tail-risk forecasting across six Tier-1 equity indices, comparing
**GARCH-EVT**, **Realized GARCH** and **Quantile Regression**.

This repository holds the full pipeline for both researchers: **dataset and documentation,
Researcher A's modelling deliverables** — baseline GARCH/GJR/EGARCH, Realized GARCH, a rolling
out-of-sample forecast engine, and robustness checks — **and Researcher B's** — GARCH-EVT,
quantile regression, evaluation, crisis/regime analysis, and statistical comparison. All 15
plan deliverables are coded, merged and tested as of 2026-08-30.

| | |
|---|---|
| **Indices** | SPX, NDX, UKX, DAX, NKY, HSI (5 regions) |
| **Daily coverage** | 1990-01-02 → 2026-08-21 |
| **Intraday / realized** | 2011-09 onward (DAX from 2013-09) |
| **Primary sample** | B — all six indices, 2013-09-30 → 2026-08-21, 2,685 common days |
| **Dataset status** | Complete and validated — 1,195 + 158 checks, **0 failures** |
| **Model status** | Baseline GARCH/GJR/EGARCH ✅ · Realized GARCH ✅ (walk-forward re-estimated) · Rolling engine ✅ · Robustness checks ✅ · GARCH-EVT ✅ · Quantile Regression ✅ · Evaluation ✅ · Crisis/regime ✅ · Statistical tests ✅. See [Division of labour](#division-of-labour) |
| **Reproducibility** | 45/45 unit tests pass; audit 31 checks, 30/31 locally (1 expected local-Python-version mismatch), 31/31 on the pinned environment |
| **Data cost** | £0 — every source is free and keyless |

---

## Start here

1. **`Datasets/00_DOCUMENTATION/RESEARCHER_A_DECISIONS.md`** — **read this first if you are
   Researcher B.** The forecast-file contract, the non-synchronous-session convention, and
   the crisis-coverage statement — the three things B's code must agree with before writing
   any evaluation or GARCH-EVT code.
2. **`Datasets/00_DOCUMENTATION/RESEARCHER_A_SCOPE.md`** — what A built on top of the
   dataset (baseline GARCH/GJR/EGARCH, Realized GARCH, rolling engine, robustness checks)
   and the results tables for each.
3. **`Datasets/00_DOCUMENTATION/Dataset_Guide.pdf`** — 14 pages. **§3 is 24 precautions.
   Read it before writing any modelling code.**
4. **`Datasets/00_DOCUMENTATION/Handoff_to_Researcher_B.pdf`** — what is done, what is not,
   and where the delivered data departs from the project plan.
5. **`Datasets/00_DOCUMENTATION/Figure_Guide.pdf`** — every diagnostic figure explained.
6. **`Datasets/00_DOCUMENTATION/DATASET_MASTER_REPORT.xlsx`** — 26 sheets: data dictionary,
   precautions, every EDA table, full file inventory.

## Quick start

```python
import pandas as pd

a = pd.read_csv('Datasets/01_ANALYSIS_READY/SPX_analysis.csv', parse_dates=['Date'])

# Daily-only models (GARCH-EVT, Quantile Regression) can use all history back to 1990
r = a.loc[a['Return'].notna(), ['Date', 'Return']]

# Realized measures MUST be gated on RV_Valid — they are NaN on defective sessions
rv = a.loc[a['RV_Valid'] & a['InSample_B'], ['Date', 'RV', 'LogRV', 'RS_neg']]

# YOU apply the forecasting lag. Nothing in the file is pre-lagged.
X = a[['LogRV', 'LogRV_w', 'LogIV', 'NegReturn']].shift(1)
y = a['Return']

# GARCH-EVT stage 1 is already fitted — start stage 2 (GPD on the standardised residuals)
# straight from this file. Do not refit stage 1 independently; use A's std_resid so both
# researchers' stage-1 model is identical.
std = pd.read_csv('Datasets/06_REALIZED_MEASURES/SPX_std_resid.csv', parse_dates=['Date'])

# Baseline-GARCH and Realized-GARCH forecasts, contract-format, ready for evaluation:
import sys; sys.path.insert(0, 'Datasets/10_SCRIPTS')
import importlib
fio = importlib.import_module('26_forecast_io')
fc = fio.read_forecasts('Datasets/20_FORECASTS/GJR-skewt__SPX_forecasts.csv')
ev = fio.eval_frame(fc)   # adds QLIKE, squared error, VaR breach indicators
```

## The four rules that matter

Ignore any of these and the results are invalid, not merely inaccurate.

1. **Model on `01_ANALYSIS_READY/`, never `07_PANEL_INTERMEDIATE/`.** The panel is the raw
   join kept for provenance — it has no session-quality gating and still contains the
   Nikkei's biased 2016–17 realized variances.
2. **Gate every realized measure on `RV_Valid`.**
3. **Apply the forecasting lag yourself.** Every predictor is dated at the close of day *t*;
   nothing is pre-lagged. This is the most common way to invalidate a volatility paper.
4. **Do not winsorize, trim or de-jump the returns.** The tail is the object of study.
   Clipping it shrinks the estimated GPD shape parameter and deletes the exceedances that
   identify it.

The remaining twenty precautions are in §3 of the Dataset Guide.

---

## Repository layout

```
Datasets/
├── START_HERE.md
├── requirements.txt
├── 00_DOCUMENTATION/        reports, PDFs, dictionaries      ← start here
├── 01_ANALYSIS_READY/       ★ the dataset to model on        96 cols × 6 indices
├── 02_RAW_DAILY/            exchange daily OHLC, 1990–2026
├── 04_RAW_VOLATILITY/       17 implied-volatility indices
├── 05_RAW_MACRO/            22 keyless macro / risk series
├── 06_REALIZED_MEASURES/    daily realized measures, base + extended
├── 08_VALIDATION/           every table behind the reports
├── 09_FIGURES/              21 diagnostic figures
├── 10_SCRIPTS/              52 numbered, rerunnable scripts
└── 11_LOGS/                 manifests and run summaries
```

### What is **not** in this repository

Two folders are excluded because they are large and fully regenerable. They live in the
project Google Drive alongside this repo.

| Folder | Size | How to regenerate |
|---|---|---|
| `03_RAW_INTRADAY/` | 1.1 GB | `python 10_SCRIPTS/05_build_intraday_and_RV.py` from the cache |
| `12_CACHE_REGENERATION/` | 810 MB | `python 10_SCRIPTS/03_download_intraday.py SPX NDX UKX DAX NKY HSI` (resumable) |
| `07_PANEL_INTERMEDIATE/` | 17 MB | `python 10_SCRIPTS/10_build_master_panel.py` |

Everything needed to **use** the dataset is here. Only rebuilding the realized measures
from raw ticks requires the Drive copy.

---

## Documentation index

| File | Pages / size | Contents |
|---|---|---|
| `Dataset_Guide.pdf` | 14 pp | Quick start, folder map, **24 precautions**, sample definition, 10 cleaning decisions, quality findings, statistical properties, predictor screening, EVT thresholds, limitations, reproduction, data dictionary |
| `Handoff_to_Researcher_B.pdf` | 6 pp | Task status against the plan, milestones, 12 deviations from the Executive Summary, first eight steps, settled specification, open questions |
| `Figure_Guide.pdf` | 12 pp | Each figure: what it plots, how to read it, what our data shows, what it decides, caveat |
| `Diagnostic_Figures.pdf` | 11 pp | The ten figures at full size with captions |
| `DATASET_MASTER_REPORT.xlsx` | 26 sheets | Every table in one workbook |
| `EDA_REPORT.md` / `.xlsx` | 29 k / 18 sheets | The full cleaning and EDA record |
| `DATA_ACQUISITION_REPORT.xlsx` | 28 sheets | Sources, API links, fields, rejected sources with evidence |
| `README.md`, `TASK_PROGRESS.md` | — | Provenance narrative and the phase-by-phase tracker |

---

## What the EDA already settled

These are conclusions from tests in the EDA report, each reproducible from the shipped
validation tables. They are not open questions.

| Conclusion | Evidence |
|---|---|
| Plain GARCH(1,1) is misspecified — use **GJR or EGARCH** | Engle–Ng sign-bias rejects on all six indices (worst p = 7e-5) |
| A Gaussian innovation is insufficient | Hill tail index 2.6–3.9; every estimate below 4, so the fourth moment does not exist |
| The EVT stage is justified | GPD shape ξ is positive on all six indices. Measured on genuine GARCH residuals at q=0.95 the range is **0.050–0.159** (SPX 0.155, NDX 0.057, UKX 0.061, DAX 0.154, NKY 0.050, HSI 0.159), not the 0.15–0.25 previously quoted here — that band came from rolling-standardised returns, before any GARCH model existed. The conclusion is unchanged; the magnitude is smaller. |
| Long memory is present — use the HAR cascade | GPH *d* on log RV is 0.50–0.63; ADF and KPSS both reject |
| An AR(1) mean term is warranted | Ljung–Box on returns rejects for 5 of 6 indices |
| Use the level-plus-share realized block | `LogRV` + `LogRS_neg` gives VIF ≈ 95; `LogRV` + `RSV_Ratio` + `JumpShare` + `RSkew` gives max VIF 8.1 |
| Differenced macro only | `US10Y_pct` and `TermSpread_pct` fail ADF in levels (p = 0.87, 0.33) |
| Implied vol leads for the tail | For the 1% quantile the vol index beats every realized measure; for the *level* of volatility the realized measures win |
| Recommended POT threshold | 95th–97.5th percentile, 230–460 exceedances — re-estimate on real GARCH residuals |

---

## Three things that constrain the paper

1. **The realized-measure models cannot see the 2008 crisis.** Free intraday history begins
   in 2011 (2013 for the DAX). Sample B contains 6 of 10 named crisis windows but misses the
   GFC, the dot-com bust, the Asian crisis and the Euro sovereign crisis. GARCH-EVT and
   quantile regression *do* see them, because they estimate on daily data from 1990. **The
   three models therefore see different crisis histories and the paper must say so.**
2. **The intraday data is a bid-side index CFD, not the exchange index.** Aligned correlation
   against the exchange daily return is 0.97–0.99, so it tracks well, but it is a proxy and
   must be described as one.
3. **Sessions are not synchronous.** Tokyo and Hong Kong close before New York opens, so a US
   shock on day *t* reaches Asia on day *t+1*. Same-date Asia–US return correlation is only
   ~0.18 — a timing artefact, not economic independence. Settle the convention before
   producing any cross-index result.

---

## Division of labour

Per `Executive Summary.pdf`, the project splits between two researchers.

| Module | Lead | Budget | Status |
|---|---|---|---|
| Data acquisition & cleaning | A | 40 h | ✅ **Complete** |
| Realized volatility construction | A | 16 h | ✅ **Complete** (exceeds scope) |
| Baseline GARCH & GJR/EGARCH | A | 24 h | ✅ **Complete** |
| Realized GARCH | A | 24 h | ✅ **Complete** — walk-forward annual expanding-window re-estimation (2026-08-29), not a single full-sample fit |
| Rolling out-of-sample engine | A | 24 h | ✅ **Complete**, GARCH-family and Realized GARCH both walk-forward |
| Robustness checks | A | 16 h | ✅ **Complete** (6 checks — sub-sample, distribution, RV frequency, refit cadence, window length, horizon extension) |
| GARCH-EVT | **B** | 24 h | ✅ **Complete** — genuine GJR-GARCH residuals, POT threshold re-estimated (`41_evt_threshold.py`) |
| Quantile regression | **B** | 16 h | ✅ **Complete** |
| Evaluation metrics | **B** | 24 h | ✅ **Complete** — QLIKE/VaR/ES backtests, plus a strict common-evaluation-window variant (`47_evaluation.py`) |
| Crisis / regime analysis | **B** | 16 h | ✅ **Complete** (`48_crisis_regime.py`) |
| Statistical tests (DM, MCS) | **B** | 12 h | ✅ **Complete** — HAC/Newey-West-corrected Diebold-Mariano, Model Confidence Set, Basel traffic light (`49_model_comparison.py`) |

**All 15 plan deliverables are complete and merged on `main`.** See
`Datasets/00_DOCUMENTATION/RESEARCHER_A_SCOPE.md` for what A built,
`RESEARCHER_A_DECISIONS.md` for the forecast-file contract B's code depends on, and
`EXECUTIVE_SUMMARY_ADDENDUM.md` for every place delivery differs from the Executive Summary's
own text — including two P0 look-ahead fixes closed 2026-08-29 (causal Hansen-Lunde scaling;
walk-forward Realized GARCH, which closed the "one open item" this section used to describe)
and a strict common-evaluation-window fix closed 2026-08-30 after code review.

---

## Reproducing

```bash
pip install -r Datasets/requirements.txt
pip install arch          # named in the plan for GARCH and VaR tests; not yet a dependency

cd Datasets

# acquisition (needs network; step 3 is resumable)
python 10_SCRIPTS/01_download_daily.py
python 10_SCRIPTS/02_download_volatility.py
python 10_SCRIPTS/03_download_intraday.py SPX NDX UKX DAX NKY HSI
python 10_SCRIPTS/09_download_macro_yahoo.py
python 10_SCRIPTS/05_build_intraday_and_RV.py
python 10_SCRIPTS/10_build_master_panel.py

# cleaning, EDA and documentation
python 10_SCRIPTS/11_define_samples.py
python 10_SCRIPTS/12_extended_realized_measures.py
python 10_SCRIPTS/13_eda_quality_audit.py
python 10_SCRIPTS/14_session_classification.py
python 10_SCRIPTS/15_build_analysis_dataset.py
python 10_SCRIPTS/16_eda_stylized_facts.py
python 10_SCRIPTS/17_eda_predictor_screening.py
python 10_SCRIPTS/18_eda_tails_breaks_features.py
python 10_SCRIPTS/19_eda_figures.py
python 10_SCRIPTS/20_finalise_and_document.py
python 10_SCRIPTS/24_add_crisis_regimes.py
python 10_SCRIPTS/21_build_eda_report.py
python 10_SCRIPTS/22_validate_analysis.py     # 158 checks, must report 0 failures
python 10_SCRIPTS/23_build_documentation.py
python 10_SCRIPTS/25_build_figure_and_handoff_docs.py

# modelling (Researcher A's deliverables) - needs: pip install arch statsmodels scipy
# matplotlib tabulate (all pinned in requirements.txt)
python 10_SCRIPTS/27_baseline_garch.py             # ~10s. Writes std_resid files B needs.
python 10_SCRIPTS/28_realized_garch.py             # ~9 min (custom optimiser, 6 indices)
python 10_SCRIPTS/29_rolling_forecast_engine.py    # ~3 min. Walk-forward GJR-skewt forecasts.
python 10_SCRIPTS/30_robustness_checks.py          # ~1 min
python 10_SCRIPTS/31_build_synthetic_forecasts.py  # instant; superseded once real files exist
python 10_SCRIPTS/32_modelling_figures.py          # ~1 min. 7 figures backing the tables above
python 10_SCRIPTS/33_window_and_horizon_robustness.py  # ~5 min. Closes the Exec Summary's
                                                        # window-length and horizon items
```

Steps 12 and 14 read the 24,000-file Dukascopy cache and take several minutes each.
Step 28 is the only other slow step (a hand-written quasi-MLE optimiser, not `arch`).
Everything else runs in seconds.

### Two operational traps that cost real time

- **An expired CA bundle silently breaks every HTTPS fetch.** If downloads fail wholesale,
  update `certifi` before debugging anything else.
- **Dukascopy throttles per IP, and its URL months are ZERO-INDEXED** (January = `00`). Use
  one process with HTTP keep-alive; extra parallel processes made throughput *worse*
  (4.9 → 1.9 req/s) and triggered a lasting penalty.

---

## Data sources

| Layer | Source | Access |
|---|---|---|
| Daily index OHLC | Yahoo Finance (`yfinance`) | Free, keyless |
| Intraday 1-min | Dukascopy historical tick/candle feed | Free, keyless |
| Volatility indices | CBOE, STOXX, Yahoo | Free, keyless |
| Macro / risk factors | Yahoo Finance | Free, keyless |

Sources evaluated and **rejected**, with evidence, are documented in
`DATA_ACQUISITION_REPORT.xlsx` (Stooq — JS anti-bot; Alpha Vantage — intraday now paid;
FRED S&P 500 — 10-year limit and no redistribution; Oxford-Man realized library —
discontinued).

> **Note on redistribution.** The CSVs here are derived from publicly accessible endpoints
> and are committed to support reproducibility of academic research. They remain subject to
> the originating providers' terms of use. Anyone reusing them should consult those terms
> rather than treating this repository as the licence.

## Licence

Code in `10_SCRIPTS/` is available for academic use. Data files are subject to the
originating providers' terms as noted above.
