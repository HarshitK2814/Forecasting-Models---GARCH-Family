# Datasets — START HERE

**Volatility and tail-risk forecasting across six Tier-1 equity indices**
GARCH-EVT · Realized GARCH · Quantile Regression — Researcher A (data)

---

## Read these three things, in this order

| # | File | Why |
|---|---|---|
| 1 | `00_DOCUMENTATION/Dataset_Guide.pdf` | The complete guide. **Section 3 is 24 precautions — read it before writing any modelling code.** |
| 2 | `00_DOCUMENTATION/DATASET_MASTER_REPORT.xlsx` | Every table in one workbook: dictionary, precautions, all EDA results, file inventory. |
| 3 | `00_DOCUMENTATION/Diagnostic_Figures.pdf` | The ten diagnostic figures with captions. |

## The one thing to get right

**Model on `01_ANALYSIS_READY/<CODE>_analysis.csv`. Nothing else.**

`07_PANEL_INTERMEDIATE/` is the raw join kept only for provenance — it has no session-quality
gating and still contains the Nikkei's biased 2016–17 realized variances.

```python
import pandas as pd
a = pd.read_csv('01_ANALYSIS_READY/SPX_analysis.csv', parse_dates=['Date'])

# realized measures MUST be gated on RV_Valid
rv = a.loc[a['RV_Valid'] & a['InSample_B'], ['Date', 'RV', 'LogRV', 'RS_neg']]

# YOU apply the forecasting lag — nothing in the file is pre-lagged
X = a[['LogRV', 'LogRV_w', 'LogIV', 'NegReturn']].shift(1)
y = a['Return']
```

## Folder map

| Folder | Contents |
|---|---|
| `00_DOCUMENTATION/` | Reports, PDFs, dictionaries. Start here. |
| `01_ANALYSIS_READY/` | **The dataset to model on.** One cleaned CSV per index, 96 columns. |
| `02_RAW_DAILY/` | Exchange daily OHLC per index, 1990–2026. Yahoo Finance. |
| `03_RAW_INTRADAY/` | 1-min and 5-min session bars per index-year. Dukascopy. |
| `04_RAW_VOLATILITY/` | 17 implied-volatility indices. CBOE, STOXX, Yahoo. |
| `05_RAW_MACRO/` | 22 keyless macro / risk-factor series. Yahoo Finance. |
| `06_REALIZED_MEASURES/` | Daily realized measures per index, base and extended. |
| `07_PANEL_INTERMEDIATE/` | The raw join, before cleaning. Provenance only. |
| `08_VALIDATION/` | Every validation and EDA table behind the reports. |
| `09_FIGURES/` | Ten diagnostic figures. |
| `10_SCRIPTS/` | Every script, numbered in execution order. |
| `11_LOGS/` | Manifests and per-phase run summaries. |
| `12_CACHE_REGENERATION/` | Raw Dukascopy `.npy` cache. Only needed to rebuild realized measures. |

## Status

| | |
|---|---|
| Indices | SPX, NDX, UKX, DAX, NKY, HSI |
| Daily coverage | 1990-01-02 → 2026-08-21 |
| Realized coverage | 2011-09 onward (DAX from 2013-09); **NKY has no valid RV in 2016–17** |
| Primary sample | B — all six indices, 2013-09-30 → 2026-08-21, 2,685 common days |
| Validation | 1,195 acquisition checks + 158 analysis checks, **0 failures** |
| Models fitted | **None.** This is a dataset only — no model has been estimated. |

## The four things that will invalidate results if ignored

1. Model on `01_ANALYSIS_READY/`, never `07_PANEL_INTERMEDIATE/`.
2. Gate every realized measure on `RV_Valid`.
3. Apply the forecasting lag yourself — the file is contemporaneous, nothing is pre-lagged.
4. Do not winsorize, trim or de-jump the returns. The tail is the object of study.

The remaining twenty precautions are in §3 of the Dataset Guide.

## Reproducing

Run from inside this folder:

```
pip install -r requirements.txt
python 10_SCRIPTS/11_define_samples.py
... see §13 of the Dataset Guide for the full ordered list
```

Steps 12 and 14 read the 24,000-file Dukascopy cache and take several minutes each.
Everything else runs in seconds.
