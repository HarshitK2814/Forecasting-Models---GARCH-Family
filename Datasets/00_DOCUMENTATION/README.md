# Dataset — GARCH-EVT vs Realized GARCH vs Quantile Regression

Assembled by **Researcher A**. Everything here is from a **free, no-account, no-payment**
source. Last full rebuild: see `11_LOGS/` timestamps.

If you only read one thing, read **§3 Known limitations**. Everything else is provenance.

---

## 1. What is here

```
data/
├── daily/<CODE>/            exchange index daily OHLC + log return, 1990 → today
├── intraday/1min/<CODE>/    session-filtered 1-minute bars, one CSV per year
├── intraday/5min/<CODE>/    same, aggregated to 5 minutes  ← the modelling frequency
├── volatility/<SYM>/        17 implied-volatility / skew indices
├── macro/<CODE>/            22 keyless macro & risk-factor series
├── realized_volatility/     <CODE>_RV_daily.csv — RV, BPV, jumps, overnight return
├── panel/                   <CODE>_panel_daily.csv ← THE ANALYSIS FILE. Start here.
├── _validation/             machine-generated evidence that the data is sound
├── _logs/                   one manifest CSV per download phase
├── _scripts/                every download and build step, numbered, re-runnable
├── _cache_duka/             raw Dukascopy .npy day-caches (regeneration cache, not data)
├── requirements.txt
├── TASK_PROGRESS.md
└── DATA_ACQUISITION_REPORT.xlsx
```

Every file is **CSV**, UTF-8, comma-separated, `YYYY-MM-DD` dates, no thousands separators.

### Final coverage

| Code | Daily rows | Intraday days | RV session-days | Median 5-min bars | RV starts |
|---|---|---|---|---|---|
| SPX | 9,227 | 4,080 / 4,080 | 3,755 | 78 | 2011-09-19 |
| NDX | 9,227 | 4,074 / 4,080 | 3,733 | 78 | 2011-09-19 |
| UKX | 9,255 | 4,080 / 4,080 | 3,728 | 102 | 2011-09-19 |
| DAX | 9,270 | 3,551 / 3,559 | 3,307 | 102 | 2013-09-30 |
| NKY | 8,990 | 4,080 / 4,080 | 3,543 | 60 | 2011-09-20 |
| HSI | 9,041 | 4,080 / 4,080 | 3,576 | 66 | 2011-09-19 |

The 14 missing intraday days survived three independent retry passes — they are absent at
source, not transient failures. The median 5-min bar count matches each exchange calendar
exactly (390 min ÷ 5 = 78; 510 ÷ 5 = 102; Tokyo 300 ÷ 5 = 60; Hong Kong 330 ÷ 5 = 66), which
independently confirms the session windows and DST handling are right.

DAX intraday starts 2013-09-30 rather than 2011 — that is where Dukascopy's DEUIDXEUR history
actually begins, verified by probing earlier years and getting empty responses.

### The six indices

| Code | Index | Exchange | Cash session (local) | Region |
|---|---|---|---|---|
| SPX | S&P 500 | NYSE/Nasdaq | 09:30–16:00 America/New_York | US large cap |
| NDX | Nasdaq-100 | Nasdaq | 09:30–16:00 America/New_York | US tech |
| UKX | FTSE 100 | LSE | 08:00–16:30 Europe/London | UK |
| DAX | DAX 40 | Xetra | 09:00–17:30 Europe/Berlin | Euro area |
| NKY | Nikkei 225 | TSE | 09:00–11:30, 12:30–15:00 Asia/Tokyo | Japan |
| HSI | Hang Seng | HKEX | 09:30–12:00, 13:00–16:00 Asia/Hong_Kong | Greater China |

Sessions are applied in **exchange-local time with DST handled by the tz database**, not by a
fixed UTC offset. Getting this wrong silently shifts a whole winter of bars.

---

## 2. Where each thing comes from

| Layer | Source | Access | Notes |
|---|---|---|---|
| Daily index | Yahoo Finance via `yfinance` | keyless HTTP | `^GSPC ^NDX ^FTSE ^GDAXI ^N225 ^HSI` |
| Intraday | **Dukascopy** historical feed | keyless HTTP | index CFDs, 1-minute, BID side |
| Volatility (US) | CBOE `cdn.cboe.com` | keyless CSV | 13 series incl. VIX, VXN, VVIX, SKEW |
| Volatility (EU) | STOXX `stoxx.com` `.txt` | keyless | VSTOXX (V2TX), VDAX-NEW (V1X), V6I1 |
| Volatility (JP) | Yahoo `^NKVI.OS` | keyless | Nikkei VI, 2018+ only |
| Macro / factors | Yahoo Finance via `yfinance` | keyless | rates, FX, commodities, credit ETFs |
| Macro (canonical) | FRED | **needs a free key** | script 04, not pre-pulled — see §3.5 |

### Sources evaluated and **rejected**, with the reason

These were each tested, not assumed. Do not re-add them without re-testing.

| Source | Verdict | Evidence |
|---|---|---|
| Stooq | **BROKEN** | Returns a JavaScript anti-bot HTML challenge, not CSV |
| Alpha Vantage intraday | **NOW PAID** | The `month=` extended-history parameter moved behind a paid tier |
| FRED `SP500` series | **UNUSABLE** | Capped at 10 years, and redistribution is prohibited |
| Oxford-Man Realized Library | **DISCONTINUED** | No longer published |
| FX / crypto as an intraday proxy | **REJECTED ON METHOD** | See §3.1 |

---

## 3. Known limitations — read this section

### 3.1 Intraday is a CFD, not the exchange index. This is measured, not assumed.

No free source publishes intraday **exchange index** data with a decade of history. Dukascopy
publishes index **CFDs**, which track the index but are broker-quoted.

The project's original plan was to substitute FX/crypto intraday as a proxy. That was rejected:
Realized GARCH would then be fed a realized measure computed on a *different asset* from the
one whose returns it models, which is not a limitation but an error.

Instead we use the index CFD and **measure the discrepancy explicitly** —
`10_SCRIPTS/08_cfd_vs_index_check.py`, results in `08_VALIDATION/cfd_vs_index_check.csv`.
It regresses CFD session close-to-close returns on exchange index daily returns.

**The alignment rule matters enormously.** The CFD return on day *t* is against the previous
day *present in the CFD file*. Where the feed is missing a day, that becomes a two-day return
compared against a one-day index return. Those observations are not comparable. Excluding them
changes the verdict completely:

Final results, all six indices (BID side, aligned sample):

| Index | n | % aligned | corr | **aligned R²** | naive R² | slope | tracking err | verdict |
|---|---|---|---|---|---|---|---|---|
| SPX | 3,566 | 97.4% | 0.9964 | **0.9928** | 0.9881 | 0.994 | 1.43%/yr | **PASS** |
| NDX | 3,549 | 97.3% | 0.9985 | **0.9971** | 0.9720 | 0.992 | 1.12%/yr | **PASS** |
| DAX | 3,229 | 98.9% | 0.9912 | **0.9825** | 0.9764 | 0.992 | 2.48%/yr | usable — disclose |
| HSI | 3,538 | 99.3% | 0.9936 | **0.9873** | 0.9433 | 1.011 | 2.38%/yr | usable — disclose |
| NKY | 3,153 | 94.2% | 0.9928 | **0.9856** | 0.9270 | 0.994 | 2.59%/yr | usable — disclose |
| UKX | 3,619 | 98.4% | 0.9849 | **0.9700** | 0.9124 | 0.989 | 2.58%/yr | usable — disclose |

**No index fails.** On the naive number NDX (0.972), NKY (0.927), HSI (0.943) and UKX (0.912)
would all have looked far worse and UKX would have been discarded outright. The headline figure
in the validation CSV is the aligned one; the naive one is kept beside it as `R2_Naive_DoNotUse`.

UKX is the weakest at 0.970 — a genuine ~15 bp/day structural tracking difference, stable in
every year from 2012 to 2026, i.e. a real property of the FTSE CFD rather than a data fault.
**Disclose it in the paper.** Do not present UKX realized measures as exchange-quality.

### 3.2 The intraday series is BID-side only

Realized measures are computed on bid quotes. Using one side consistently avoids bid-ask bounce
contaminating RV, which is the usual concern. It does mean the level carries the half-spread,
and no spread-based liquidity measure can be derived. Mid-price RV would require downloading the
ASK side as well.

### 3.3 Frozen sessions are dropped, and every one is logged

On days a cash market is shut, Dukascopy still streams a nominally-live CFD that drifts by a
fraction of a point. Those bars survive a naive stale-bar filter but represent no trading, and
produce RV ≈ 0. Any session whose whole-day price range is under **4 basis points** is dropped.

Calibration, over 10,847 session-days: the largest range among dropped days is 2.90 bp; the
smallest among kept days is 5.68 bp. The two clusters are cleanly separated.

It removes **15 sessions in total**, every one listed in
`08_VALIDATION/frozen_sessions_dropped.csv`:

- UKX 2013-12-25, 2013-12-26, 2014-01-01 and DAX 2013-12-24/25/26, 2013-12-31, 2014-01-01 —
  LSE and Xetra closed (also absent from the exchange daily files)
- **SPX 2013-02-25 → 02-28 and NDX 2013-02-26 → 02-28 — a Dukascopy feed outage on days the
  exchange was open.** These would otherwise have entered the sample as zero-volatility days,
  and they were SPX's two largest CFD-vs-index residuals.

### 3.4 Yahoo's OHLC is not internally consistent for FX and futures

On a minority of days Yahoo writes a placeholder bar with `Open == High == Low` and a Close from
a different snapshot, so Close can sit outside `[Low, High]`: GOLD 6.8% of days, USDJPY 3.6%,
EURUSD 2.2%, others under 2%. Equities, ETFs and yield indices are clean.

We established which field is wrong rather than guessing. On GOLD's 441 violating days, Close is
closer to the next day's Open (32.9 bp median) than the High/Low midpoint is (61.5 bp). **Close
is reliable; Open/High/Low are the corrupted fields.**

No row is discarded. Every macro file carries an `OHLC_Consistent` boolean column.
**For the `05_RAW_MACRO/` folder use `Close` and `LogReturn` only**, unless you filter on that flag.

### 3.5 FRED is not pre-downloaded

`fred.stlouisfed.org` is unreachable from the acquisition network — 4/4 ReadTimeout at 25 s,
re-verified 2026-08-23. `api.stlouisfed.org` *is* reachable, answering in 0.5–0.8 s with
HTTP 400 "missing api_key", so the API path works and only needs a key.

Rather than ship a dataset that requires a key, script 09 rebuilds the same factor space from
keyless market instruments: `^TNX/^FVX/^TYX/^IRX` for the curve, `DX-Y.NYB` and spot FX for the
dollar, `HYG` vs `IEF` for credit stress. Only the genuinely macroeconomic series — CPI,
unemployment, industrial production, NFCI, the NBER recession flag — have no honest market
analogue. They are weekly/monthly optional covariates. Script 04 will pull them in one command
once a free key is set in `FRED_API_KEY`.

### 3.6 Not every index has its own free volatility index

Free regional vol indices exist for the US (VIX, VXN) and Europe (VSTOXX, VDAX-NEW), and for
Japan from 2018 (NKVI). There is **no free FTSE-100 or Hang Seng volatility index**. The panel
uses VXEFA (developed ex-US) and VXEEM (emerging markets) as declared proxies, with the fallback
recorded in a `VolIdx_Symbol` column so the substitution is never invisible.

---

## 4. The panel files — start here

`07_PANEL_INTERMEDIATE/<CODE>_panel_daily.csv` is the join of everything onto one row per index trading day.

**The exchange daily file is the spine.** Everything else is left-joined onto it. The dependent
variable is the index return, so the sample must be exactly the days the index traded. Realized
measures exist only where the CFD feed had a clean session, so `RV_5min` is NaN on some spine
days. That is honest missingness — it is not filled. `HasRV_t` flags it.

### No look-ahead

Every predictor is dated at the **close of day _t_** and is meant to forecast **_t+1_**. The
panel does **not** lag anything; the modelling code must apply the lag. The single subtlety is
`Overnight_LogRet`, which spans close(*t*−1) → open(*t*): it is known at the *open* of day *t*
but still dated *t*. Treat it as the first observable piece of day *t*, not as day-*t* close
information.

### Column groups

| Group | Columns |
|---|---|
| Spine | `Date Symbol Open High Low Close Return` |
| Realized | `RV_1min RV_5min RV_10min RV_15min RV_30min BPV_5min NBars_5min RVol_5min LogRV_5min` |
| Jumps | `Jump_5min = max(RV−BPV, 0)`, `ContVar_5min = RV − Jump` (Barndorff-Nielsen–Shephard) |
| CFD cross-ref | `CFD_SessionReturn`, `Overnight_LogRet`, `HasRV_t` |
| Implied vol | `VolIdx`, `VolIdx_Symbol`, `VolIdx_Fallback`, `VolIdx_Fallback_Symbol` |
| Macro | `US10Y_pct US13W_pct TermSpread_pct DXY WTI_usd GOLD_usd HYG_px IEF_px CreditStress` |
| Derived | `AbsReturn NegReturn ParkinsonVar RangePct` |

`US10Y_pct` and `US13W_pct` are in **percent** (4.74 means 4.74%). Yahoo *used* to quote the
CBOE yield indices as percent × 10 and much older code still divides by 10 — it no longer should.
Verified against known history: raw `^TNX` averages 8.55 in 1990, 0.88 in 2020, 4.74 today.

`CreditStress` is `−(Δlog HYG − Δlog IEF)`: high-yield underperformance against duration-matched
Treasuries. It rises with credit stress, i.e. it has the same sign as a widening HY OAS.

### Multiple sampling frequencies are provided deliberately

`RV_1min` through `RV_30min` exist so the volatility signature plot can be drawn and the
microstructure-noise/variance trade-off justified rather than asserted. **5-minute is the
modelling frequency** — the standard choice, and the observed signature declines monotonically
from 1-min to 30-min as theory predicts.

---

## 5. How the data was verified

Nothing here is trusted on a single fetch.

1. **Two-identical-fetch rule.** Every Yahoo and CBOE series is downloaded repeatedly and
   accepted only when two *consecutive* fetches agree on row count *and* last close. The number
   of attempts is recorded per series in the phase manifests.
2. **Binary format proof.** The Dukascopy `.bi5` layout (LZMA, 24-byte records, `>Iiiiif`,
   OHLC as int32 ÷ 1000, **zero-indexed months** in the URL) was reverse-engineered and then
   confirmed against 2020-03-16, which reproduces the correct S&P crash-day price path.
3. **Five-axis validation sweep** (`06_validate.py`) over every CSV: schema, key uniqueness,
   monotonic time, value ranges/OHLC ordering, and continuity. Results in
   `08_VALIDATION/validation_report.csv`. Every warning it raises has been investigated
   individually and either fixed or explained in §3 — none are left unexplained.
4. **Variance decomposition cross-check.** SPX session RV 12.51% + overnight variance 10.82%
   combines to 16.54% annualised against 16.98% from actual daily returns — a 2.6% gap, which is
   what you expect if the pieces are being measured correctly.
5. **Bar-count check.** Median 5-min bars per session: SPX/NDX 78 (390 min ÷ 5), UKX 102
   (510 min ÷ 5). Exactly right, which confirms the session windows and DST handling.
6. **CFD-vs-index regression**, §3.1 — all six indices, 0.970 ≤ R² ≤ 0.997.

The full sweep is **1,195 checks across 239 files with zero failures**. The 19 warnings it
raises break down as: 7 Yahoo FX/futures OHLC files (§3.4), 6 continuity gaps that are real
holes in the source, 4 intraday files whose only flagged bar is a genuine overnight gap
(SPX/NDX 2020-03-16 COVID limit-down, DAX 2016-06-24 Brexit −10.3% open, UKX 2020-03-16), OVX
touching 325 the day after WTI printed negative, and VVIX's 2006 backfill gaps.

---

## 6. Reproducing everything

```bash
pip install -r requirements.txt          # certifi is pinned deliberately, see below

python 10_SCRIPTS/01_download_daily.py
python 10_SCRIPTS/02_download_volatility.py
python 10_SCRIPTS/03_download_intraday.py SPX NDX UKX DAX NKY HSI
python 10_SCRIPTS/09_download_macro_yahoo.py
python 10_SCRIPTS/05_build_intraday_and_RV.py
python 10_SCRIPTS/10_build_master_panel.py
python 10_SCRIPTS/06_validate.py
python 10_SCRIPTS/08_cfd_vs_index_check.py
python 10_SCRIPTS/07_build_excel_report.py

# optional, needs a free key from https://fredaccount.stlouisfed.org/apikeys
FRED_API_KEY=... python 10_SCRIPTS/04_download_macro_FRED.py
```

Step 3 is resumable and safe to re-run — cached days are skipped, so a repeat run only fetches
what is missing or previously failed.

### Two operational traps that cost real time

**certifi.** The workstation shipped certifi 2024.08.30, whose CA bundle had expired roots.
*Every* HTTPS fetch failed with `CERTIFICATE_VERIFY_FAILED` and the cause was not obvious.
`requirements.txt` pins `certifi>=2026.7.22`. Do not let it drift back.

**Dukascopy rate limiting is per-IP, and parallelism makes it worse.** One process with 6 threads,
each holding its own keep-alive `requests.Session`, sustains ~4.8 req/s. Without keep-alive it is
~0.05 req/s — a fresh TLS handshake per request, a 100× penalty. Launching *additional* processes
dropped combined throughput from 4.9 to 1.9 req/s and triggered a lasting throttle. **One process,
six threads, keep-alive.**

---

## 7. Methodological notes for the modelling stage

- **Estimation window.** The Executive Summary proposes 2,000 days. With intraday starting
  2011-09, a 2,000-day rolling window pushes the first out-of-sample forecast to roughly 2020,
  leaving a test period dominated by COVID and little else. **1,000–1,250 days** is the
  conventional choice and preserves a usable test window.
- **Sample alignment.** Daily and intraday do *not* need a common start date. Only the
  **out-of-sample test window** must be identical across models. GARCH-EVT and Quantile
  Regression can learn from 1990; Realized GARCH cannot start before 2011-09. Truncating the
  first two to match the third throws away twenty years for no methodological gain.
- **Model comparison.** With six indices and several models, pairwise Diebold–Mariano tests
  invite a multiple-testing objection. Prefer **Hansen's Model Confidence Set**, with DM
  reported as a secondary pairwise check.
- **EVT thresholds.** Standard POT practice puts the threshold at the 90th–95th percentile of
  standardised residuals; report sensitivity across that range rather than one point.

---

## 8. Cleaning and EDA — **the analysis dataset lives in `01_ANALYSIS_READY/`, not `07_PANEL_INTERMEDIATE/`**

`07_PANEL_INTERMEDIATE/` is the raw join and is kept only for provenance. **Model on `01_ANALYSIS_READY/<CODE>_analysis.csv`.**
The full record of what was cleaned and why is in **`EDA_REPORT.md`** (and `EDA_REPORT.xlsx`,
18 sheets); the diagnostic figures are in `09_FIGURES/`.

### The three findings that changed the design

1. **The Nikkei intraday feed is broken for 2016 and 2017.** The entire 09:00 opening hour is
   missing from the Dukascopy feed. Because intraday volatility is U-shaped, losing the open
   biases realized variance downward by far more than the missing time fraction — measured at
   **24% low** against the coverage-independent Parkinson benchmark (0.786 vs 1.030). Those
   days are nulled. NKY therefore has 2,265 valid RV days against ~3,100 for the others.
2. **The realized measure sees only 33–58% of daily variance.** RV covers the cash session;
   the daily return spans close to close. The Hansen–Lunde scale factor runs 1.71 (DAX) to
   3.04 (NKY), ordered exactly by session length. The Realized GARCH measurement equation must
   absorb this, and the constant is index-specific.
3. **`LogRV` and `LogRS_neg` are the same variable** up to an almost-constant (VIF ≈ 95),
   because `RS_pos + RS_neg = RV` identically and the downside share sits at 0.50. Use the
   level-plus-share parameterisation: max VIF falls from 21.9 to 8.1.

### Gates and flags you must respect

| Flag | Meaning |
|---|---|
| `RV_Valid` | **The gate for every realized measure.** True only for FULL sessions. All RV-derived columns are NaN elsewhere. |
| `SessionClass` | `FULL` / `HALFDAY` / `DEFECT` / `MISSING`. Half-days are real short sessions, defects are feed failures — separated on where the missing 5-min blocks sit. |
| `InSample_B` | The primary window. Use for per-index work. |
| `BalancedRV_B` | 1,994 dates where all six indices have valid RV. **Pooled cross-index statistics only** — using it per-index throws away 5,909 good index-days. |

### Decisions worth knowing before you model

- **Returns are not winsorized, trimmed or de-jumped.** The tail is the object of study;
  clipping it would shrink the estimated GPD shape and manufacture well-calibrated VaR.
- **Rows are never deleted for a bad intraday feed** — only the realized columns are nulled.
  The exchange close is still valid, so the return is still an observation.
- **Macro is forward-filled ≤5 business days, forward only.** On a day the local market trades
  and the US does not, the last published value *is* the information set.
- **`US10Y_pct` and `TermSpread_pct` are non-stationary in levels** (ADF p = 0.87 / 0.33). Use
  `US10Y_diff` and `TermSpread_diff` as regressors.
- **`ScaleFactor_HL` is a full-sample constant.** Fine descriptively; re-estimate per window
  for strict recursive out-of-sample work.

### What the diagnostics say about model choice

- Hill tail index **2.6–3.9**, every estimate below 4 — the fourth moment does not exist.
  This is the quantitative case for the EVT stage over Gaussian or Student-t GARCH.
- Engle–Ng sign-bias rejects on all six (worst p = 7e-5). **Plain GARCH(1,1) is misspecified;
  GJR or EGARCH is required.**
- GPH fractional-integration estimate on log RV is **0.50–0.63**, with ADF and KPSS both
  rejecting — long memory, which is the case for the HAR cascade.
- Recommended POT threshold: **95th–97.5th percentile** of the standardised residuals
  (230–460 exceedances). See figure 07.
- For forecasting the *level* of volatility the realized measures win; **for the 1% tail the
  implied-volatility index wins**. That asymmetry is a result worth reporting.
- `RangePct` from the daily high–low alone reaches R² = 0.42 for next-day log RV and is
  available on *every* day — the natural fallback across the Nikkei gap.

### Validation

`22_validate_analysis.py` runs **158 independent checks, 0 failures**, including a
prefix-stability test for look-ahead: every time-dependent column is rebuilt from truncated
data and compared at the cut. Identities (`RS_pos+RS_neg=RV`, `ContVar+Jump=RV`) hold to
floating-point precision.

### Files

| Path | Contents |
|---|---|
| `01_ANALYSIS_READY/<CODE>_analysis.csv` | **The model-ready dataset.** 96 columns, 1990→2026. |
| `01_ANALYSIS_READY/DATA_DICTIONARY.csv` | Every column: units, availability, timing, caveats. |
| `01_ANALYSIS_READY/FEATURE_SETS.csv` | Which fields feed which of the three models. |
| `01_ANALYSIS_READY/SCALE_FACTORS.csv` | Hansen–Lunde factors per index. |
| `06_REALIZED_MEASURES/<CODE>_RV_extended.csv` | Semivariance, MedRV, quarticity, subsampled RV. |
| `EDA_REPORT.md` / `.xlsx` | The full cleaning and EDA record. |
| `09_FIGURES/*.png` | Ten diagnostic figures. |
| `08_VALIDATION/eda*.csv` | Every table behind the report. |
