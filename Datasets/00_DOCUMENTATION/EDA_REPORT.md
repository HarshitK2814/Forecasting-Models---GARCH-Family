# Exploratory Data Analysis and Preprocessing Report
### Forecasting volatility and tail risk: GARCH-EVT vs Realized GARCH vs Quantile Regression
Researcher A — data acquisition and preprocessing · generated 2026-08-23

---

## 0. What this document is

This is the record of the cleaning and exploratory analysis that turns the raw downloads
into the dataset the three models are estimated on. It is organised so that every cleaning
rule appears next to the evidence that forced it, because in a tail-risk study the
preprocessing choices move the results at least as much as the model choice does.

Headline: **55,010 index-days across six indices**, of which **2,685 days**
are common to all six inside the primary sample and **1,994 days** additionally have a
valid realized measure on every index.

Three findings changed the design and are the ones to read first:

1. **The Nikkei intraday feed is broken for 2016 and 2017** — the entire 09:00 opening hour
   is absent — and the resulting realized variance is biased downward by roughly a quarter.
   These days are nulled, not silently used. (§2)
2. **The realized measure only sees 33–58% of daily variance**
   because it covers the cash session and the daily return spans close to close. The
   Hansen–Lunde scaling factor runs from 1.71 to
   3.04, ordered exactly by session length. (§3)
3. **`LogRV` and `LogRS_neg` are the same variable** up to an almost-constant — VIF ≈ 95.
   The realized-measure block has to be re-parameterised as level plus share. (§6)

---

## 1. Sample definition

| Sample   |   N_Indices | Start      | End        | Binding_Index   |   Common_Days |   Approx_Years |
|:---------|------------:|:-----------|:-----------|:----------------|--------------:|---------------:|
| A        |           5 | 2011-09-20 | 2026-08-21 | NKY             |          3056 |          12.13 |
| B        |           6 | 2013-09-30 | 2026-08-21 | DAX             |          2685 |          10.65 |
| C        |           6 | 2018-01-22 | 2026-08-21 | NKY             |          1892 |           7.51 |

**Sample B is primary.** All six indices, from the date the DAX intraday history begins,
with the Nikkei on its declared VXEFA proxy. Sample A drops the euro area entirely and
sample C spends a third of the observations to remove one declared proxy; both are retained
as robustness runs.

An important distinction that the file encodes explicitly:

- `InSample_B` — the window. Use for per-index work.
- `CommonDate_B` (2,685 days) — dates on which all six indices traded.
- `BalancedRV_B` (1,994 days) — dates on which all six additionally have a *valid*
  realized measure. **Use only for pooled cross-index statistics.**

Per-index estimation and per-index Diebold–Mariano or Model Confidence Set comparisons
should use each index's own valid days. Forcing a balanced panel there would discard
5,909 perfectly good index-days to accommodate the worst
index.

---

## 2. Data quality audit

### 2.1 Structural integrity — clean

| Code   |   Rows |   Dup_Dates |   Weekend_Dates |   Gaps_Over_7d |   ZeroReturn_Pct |   RepeatedClose_MaxRun |   Zero_RV |   OHLC_Violations |
|:-------|-------:|------------:|----------------:|---------------:|-----------------:|-----------------------:|----------:|------------------:|
| SPX    |   9227 |           0 |               0 |              0 |            0.054 |                      1 |         0 |                 0 |
| NDX    |   9227 |           0 |               0 |              0 |            0.065 |                      1 |         0 |                 0 |
| UKX    |   9255 |           0 |               0 |              0 |            0.151 |                      1 |         0 |                 0 |
| DAX    |   9270 |           0 |               0 |              0 |            0     |                      0 |         0 |                 0 |
| NKY    |   8990 |           0 |               0 |              1 |            0.011 |                      1 |         0 |                 0 |
| HSI    |   9041 |           0 |               0 |              0 |            0.055 |                      1 |         0 |                 0 |

No duplicated dates, no weekend dates, no OHLC ordering violations, no zero realized
variances, and no runs of repeated closing prices. Zero-return frequency is at most
0.15% — normal for liquid index data, not a staleness problem.

### 2.2 The intraday session classification

A blanket "drop any day below 90% coverage" rule would have been wrong, because it conflates
two entirely different things: a genuine exchange half-day, where the market really did close
early and the measured variance is correct for a short session, and a feed failure, where the
market was open and we simply lack the data. They are separable by *where* the minutes are
missing — a half-day is a contiguous truncation at the end, a defect is a missing open or an
interior hole.

Coverage is measured on the **5-minute grid**, not the 1-minute grid. A quiet minute with no
quote change is dropped upstream as indistinguishable from padding, so a perfectly good
session shows 93–97% *minute* coverage; a first pass using minutes produced 674 "half-days"
for the S&P against roughly three a year that the exchange actually schedules.

| Symbol   |   DEFECT |   FULL |   HALFDAY |   Total |   Pct_DEFECT |
|:---------|---------:|-------:|----------:|--------:|-------------:|
| SPX      |      346 |   3297 |       117 |    3760 |         9.2  |
| NDX      |      122 |   3496 |       119 |    3737 |         3.26 |
| UKX      |       81 |   3641 |        17 |    3739 |         2.17 |
| DAX      |       30 |   3280 |         2 |    3312 |         0.91 |
| NKY      |      672 |   2869 |         8 |    3549 |        18.93 |
| HSI      |       57 |   3497 |        32 |    3586 |         1.59 |

The half-days land on exactly the dates they should — 07-03, 07-04 and 12-24 for the US
indices, 12-24 and 12-31 for London and Hong Kong — which is the confirmation that the
classifier is separating the two populations correctly.

### 2.3 The Nikkei defect

`NKY` shows 672 defective sessions, 18.9%
of its history, and they are not scattered. **2016 and 2017 are 100% defective**: the
Dukascopy feed carries no bars at all in the 09:00 local hour, and 2015, 2018 and 2019 are
partially degraded.

This is the worst possible shape for the problem. Intraday volatility is U-shaped, so the
opening hour carries far more than its proportional share of the session's variance. Losing
it does not bias RV by the missing time fraction — it biases it by much more, and the
scaling factor that would correct it can only be estimated from the very period we are
missing.

The bias was measured rather than assumed, against two benchmarks that do not depend on
intraday coverage at all — the squared daily return and the Parkinson high–low estimator
from the exchange daily bar:

| NKY session class | n | RV / Parkinson (median) |
|---|---|---|
| FULL | 2265 | 1.030 |
| DEFECT | 598 | 0.786 |

Realized variance on defect days runs about **24% below** where it should. For the S&P, whose
defects are few and different in character, the same comparison gives 1.135 against 1.035 —
a 9% gap. The Nikkei defect is real, large, and one-directional.

**Rule: realized measures are nulled on DEFECT sessions; the row is kept.** The exchange close
is still correct on those days, so the daily return is still a valid observation. Deleting the
row would shorten the return series that GARCH-EVT and the quantile regression estimate on,
and would do so non-randomly with respect to volatility.

---

## 3. The realized measure does not measure the whole day

Realized variance covers the cash session. The daily return spans close to close and so also
contains the overnight gap. RV is therefore a biased-low estimator of daily variance *by
construction*, and the gap is large:

| Code   |   Session variance share, % |   Hansen-Lunde scale factor |
|:-------|----------------------------:|----------------------------:|
| SPX    |                        55.3 |                      1.8085 |
| NDX    |                        55.5 |                      1.8026 |
| UKX    |                        57.5 |                      1.7382 |
| DAX    |                        58.4 |                      1.7116 |
| NKY    |                        32.9 |                      3.0394 |
| HSI    |                        46.7 |                      2.1409 |

The ordering is the check that this is real and not an artefact: the scale factor rises
monotonically as the cash session shortens — DAX and FTSE at 8.5 hours sit at 1.71 and 1.74,
the US indices at 6.5 hours at 1.80, Hong Kong at 5.5 hours at 2.14, and Tokyo at 5 hours at
3.04. The Nikkei's overnight window contains the entire US session, which is why two thirds
of its daily variance is invisible to the intraday data.

**Consequence for the Realized GARCH measurement equation:** the intercept must absorb this
scale gap, or `RV_Scaled` must be used instead of `RV`. It is not optional and it is not the
same constant across indices.

---

## 4. Distributional properties of returns

| Code   |    N |    Mean |      SD |     Skew |   ExcessKurt |      Min |       P1 |     P99 |     Max |
|:-------|-----:|--------:|--------:|---------:|-------------:|---------:|---------:|--------:|--------:|
| SPX    | 9226 | 0.00033 | 0.01134 | -0.36397 |     10.8668  | -0.12765 | -0.03154 | 0.02982 | 0.10957 |
| NDX    | 9226 | 0.00053 | 0.0166  | -0.01216 |      6.31497 | -0.13003 | -0.04616 | 0.0443  | 0.17203 |
| UKX    | 9254 | 0.00016 | 0.01073 | -0.30677 |      7.80072 | -0.11512 | -0.03102 | 0.0281  | 0.09384 |
| DAX    | 9269 | 0.00029 | 0.01369 | -0.20325 |      5.87264 | -0.13055 | -0.03999 | 0.03456 | 0.10797 |
| NKY    | 8989 | 6e-05   | 0.01491 | -0.17038 |      5.92599 | -0.13234 | -0.04033 | 0.03763 | 0.13235 |
| HSI    | 9040 | 0.00025 | 0.01554 | -0.07965 |      9.18364 | -0.14735 | -0.04247 | 0.04054 | 0.17247 |

Annualised volatility over the full history: SPX 18.0%, NDX 26.3%, UKX 17.0%, DAX 21.7%, NKY 23.7%, HSI 24.7%.

Every index shows negative skew and heavy excess kurtosis (5.9
to 10.9). Jarque–Bera rejects normality at any conventional
level for all six.

### 4.1 Tail index — the case for EVT

Hill estimates of the tail index alpha, left tail, k = 5% of observations:

| Code   | Scope      |   k |   Hill_Alpha |    SE |
|:-------|:-----------|----:|-------------:|------:|
| SPX    | full 1990+ | 213 |        2.982 | 0.204 |
| NDX    | full 1990+ | 208 |        3.618 | 0.251 |
| UKX    | full 1990+ | 220 |        3.242 | 0.219 |
| DAX    | full 1990+ | 217 |        3.147 | 0.214 |
| NKY    | full 1990+ | 219 |        3.384 | 0.229 |
| HSI    | full 1990+ | 218 |        2.966 | 0.201 |

Alpha runs from **2.97 to 3.62**. Every estimate is below 4, which means
the fourth moment does not exist and the sample kurtosis reported above is not estimating any
finite population quantity. Several are close to 3, putting the third moment in doubt too.

This is the quantitative case for the EVT stage. A Gaussian-innovation GARCH assumes all
moments exist; a Student-t GARCH imposes a single tail parameter on both tails and on the
whole distribution at once. Neither is consistent with alpha near 3.

### 4.2 Specification tests

| Code   |   ADF_Return_p |   KPSS_Return_p |   LB10_Return_p |   LB22_RetSq_p |   ARCH_LM10_p |   EngleNg_p |
|:-------|---------------:|----------------:|----------------:|---------------:|--------------:|------------:|
| SPX    |              0 |          0.1    |         0       |              0 |             0 |       0     |
| NDX    |              0 |          0.1    |         0       |              0 |             0 |       0     |
| UKX    |              0 |          0.1    |         0       |              0 |             0 |       0     |
| DAX    |              0 |          0.1    |         0.10074 |              0 |             0 |       7e-05 |
| NKY    |              0 |          0.0102 |         0.01716 |              0 |             0 |       0     |
| HSI    |              0 |          0.1    |         0.00822 |              0 |             0 |       0     |

Reading these as the modelling decisions they imply:

- **ADF rejects a unit root everywhere**, KPSS does not reject stationarity except marginally
  for NKY. Returns can be modelled in levels.
- **Ljung–Box on returns rejects for five of six** (DAX at p = 0.10
  is the exception). A conditional-mean term is warranted; an AR(1) is sufficient.
- **Ljung–Box on squared returns and Engle's ARCH-LM reject at p < 1e-16 everywhere.** A
  conditional-variance model is not merely defensible, it is mandatory.
- **Engle–Ng sign-bias rejects everywhere** (worst p = 6.6e-05). The
  variance response to negative and positive shocks differs. **Plain GARCH(1,1) is
  misspecified for this data; GJR or EGARCH is required.** This is the single most actionable
  test in the report.

### 4.3 Realized-measure dynamics

| Code   |   ADF_LogRV_p |   KPSS_LogRV_p |   GPH_d_LogRV |   AC1_LogRV |   AC22_LogRV |   AC66_LogRV |   Leverage_corr_r_LogRVnext |
|:-------|--------------:|---------------:|--------------:|------------:|-------------:|-------------:|----------------------------:|
| SPX    |             0 |         0.01   |        0.5706 |      0.7867 |       0.3617 |       0.1454 |                     -0.1924 |
| NDX    |             0 |         0.01   |        0.5702 |      0.7748 |       0.3877 |       0.1643 |                     -0.203  |
| UKX    |             0 |         0.0516 |        0.6262 |      0.77   |       0.3816 |       0.2482 |                     -0.1758 |
| DAX    |             0 |         0.0507 |        0.5012 |      0.7412 |       0.3545 |       0.2395 |                     -0.1848 |
| NKY    |             0 |         0.0146 |        0.6189 |      0.6731 |       0.2661 |       0.0658 |                     -0.1579 |
| HSI    |             0 |         0.01   |        0.5074 |      0.7061 |       0.3976 |       0.2862 |                     -0.1198 |

Log realized variance is strongly persistent: first-order autocorrelation around 0.67–0.79,
still 0.07–0.29 at a lag of 66 trading days. The
GPH fractional-integration estimate is **0.50–0.63**, and ADF and KPSS
*both* reject — the classic long-memory signature rather than a clean I(0) or I(1).

This is the case for the HAR cascade and for Realized GARCH over a short-memory ARMA
specification on the realized measure.

The leverage correlation between today's signed return and tomorrow's log realized variance
runs **-0.20 to -0.12**, negative for every index — the same asymmetry the
Engle–Ng test picks up, now visible directly in the realized measure. Figure 09 shows the
news-impact curve.

---

## 5. Volatility regimes and crisis coverage

Annualised volatility by year inside sample B:

|   Year |   DAX |   HSI |   NDX |   NKY |   SPX |   UKX |
|-------:|------:|------:|------:|------:|------:|------:|
|   2013 |  11.1 |  13.4 |  11.9 |  18.5 |  10.5 |   9.5 |
|   2014 |  16.8 |  14.4 |  14.1 |  20.7 |  11.4 |  11.4 |
|   2015 |  23.6 |  20.8 |  17.9 |  17.4 |  15.5 |  17.4 |
|   2016 |  20.9 |  18.7 |  16.2 |  26.6 |  13.1 |  16.8 |
|   2017 |  10.6 |  11.1 |  10.3 |  11.9 |   6.7 |   8.6 |
|   2018 |  15.6 |  19.8 |  22.8 |  18.8 |  17.1 |  12.6 |
|   2019 |  14   |  16   |  16.4 |  14   |  12.5 |  11.8 |
|   2020 |  33.2 |  23.7 |  36.6 |  26.3 |  34.7 |  29.8 |
|   2021 |  14.4 |  20.3 |  18.6 |  18.6 |  13.1 |  12.6 |
|   2022 |  23.2 |  33   |  32.5 |  20.6 |  24.2 |  16.6 |
|   2023 |  13   |  22.4 |  18.1 |  16.2 |  13.1 |  11.7 |
|   2024 |  12   |  25.4 |  18.2 |  26.8 |  12.7 |   9.5 |
|   2025 |  17.7 |  24.7 |  23.5 |  24.2 |  18.6 |  12.2 |
|   2026 |  17.6 |  19.8 |  21.3 |  32.5 |  13.6 |  12.8 |

Days worse than −3%, per year — the observations that actually identify a tail:

|   Year |   DAX |   HSI |   NDX |   NKY |   SPX |   UKX |
|-------:|------:|------:|------:|------:|------:|------:|
|   2013 |     0 |     0 |     0 |     0 |     0 |     0 |
|   2014 |     1 |     0 |     1 |     5 |     0 |     0 |
|   2015 |     8 |     5 |     3 |     1 |     3 |     2 |
|   2016 |     4 |     4 |     5 |    13 |     1 |     2 |
|   2017 |     0 |     0 |     0 |     0 |     0 |     0 |
|   2018 |     1 |     4 |    10 |     4 |     5 |     1 |
|   2019 |     1 |     0 |     5 |     1 |     1 |     1 |
|   2020 |    19 |     7 |    19 |     8 |    16 |    17 |
|   2021 |     1 |     5 |     2 |     3 |     0 |     1 |
|   2022 |     7 |    13 |    21 |     2 |     8 |     3 |
|   2023 |     2 |     2 |     0 |     0 |     0 |     1 |
|   2024 |     0 |     3 |     4 |     5 |     1 |     0 |
|   2025 |     5 |     2 |     7 |     5 |     3 |     2 |
|   2026 |     1 |     1 |     2 |    11 |     0 |     0 |

Sample B contains 2020 (33–37% annualised), 2022 (17–33%), the 2015–16 China devaluation,
the 2018 volatility shock and the 2025–26 period, against a very quiet 2017. Total sub-−3%
days per index range from 30 to
79. That is enough distinct stress episodes
to identify a tail without the estimate resting on a single crisis.

**What sample B does not contain is 2008.** The daily-only models see it — they estimate from
1990 — but the Realized GARCH cannot. This asymmetry in the information available to the
three models must be stated in the paper; it is a consequence of when free intraday data
begins, not a choice.

---

## 6. Predictor screening

### 6.1 Stationarity

Non-stationary in levels on all six indices (ADF fails to reject at 5%): **US10Y_pct, TermSpread_pct**.

Both are stationary in first differences at p < 0.001. The differenced forms `US10Y_diff`
and `TermSpread_diff` are supplied and the levels are marked in the data dictionary as unfit
for use as regressors. The implied-volatility indices are stationary in levels and may be
used as they are.

### 6.2 Collinearity

The naive realized-measure block is unusable:

| parameterisation | max mean VIF |
|---|---|
| `LogRV` + `LogRS_neg` + HAR terms | 21.9 |
| `LogRV` + `RSV_Ratio` + `JumpShare` + `RSkew` + HAR terms | 8.1 |

The reason is an identity, not an accident: `RS_pos + RS_neg = RV` exactly, and the downside
share sits at 0.503 on average with little
variation, so `log RS_neg` is `log RV` plus an almost-constant. The fix is to re-parameterise
into a **level** and a **share** rather than to drop a variable — the asymmetry information is
real and worth keeping, it just cannot be carried by a second log-level term.

The mechanical identities were verified numerically and hold to within 1e-12 against a
median RV of order 3e-5, i.e. to floating-point precision.

### 6.3 What actually predicts

Averaged across the six indices. `R2_LogRV_next` is the predictive R² for next-day log
realized variance with Newey–West standard errors; `q01`/`q05` are Koenker–Machado pseudo-R²
for the 1% and 5% quantile of the next-day return. **All predictors are dated t, all targets
t+1** — these are forecasting numbers, not contemporaneous correlations.

| Predictor   |     R2 |    q01 |    q05 |
|:------------|-------:|-------:|-------:|
| LogRV_w     | 0.5671 | 0.1564 | 0.0943 |
| LogRV       | 0.5531 | 0.1813 | 0.1045 |
| LogRS_neg   | 0.5158 | 0.1786 | 0.0995 |
| LogRV_m     | 0.4251 | 0.0931 | 0.0638 |
| LogIV       | 0.4219 | 0.1814 | 0.1049 |
| RangePct    | 0.4169 | 0.1657 | 0.0841 |
| VolIdx      | 0.4093 | 0.1983 | 0.1107 |
| IV_DailyVar | 0.3244 | 0.1986 | 0.1055 |
| RSneg_w     | 0.2914 | 0.179  | 0.0875 |
| RV_w        | 0.2885 | 0.1788 | 0.0871 |

For the **1% left tail** specifically, the ranking changes:

| Predictor   |     R2 |    q01 |    q05 |
|:------------|-------:|-------:|-------:|
| IV_DailyVar | 0.3244 | 0.1986 | 0.1055 |
| ContVar     | 0.2495 | 0.1983 | 0.0898 |
| VolIdx      | 0.4093 | 0.1983 | 0.1107 |
| RV          | 0.2497 | 0.1928 | 0.0876 |
| LogIV       | 0.4219 | 0.1814 | 0.1049 |
| LogRV       | 0.5531 | 0.1813 | 0.1045 |
| RSneg_w     | 0.2914 | 0.179  | 0.0875 |
| RV_w        | 0.2885 | 0.1788 | 0.0871 |
| LogRS_neg   | 0.5158 | 0.1786 | 0.0995 |
| RangePct    | 0.4169 | 0.1657 | 0.0841 |

Two results worth carrying into the modelling:

- **For forecasting the level of volatility, the realized measures win** — the weekly HAR
  term `LogRV_w` leads at R² = 0.567.
- **For forecasting the 1% tail, the implied-volatility index wins.** `IV_DailyVar` and
  `VolIdx` top the tail table. The option market carries forward-looking information about
  extreme outcomes that the backward-looking realized measures do not. That is a genuine
  finding and a natural thing for the paper to say.
- **`RangePct`, computed from the exchange daily high–low alone, reaches R² =
  0.417** — close to the implied-vol index and
  available on *every* day, including the ones where the intraday feed failed. It is the
  natural fallback regressor for the Nikkei's 2016–17 gap.

---

## 7. Cross-index structure

Correlation of log realized variance on the 1,994 balanced days:

| A   |   SPX |   NDX |   UKX |   DAX |   NKY |   HSI |
|:----|------:|------:|------:|------:|------:|------:|
| SPX | 1     | 0.956 | 0.704 | 0.722 | 0.391 | 0.384 |
| NDX | 0.956 | 1     | 0.653 | 0.658 | 0.407 | 0.419 |
| UKX | 0.704 | 0.653 | 1     | 0.851 | 0.396 | 0.311 |
| DAX | 0.722 | 0.658 | 0.851 | 1     | 0.375 | 0.293 |
| NKY | 0.391 | 0.407 | 0.396 | 0.375 | 1     | 0.306 |
| HSI | 0.384 | 0.419 | 0.311 | 0.293 | 0.306 | 1     |

Correlation of daily returns:

| A   |   SPX |   NDX |   UKX |   DAX |   NKY |   HSI |
|:----|------:|------:|------:|------:|------:|------:|
| SPX | 1     | 0.932 | 0.498 | 0.542 | 0.162 | 0.182 |
| NDX | 0.932 | 1     | 0.366 | 0.454 | 0.135 | 0.172 |
| UKX | 0.498 | 0.366 | 1     | 0.807 | 0.296 | 0.349 |
| DAX | 0.542 | 0.454 | 0.807 | 1     | 0.318 | 0.318 |
| NKY | 0.162 | 0.135 | 0.296 | 0.318 | 1     | 0.409 |
| HSI | 0.182 | 0.172 | 0.349 | 0.318 | 0.409 | 1     |

The first principal component of the log-RV correlation matrix explains 62.2% of the
variation, with a clear regional block structure: SPX–NDX at 0.96, UKX–DAX at 0.85, and Asia
largely detached.

**The low Asia–US correlations are a timing artefact and must not be read as economic
independence.** Tokyo and Hong Kong close before New York opens, so a US shock on day *t*
reaches Asia on day *t+1*. Any pooled analysis that aligns these six indices on the calendar
date is mixing information sets. Either adopt a lagged-information convention or estimate
index by index and pool only the loss series — and say which in the paper.

---

## 8. EVT threshold selection

GPD shape parameter ξ fitted to the left tail of **rolling-standardised** returns across
candidate thresholds. Standardising first is what makes this the right diagnostic: McNeil–Frey
applies the GPD to GARCH residuals, not to raw returns, because the limit theory assumes
independence.

|     q |   DAX |   HSI |   NDX |   NKY |   SPX |   UKX |
|------:|------:|------:|------:|------:|------:|------:|
| 0.8   | 0.034 | 0.099 | 0.036 | 0.074 | 0.086 | 0.082 |
| 0.85  | 0.056 | 0.144 | 0.059 | 0.084 | 0.107 | 0.104 |
| 0.9   | 0.071 | 0.214 | 0.093 | 0.133 | 0.14  | 0.116 |
| 0.925 | 0.099 | 0.226 | 0.088 | 0.132 | 0.17  | 0.114 |
| 0.95  | 0.163 | 0.207 | 0.118 | 0.153 | 0.175 | 0.143 |
| 0.96  | 0.168 | 0.181 | 0.098 | 0.202 | 0.144 | 0.135 |
| 0.97  | 0.187 | 0.196 | 0.137 | 0.196 | 0.174 | 0.155 |
| 0.975 | 0.222 | 0.196 | 0.159 | 0.277 | 0.181 | 0.126 |
| 0.98  | 0.245 | 0.227 | 0.173 | 0.252 | 0.23  | 0.171 |
| 0.99  | 0.209 | 0.172 | 0.228 | 0.183 | 0.381 | 0.19  |

Exceedance counts at the same thresholds:

|     q |   DAX |   HSI |   NDX |   NKY |   SPX |   UKX |
|------:|------:|------:|------:|------:|------:|------:|
| 0.8   |  1834 |  1788 |  1825 |  1778 |  1825 |  1831 |
| 0.85  |  1376 |  1341 |  1369 |  1334 |  1369 |  1373 |
| 0.9   |   917 |   894 |   913 |   889 |   913 |   916 |
| 0.925 |   688 |   671 |   685 |   667 |   685 |   687 |
| 0.95  |   459 |   447 |   457 |   445 |   457 |   458 |
| 0.96  |   367 |   358 |   365 |   356 |   365 |   367 |
| 0.97  |   276 |   269 |   274 |   267 |   274 |   275 |
| 0.975 |   230 |   224 |   229 |   223 |   229 |   229 |
| 0.98  |   184 |   179 |   183 |   178 |   183 |   184 |
| 0.99  |    92 |    90 |    92 |    89 |    92 |    92 |

ξ is positive throughout — a heavy, Fréchet-domain tail that survives volatility
standardisation, which is precisely the McNeil–Frey argument for a second EVT stage on top of
the GARCH filter. ξ drifts upward at low thresholds, where the GPD approximation has not bitten,
and becomes unstable above the 98th percentile, where fewer than 190 exceedances remain.

**Recommended POT threshold: the 95th–97.5th percentile of the standardised residuals**, giving
roughly 230–460 exceedances per index. Figure 07 shows the stability plot with this region
shaded. Re-estimate on the actual GARCH residuals once the stage-1 model is fitted — these
numbers set the expectation, they do not replace that step.

---

## 9. Cleaning decisions, in full

| # | Decision | Reason |
|---|---|---|
| 1 | Realized measures nulled on DEFECT sessions, rows retained | The close is still valid, so the return is still an observation. Deleting rows would shorten the return series non-randomly with respect to volatility. |
| 2 | Half-days flagged and excluded from `RV_Valid`, values retained | A three-hour variance fed into a measurement equation calibrated on full sessions biases the intercept. Reversible by relaxing the `RV_Valid` gate. |
| 3 | **Returns are not winsorized, trimmed or de-jumped** | The tail is the object of study. Clipping extremes would shrink the estimated GPD shape parameter and produce VaR that appears well-calibrated only because the exceedances were deleted. The extreme days were individually verified as real market events. |
| 4 | Macro forward-filled ≤5 business days, forward only | These are US series on six exchange calendars. The last published value *is* the information set on a day the US did not trade. Backward filling would be look-ahead and is never used. |
| 5 | Hansen–Lunde scale factor estimated per index | The session/close-to-close gap is 1.71×–3.04× and index-specific. Measured, not assumed away. |
| 6 | Nothing standardised, differenced or de-meaned in storage | Those are estimator-specific choices. Levels are stored; transformations belong in the modelling code. |
| 7 | Non-stationary macro levels supplemented with differences | ADF p = 0.87 for the 10-year yield level. Levels retained for plotting, flagged as unfit as regressors. |
| 8 | Realized block re-parameterised as level + share | `LogRV` and `LogRS_neg` are collinear by identity (VIF ≈ 95). |

---

## 10. Known limitations

1. **The Nikkei has no valid realized measure for 2016–17.** Its Realized GARCH rests on
   2,265 days with a two-year hole. A rolling window will span
   that gap; it must use the last *available* observations and forecasts cannot be produced or
   evaluated inside it. Disclose prominently.
2. **The intraday data is a bid-side index CFD, not the exchange index.** Aligned correlation
   against the exchange daily return is 0.97–0.99, so it tracks well, but it is a proxy. The
   subsampled `RV_ss` and the 1-min/5-min noise ratio of 1.06–1.09 both indicate a smoothed
   feed with little microstructure noise — which also means RV may be mildly damped relative
   to a trade-based estimator.
3. **The Nikkei volatility index is a proxy** (VXEFA) in the primary sample. Sample C exists
   to test that this does not drive anything.
4. **The FTSE and Hang Seng have no native volatility index at all** in free data; VXEFA and
   VXEEM stand in.
5. **Sample B excludes 2008.** The three models therefore see different amounts of crisis
   history. Unavoidable given when free intraday data begins.
6. **Sessions are not synchronous.** See §7.

---

## 11. Files

| Path | Contents |
|---|---|
| `01_ANALYSIS_READY/<CODE>_analysis.csv` | The model-ready dataset. 96 columns, 1990→2026. |
| `01_ANALYSIS_READY/DATA_DICTIONARY.csv` | Every column: units, availability, timing, caveats. |
| `01_ANALYSIS_READY/FEATURE_SETS.csv` | Which fields feed which of the three models. |
| `01_ANALYSIS_READY/SCALE_FACTORS.csv` | Hansen–Lunde factors per index. |
| `06_REALIZED_MEASURES/<CODE>_RV_extended.csv` | Semivariance, MedRV, quarticity, subsampled RV. |
| `08_VALIDATION/eda*.csv` | Every table in this report. |
| `09_FIGURES/*.png` | The ten diagnostic figures. |
| `07_PANEL_INTERMEDIATE/<CODE>_panel_daily.csv` | The pre-cleaning join, kept for provenance. |

### Reproduction

```
python Datasets/10_SCRIPTS/11_define_samples.py
python Datasets/10_SCRIPTS/12_extended_realized_measures.py
python Datasets/10_SCRIPTS/13_eda_quality_audit.py
python Datasets/10_SCRIPTS/14_session_classification.py
python Datasets/10_SCRIPTS/15_build_analysis_dataset.py
python Datasets/10_SCRIPTS/16_eda_stylized_facts.py
python Datasets/10_SCRIPTS/17_eda_predictor_screening.py
python Datasets/10_SCRIPTS/18_eda_tails_breaks_features.py
python Datasets/10_SCRIPTS/19_eda_figures.py
python Datasets/10_SCRIPTS/20_finalise_and_document.py
python Datasets/10_SCRIPTS/21_build_eda_report.py
```

Steps 12 and 14 read the ~24,000-file Dukascopy cache and take several minutes each. The rest
run in seconds.
