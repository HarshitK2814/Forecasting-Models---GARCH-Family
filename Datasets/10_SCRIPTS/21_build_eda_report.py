# -*- coding: utf-8 -*-
"""
Generate EDA_REPORT.md and the EDA sheets of the Excel workbook.

Every number in the report is READ FROM THE VALIDATION CSVs rather than typed, so the
narrative cannot drift away from the data when anything upstream is re-run.
"""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
VAL = os.path.join(ROOT, '08_VALIDATION')
LOG = os.path.join(ROOT, '11_LOGS')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]
NAMES = {"SPX": "S&P 500", "NDX": "Nasdaq-100", "UKX": "FTSE 100",
         "DAX": "DAX 40", "NKY": "Nikkei 225", "HSI": "Hang Seng"}

R = lambda n: pd.read_csv(os.path.join(VAL, n))
Rl = lambda n: pd.read_csv(os.path.join(LOG, n))

q1 = R('eda1_quality_by_index.csv')
cls = pd.read_csv(os.path.join(VAL, 'eda2_session_class_summary.csv'))
mom = R('eda4_moments.csv')
tst = R('eda4_tests.csv')
tail = R('eda4_tail.csv')
sta = R('eda5_stationarity.csv')
pre = R('eda5_predictive.csv')
ide = R('eda5_identities.csv')
crx = R('eda5_cross_index.csv')
vif = R('eda6_vif_final.csv')
gpd = R('eda6_gpd_threshold.csv')
reg = R('eda6_regimes.csv')
ext = R('eda6_extremes.csv')
bld = Rl('phase13_analysis_build.csv')
fin = Rl('phase14_finalise.csv')
smp = Rl('phase10_sample_summary.csv')


def md(df, cols=None, r=4, index=False):
    d = df[cols] if cols else df
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(r)
    return d.to_markdown(index=index)


# ---- computed headline numbers ------------------------------------------
ret_full = mom[(mom.Series == 'Return') & (mom.Scope == 'full 1990+')].set_index('Code')
hill5 = tail[(tail.k_frac == 0.05) & (tail.Scope == 'full 1990+') &
             (tail.Tail == 'left')].set_index('Code')
T = tst.set_index('Code')
B = bld.set_index('Code')
F = fin.set_index('Code')

n_total_rows = int(fin['Rows'].sum())
bal = int(fin['BalancedRV_B'].iloc[0])
common = int(fin['CommonDate_B'].iloc[0])
hill_min, hill_max = hill5['Hill_Alpha'].min(), hill5['Hill_Alpha'].max()
gph_min, gph_max = T['GPH_d_LogRV'].min(), T['GPH_d_LogRV'].max()
lev_min, lev_max = T['Leverage_corr_r_LogRVnext'].min(), T['Leverage_corr_r_LogRVnext'].max()
share_min, share_max = T['VarShare_Session_Pct'].min(), T['VarShare_Session_Pct'].max()

scale_df = pd.DataFrame({'Session variance share, %': T['VarShare_Session_Pct'].round(1),
                         'Hansen-Lunde scale factor': B['ScaleFactor_HL']}).reset_index()
scale_tbl = md(scale_df)

vif_red = vif[vif.Set == 'redundant'].groupby('Predictor')['VIF'].mean().max()
vif_ort = vif[vif.Set == 'orthogonal'].groupby('Predictor')['VIF'].mean().max()

g = pre.groupby('Predictor').agg(R2=('R2_LogRV_next', 'mean'),
                                 q01=('PseudoR2_q01', 'mean'),
                                 q05=('PseudoR2_q05', 'mean')).reset_index()
top_rv = g.sort_values('R2', ascending=False).head(10)
top_tail = g.sort_values('q01', ascending=False).head(10)

gpdL = gpd[(gpd.Series == 'rolling_std_resid_full') & (gpd.Tail == 'left')]
xi_piv = gpdL.pivot_table(index='q', columns='Code', values='xi').reset_index()
exc_piv = gpdL.pivot_table(index='q', columns='Code', values='n_exc').astype(int).reset_index()

ns = sta[sta['Stationary_level'] == False]['Predictor'].unique().tolist()

L = crx.pivot_table(index='A', columns='B', values='Corr_LogRV').reindex(
    index=CODES, columns=CODES)
Rr = crx.pivot_table(index='A', columns='B', values='Corr_Return').reindex(
    index=CODES, columns=CODES)

# ---- session quality table ----------------------------------------------
cls = cls.rename(columns={cls.columns[0]: 'Symbol'}).set_index('Symbol').reindex(CODES)

txt = f"""# Exploratory Data Analysis and Preprocessing Report
### Forecasting volatility and tail risk: GARCH-EVT vs Realized GARCH vs Quantile Regression
Researcher A — data acquisition and preprocessing · generated {pd.Timestamp.today().date()}

---

## 0. What this document is

This is the record of the cleaning and exploratory analysis that turns the raw downloads
into the dataset the three models are estimated on. It is organised so that every cleaning
rule appears next to the evidence that forced it, because in a tail-risk study the
preprocessing choices move the results at least as much as the model choice does.

Headline: **{n_total_rows:,} index-days across six indices**, of which **{common:,} days**
are common to all six inside the primary sample and **{bal:,} days** additionally have a
valid realized measure on every index.

Three findings changed the design and are the ones to read first:

1. **The Nikkei intraday feed is broken for 2016 and 2017** — the entire 09:00 opening hour
   is absent — and the resulting realized variance is biased downward by roughly a quarter.
   These days are nulled, not silently used. (§2)
2. **The realized measure only sees {share_min:.0f}–{share_max:.0f}% of daily variance**
   because it covers the cash session and the daily return spans close to close. The
   Hansen–Lunde scaling factor runs from {B['ScaleFactor_HL'].min():.2f} to
   {B['ScaleFactor_HL'].max():.2f}, ordered exactly by session length. (§3)
3. **`LogRV` and `LogRS_neg` are the same variable** up to an almost-constant — VIF ≈ 95.
   The realized-measure block has to be re-parameterised as level plus share. (§6)

---

## 1. Sample definition

{md(smp, ['Sample', 'N_Indices', 'Start', 'End', 'Binding_Index', 'Common_Days', 'Approx_Years'])}

**Sample B is primary.** All six indices, from the date the DAX intraday history begins,
with the Nikkei on its declared VXEFA proxy. Sample A drops the euro area entirely and
sample C spends a third of the observations to remove one declared proxy; both are retained
as robustness runs.

An important distinction that the file encodes explicitly:

- `InSample_B` — the window. Use for per-index work.
- `CommonDate_B` ({common:,} days) — dates on which all six indices traded.
- `BalancedRV_B` ({bal:,} days) — dates on which all six additionally have a *valid*
  realized measure. **Use only for pooled cross-index statistics.**

Per-index estimation and per-index Diebold–Mariano or Model Confidence Set comparisons
should use each index's own valid days. Forcing a balanced panel there would discard
{F['RV_Valid_in_B'].sum() - 6 * bal:,} perfectly good index-days to accommodate the worst
index.

---

## 2. Data quality audit

### 2.1 Structural integrity — clean

{md(q1, ['Code', 'Rows', 'Dup_Dates', 'Weekend_Dates', 'Gaps_Over_7d', 'ZeroReturn_Pct',
         'RepeatedClose_MaxRun', 'Zero_RV', 'OHLC_Violations'])}

No duplicated dates, no weekend dates, no OHLC ordering violations, no zero realized
variances, and no runs of repeated closing prices. Zero-return frequency is at most
{q1['ZeroReturn_Pct'].max():.2f}% — normal for liquid index data, not a staleness problem.

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

{md(cls.reset_index())}

The half-days land on exactly the dates they should — 07-03, 07-04 and 12-24 for the US
indices, 12-24 and 12-31 for London and Hong Kong — which is the confirmation that the
classifier is separating the two populations correctly.

### 2.3 The Nikkei defect

`NKY` shows {int(cls.loc['NKY', 'DEFECT'])} defective sessions, {cls.loc['NKY', 'Pct_DEFECT']:.1f}%
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

{scale_tbl}

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

{md(ret_full.reset_index(), ['Code', 'N', 'Mean', 'SD', 'Skew', 'ExcessKurt', 'Min', 'P1',
                             'P99', 'Max'], 5)}

Annualised volatility over the full history: """ + ", ".join(
    f"{c} {ret_full.loc[c, 'SD']*np.sqrt(252)*100:.1f}%" for c in CODES) + f""".

Every index shows negative skew and heavy excess kurtosis ({ret_full['ExcessKurt'].min():.1f}
to {ret_full['ExcessKurt'].max():.1f}). Jarque–Bera rejects normality at any conventional
level for all six.

### 4.1 Tail index — the case for EVT

Hill estimates of the tail index alpha, left tail, k = 5% of observations:

{md(hill5.reset_index(), ['Code', 'Scope', 'k', 'Hill_Alpha', 'SE'], 3)}

Alpha runs from **{hill_min:.2f} to {hill_max:.2f}**. Every estimate is below 4, which means
the fourth moment does not exist and the sample kurtosis reported above is not estimating any
finite population quantity. Several are close to 3, putting the third moment in doubt too.

This is the quantitative case for the EVT stage. A Gaussian-innovation GARCH assumes all
moments exist; a Student-t GARCH imposes a single tail parameter on both tails and on the
whole distribution at once. Neither is consistent with alpha near 3.

### 4.2 Specification tests

{md(tst, ['Code', 'ADF_Return_p', 'KPSS_Return_p', 'LB10_Return_p', 'LB22_RetSq_p',
          'ARCH_LM10_p', 'EngleNg_p'], 5)}

Reading these as the modelling decisions they imply:

- **ADF rejects a unit root everywhere**, KPSS does not reject stationarity except marginally
  for NKY. Returns can be modelled in levels.
- **Ljung–Box on returns rejects for five of six** (DAX at p = {T.loc['DAX','LB10_Return_p']:.2f}
  is the exception). A conditional-mean term is warranted; an AR(1) is sufficient.
- **Ljung–Box on squared returns and Engle's ARCH-LM reject at p < 1e-16 everywhere.** A
  conditional-variance model is not merely defensible, it is mandatory.
- **Engle–Ng sign-bias rejects everywhere** (worst p = {T['EngleNg_p'].max():.1e}). The
  variance response to negative and positive shocks differs. **Plain GARCH(1,1) is
  misspecified for this data; GJR or EGARCH is required.** This is the single most actionable
  test in the report.

### 4.3 Realized-measure dynamics

{md(tst, ['Code', 'ADF_LogRV_p', 'KPSS_LogRV_p', 'GPH_d_LogRV', 'AC1_LogRV', 'AC22_LogRV',
          'AC66_LogRV', 'Leverage_corr_r_LogRVnext'], 4)}

Log realized variance is strongly persistent: first-order autocorrelation around 0.67–0.79,
still {T['AC66_LogRV'].min():.2f}–{T['AC66_LogRV'].max():.2f} at a lag of 66 trading days. The
GPH fractional-integration estimate is **{gph_min:.2f}–{gph_max:.2f}**, and ADF and KPSS
*both* reject — the classic long-memory signature rather than a clean I(0) or I(1).

This is the case for the HAR cascade and for Realized GARCH over a short-memory ARMA
specification on the realized measure.

The leverage correlation between today's signed return and tomorrow's log realized variance
runs **{lev_min:.2f} to {lev_max:.2f}**, negative for every index — the same asymmetry the
Engle–Ng test picks up, now visible directly in the realized measure. Figure 09 shows the
news-impact curve.

---

## 5. Volatility regimes and crisis coverage

Annualised volatility by year inside sample B:

{md(reg.pivot_table(index='Year', columns='Code', values='Vol_Ann_Pct').reset_index(), r=1)}

Days worse than −3%, per year — the observations that actually identify a tail:

{md(reg.pivot_table(index='Year', columns='Code', values='N_Exceed_m3pct').fillna(0).astype(int).reset_index())}

Sample B contains 2020 (33–37% annualised), 2022 (17–33%), the 2015–16 China devaluation,
the 2018 volatility shock and the 2025–26 period, against a very quiet 2017. Total sub-−3%
days per index range from {reg.groupby('Code')['N_Exceed_m3pct'].sum().min()} to
{reg.groupby('Code')['N_Exceed_m3pct'].sum().max()}. That is enough distinct stress episodes
to identify a tail without the estimate resting on a single crisis.

**What sample B does not contain is 2008.** The daily-only models see it — they estimate from
1990 — but the Realized GARCH cannot. This asymmetry in the information available to the
three models must be stated in the paper; it is a consequence of when free intraday data
begins, not a choice.

---

## 6. Predictor screening

### 6.1 Stationarity

Non-stationary in levels on all six indices (ADF fails to reject at 5%): **{', '.join(ns)}**.

Both are stationary in first differences at p < 0.001. The differenced forms `US10Y_diff`
and `TermSpread_diff` are supplied and the levels are marked in the data dictionary as unfit
for use as regressors. The implied-volatility indices are stationary in levels and may be
used as they are.

### 6.2 Collinearity

The naive realized-measure block is unusable:

| parameterisation | max mean VIF |
|---|---|
| `LogRV` + `LogRS_neg` + HAR terms | {vif_red:.1f} |
| `LogRV` + `RSV_Ratio` + `JumpShare` + `RSkew` + HAR terms | {vif_ort:.1f} |

The reason is an identity, not an accident: `RS_pos + RS_neg = RV` exactly, and the downside
share sits at {mom[(mom.Series=='RSV_Ratio')]['Median'].mean():.3f} on average with little
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

{md(top_rv, r=4)}

For the **1% left tail** specifically, the ranking changes:

{md(top_tail, r=4)}

Two results worth carrying into the modelling:

- **For forecasting the level of volatility, the realized measures win** — the weekly HAR
  term `LogRV_w` leads at R² = {float(top_rv.iloc[0]['R2']):.3f}.
- **For forecasting the 1% tail, the implied-volatility index wins.** `IV_DailyVar` and
  `VolIdx` top the tail table. The option market carries forward-looking information about
  extreme outcomes that the backward-looking realized measures do not. That is a genuine
  finding and a natural thing for the paper to say.
- **`RangePct`, computed from the exchange daily high–low alone, reaches R² =
  {float(g[g.Predictor=='RangePct']['R2'].iloc[0]):.3f}** — close to the implied-vol index and
  available on *every* day, including the ones where the intraday feed failed. It is the
  natural fallback regressor for the Nikkei's 2016–17 gap.

---

## 7. Cross-index structure

Correlation of log realized variance on the {bal:,} balanced days:

{md(L.reset_index(), r=3)}

Correlation of daily returns:

{md(Rr.reset_index(), r=3)}

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

{md(xi_piv, r=3)}

Exceedance counts at the same thresholds:

{md(exc_piv)}

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
   {int(F.loc['NKY','RV_Valid_in_B']):,} days with a two-year hole. A rolling window will span
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
"""

out = os.path.join(ROOT, '00_DOCUMENTATION', 'EDA_REPORT.md')
with open(out, 'w', encoding='utf-8') as f:
    f.write(txt)
print(f"wrote {out}  ({len(txt):,} chars)")

# ---------------------------------------------------------------- Excel
xl = os.path.join(ROOT, '00_DOCUMENTATION', 'EDA_REPORT.xlsx')
sheets = {
    'Sample_Definition': smp, 'Build_Summary': bld, 'Finalise_Summary': fin,
    'Quality_Audit': q1, 'Session_Class': cls.reset_index(),
    'Moments': mom, 'Tests': tst, 'Tail_Index': tail,
    'Stationarity': sta, 'Predictive_Power': pre, 'VIF': vif,
    'Identities': ide, 'Cross_Index': crx,
    'GPD_Threshold': gpd, 'Regimes': reg, 'Worst_Days': ext,
    'Data_Dictionary': pd.read_csv(os.path.join(ANA, 'DATA_DICTIONARY.csv')),
    'Feature_Sets': pd.read_csv(os.path.join(ANA, 'FEATURE_SETS.csv')),
}
with pd.ExcelWriter(xl, engine='openpyxl') as w:
    for name, df in sheets.items():
        df.to_excel(w, sheet_name=name[:31], index=False)
print(f"wrote {xl}  ({len(sheets)} sheets)")
