# -*- coding: utf-8 -*-
"""
MODELLING FIGURES — the visual counterparts to the numbers in 08_VALIDATION/*.csv.

Researcher A. Each figure backs a specific number already reported in
RESEARCHER_A_SCOPE.md / ROBUSTNESS_SUMMARY.md - built so a reviewer (or Researcher B)
can see the claim, not just read it.

  11  VaR_breach.png                Return series with 1% VaR overlaid, breaches marked
  12  forecast_vs_realized.png      GJR-skewt vs Realized GARCH sigma-hat vs sqrt(RV_Scaled)
  13  residual_diagnostics.png      QQ (skew-t) + ACF(resid^2), 6-index grid
  14  subsample_stability.png       Persistence and skew, pre- vs post-COVID, per index
  15  frequency_sensitivity.png     Hansen-Lunde scale factor vs sampling frequency
  16  refit_cadence_overlay.png     SPX sigma-hat, 21-day vs 63-day refit, overlaid
  17  nky_gap.png                   NKY conditional variance through 2016-17, gap shaded

Run after 27-30 have produced their output files.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
RVDIR = os.path.join(ROOT, '06_REALIZED_MEASURES')
VAL = os.path.join(ROOT, '08_VALIDATION')
FC = os.path.join(ROOT, '20_FORECASTS')
FIG = os.path.join(ROOT, '09_FIGURES')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]
COVID_SPLIT = pd.Timestamp('2020-02-20')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
fio = importlib.import_module('26_forecast_io')

plt.rcParams.update({'figure.dpi': 120, 'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3})


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# 11. VaR breach plot
# ---------------------------------------------------------------------------
def fig_var_breach():
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=False)
    for ax, code in zip(axes.flat, CODES):
        fc = fio.read_forecasts(os.path.join(FC, f'GJR-skewt__{code}_forecasts.csv'), strict=False)
        fc['Date'] = pd.to_datetime(fc['Date'])
        v = fc[fc['Valid']]
        breach = v['Realized'] < v['VaR_01']
        ax.plot(v['Date'], v['Realized'] * 100, color='steelblue', linewidth=0.4, label='Return')
        ax.plot(v['Date'], v['VaR_01'] * 100, color='firebrick', linewidth=0.7, label='VaR 1%')
        ax.scatter(v.loc[breach, 'Date'], v.loc[breach, 'Realized'] * 100,
                   color='darkred', s=8, zorder=5, label=f'Breach ({breach.sum()}/{len(v)}={100*breach.mean():.1f}%)')
        ax.set_title(code, fontsize=10, fontweight='bold')
        ax.set_ylabel('%')
        ax.legend(fontsize=6.5, loc='lower left')
        ax.axhline(0, color='grey', linewidth=0.4)
    fig.suptitle('GJR-skewt: daily return vs 1% VaR, sample B walk-forward forecasts', y=1.01, fontsize=12)
    save(fig, '11_VaR_breach.png')


# ---------------------------------------------------------------------------
# 12. forecast vs realized volatility overlay
# ---------------------------------------------------------------------------
def fig_forecast_vs_realized():
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, code in zip(axes.flat, CODES):
        gjr = fio.read_forecasts(os.path.join(FC, f'GJR-skewt__{code}_forecasts.csv'), strict=False)
        rg = fio.read_forecasts(os.path.join(FC, f'RealGARCH__{code}_forecasts.csv'), strict=False)
        for d in (gjr, rg):
            d['Date'] = pd.to_datetime(d['Date'])
        realized_vol = np.sqrt(gjr['RVProxy']) * 100
        ax.plot(gjr['Date'], realized_vol, color='grey', linewidth=0.5, alpha=0.6, label='sqrt(RV_Scaled), realized')
        ax.plot(gjr['Date'], gjr['SigmaHat'] * 100, color='steelblue', linewidth=0.8, label='GJR-skewt sigma-hat')
        ax.plot(rg['Date'], rg['SigmaHat'] * 100, color='darkorange', linewidth=0.8, alpha=0.85, label='Realized GARCH sigma-hat')
        ax.set_title(code, fontsize=10, fontweight='bold')
        ax.set_ylabel('Daily vol, %')
        ax.legend(fontsize=6.5)
    fig.suptitle('1-step-ahead volatility forecasts vs realized volatility, sample B', y=1.01, fontsize=12)
    save(fig, '12_forecast_vs_realized.png')


# ---------------------------------------------------------------------------
# 13. standardised residual diagnostics: QQ (skew-t) + ACF(resid^2)
# ---------------------------------------------------------------------------
def fig_residual_diagnostics():
    params = pd.read_csv(os.path.join(VAL, 'garch_baseline_params.csv'))
    params = params[params['Spec'] == 'GJR-skewt'].set_index('Code')

    fig, axes = plt.subplots(6, 2, figsize=(11, 20))
    for i, code in enumerate(CODES):
        sr = pd.read_csv(os.path.join(RVDIR, f'{code}_std_resid.csv'))
        z = sr['StdResid'].dropna().values
        eta, lam = params.loc[code, 'param_eta'], params.loc[code, 'param_lambda']

        ax_qq, ax_acf = axes[i, 0], axes[i, 1]
        # empirical vs theoretical skew-t quantiles via arch's own distribution object
        from arch.univariate import SkewStudent
        dist = SkewStudent()
        n = len(z)
        pits = (np.arange(1, n + 1) - 0.5) / n
        theo_q = dist.ppf(pits, [eta, lam])
        emp_q = np.sort(z)
        ax_qq.scatter(theo_q, emp_q, s=3, alpha=0.4, color='steelblue')
        lims = [min(theo_q.min(), emp_q.min()), max(theo_q.max(), emp_q.max())]
        ax_qq.plot(lims, lims, color='firebrick', linewidth=1, linestyle='--')
        ax_qq.set_title(f'{code}: QQ vs skew-t(eta={eta:.1f}, lambda={lam:.2f})', fontsize=8)
        ax_qq.set_xlabel('Theoretical quantile', fontsize=7)
        ax_qq.set_ylabel('Empirical', fontsize=7)

        lb = acorr_ljungbox(z ** 2, lags=[20], return_df=True)
        pval = float(lb['lb_pvalue'].iloc[0])
        ac = sm_acf(z ** 2, nlags=20, fft=True)[1:]
        ax_acf.bar(range(1, 21), ac, color='steelblue', width=0.6)
        ci = 1.96 / np.sqrt(n)
        ax_acf.axhline(ci, color='grey', linewidth=0.6, linestyle='--')
        ax_acf.axhline(-ci, color='grey', linewidth=0.6, linestyle='--')
        ax_acf.set_title(f'{code}: ACF(resid^2), Ljung-Box(20) p={pval:.3f}', fontsize=8)
        ax_acf.set_xlabel('Lag', fontsize=7)
    fig.suptitle('GJR-skewt standardised residual diagnostics', y=1.005, fontsize=12)
    save(fig, '13_residual_diagnostics.png')


# ---------------------------------------------------------------------------
# 14. sub-sample stability
# ---------------------------------------------------------------------------
def fig_subsample_stability():
    df = pd.read_csv(os.path.join(VAL, 'robustness_subsample_stability.csv'))
    pre = df[df['Period'] == 'pre_COVID'].set_index('Code')
    post = df[df['Period'] == 'post_COVID'].set_index('Code')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(CODES))
    w = 0.35
    ax1.bar(x - w / 2, pre.loc[CODES, 'Persistence'], w, label='pre-COVID', color='steelblue')
    ax1.bar(x + w / 2, post.loc[CODES, 'Persistence'], w, label='post-COVID', color='darkorange')
    ax1.set_xticks(x); ax1.set_xticklabels(CODES)
    ax1.set_ylabel('Persistence (alpha + 0.5*gamma + beta)')
    ax1.set_title('GJR-skewt persistence, pre vs post COVID-19')
    ax1.set_ylim(0.85, 1.0)
    ax1.legend()

    ax2.bar(x - w / 2, pre.loc[CODES, 'lam_skew'], w, label='pre-COVID', color='steelblue')
    ax2.bar(x + w / 2, post.loc[CODES, 'lam_skew'], w, label='post-COVID', color='darkorange')
    ax2.set_xticks(x); ax2.set_xticklabels(CODES)
    ax2.axhline(0, color='grey', linewidth=0.6)
    ax2.set_ylabel('Skew-t lambda (negative = left skew)')
    ax2.set_title('Innovation skew parameter, pre vs post COVID-19')
    ax2.legend()
    fig.suptitle('Sub-sample stability: split at 2020-02-20', y=1.02, fontsize=12)
    save(fig, '14_subsample_stability.png')


# ---------------------------------------------------------------------------
# 15. sampling-frequency sensitivity
# ---------------------------------------------------------------------------
def fig_frequency_sensitivity():
    df = pd.read_csv(os.path.join(VAL, 'robustness_frequency_sensitivity.csv')).set_index('Code')
    freqs = ['5min', '10min', '15min', '30min']
    xvals = [5, 10, 15, 30]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for code in CODES:
        y = [df.loc[code, f'ScaleFactor_{f}'] for f in freqs]
        ax.plot(xvals, y, marker='o', linewidth=1.2, label=code)
    ax.set_xlabel('Sampling interval (minutes)')
    ax.set_ylabel('Hansen-Lunde scale factor  (sum r^2 / sum RV)')
    ax.set_title('Realized-variance scale factor vs sampling frequency')
    ax.legend(ncol=3, fontsize=8)
    save(fig, '15_frequency_sensitivity.png')


# ---------------------------------------------------------------------------
# 16. refit-cadence overlay (SPX, 21d vs 63d)
# ---------------------------------------------------------------------------
def fig_refit_cadence():
    engine = importlib.import_module('29_rolling_forecast_engine')
    series = {}
    for cadence in [21, 63]:
        engine.REFIT_EVERY = cadence
        d, refits, elapsed = engine.run_index_spec('SPX', 'GJR-skewt')
        d['Date'] = pd.to_datetime(d['Date'])
        series[cadence] = d.set_index('Date')['SigmaHat'] * 100
    engine.REFIT_EVERY = 21

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(series[21].index, series[21].values, color='steelblue', linewidth=0.8, label='21-day refit')
    ax1.plot(series[63].index, series[63].values, color='darkorange', linewidth=0.8, alpha=0.75, label='63-day refit')
    ax1.set_ylabel('Daily sigma-hat, %')
    ax1.set_title('SPX GJR-skewt: 21-day vs 63-day expanding refit cadence')
    ax1.legend()

    diff = (series[63] - series[21]).dropna()
    ax2.plot(diff.index, diff.values, color='grey', linewidth=0.6)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('63d - 21d, pp')
    corr = series[21].corr(series[63])
    ax2.set_title(f'Difference (correlation = {corr:.5f})', fontsize=9)
    save(fig, '16_refit_cadence_overlay.png')


# ---------------------------------------------------------------------------
# 17. NKY 2016-17 gap illustration
# ---------------------------------------------------------------------------
def fig_nky_gap():
    fit = pd.read_csv(os.path.join(RVDIR, 'NKY_realized_garch_fit.csv'), parse_dates=['Date'])
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(fit['Date'], np.sqrt(fit['CondVar']) * np.sqrt(252) * 100,
            color='steelblue', linewidth=0.7, label='Realized GARCH sigma-hat (annualised %)')
    imputed = fit['RV_Imputed'].astype(bool)
    if imputed.any():
        # shade contiguous imputed blocks
        idx = np.where(imputed.values)[0]
        gaps = np.where(np.diff(idx) > 1)[0]
        starts = [idx[0]] + [idx[g + 1] for g in gaps]
        ends = [idx[g] for g in gaps] + [idx[-1]]
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            ax.axvspan(fit['Date'].iloc[s], fit['Date'].iloc[e], color='firebrick', alpha=0.12)
    ax.axvspan(fit['Date'].iloc[0], fit['Date'].iloc[0], color='firebrick', alpha=0.12,
               label=f'RV imputed in recursion ({int(imputed.sum())} days total)')
    ax.set_ylabel('Annualised volatility, %')
    ax.set_title('NKY Realized GARCH: conditional volatility through the 2016-17 intraday-feed gap')
    ax.legend()
    save(fig, '17_nky_gap.png')


def main():
    os.makedirs(FIG, exist_ok=True)
    print("11/17 VaR breach plot ...");            fig_var_breach()
    print("12/17 forecast vs realized ...");        fig_forecast_vs_realized()
    print("13/17 residual diagnostics ...");        fig_residual_diagnostics()
    print("14/17 sub-sample stability ...");        fig_subsample_stability()
    print("15/17 frequency sensitivity ...");       fig_frequency_sensitivity()
    print("16/17 refit cadence overlay ...");       fig_refit_cadence()
    print("17/17 NKY gap illustration ...");        fig_nky_gap()
    print("\nAll figures written to 09_FIGURES/")


if __name__ == "__main__":
    main()
