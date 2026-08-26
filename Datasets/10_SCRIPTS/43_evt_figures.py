# -*- coding: utf-8 -*-
"""
43 - GARCH-EVT figures. Completes the Task 1 deliverable.

Each figure backs a specific claim rather than illustrating in the abstract:
  43_xi_path.png       does the estimated tail index move over time, or is the
                       movement inside the estimator's own sampling noise?
  43_var_breaches.png  where do the 1% VaR breaches actually fall?
  43_evt_vs_skewt.png  how much wider is the EVT tail than the skew-t it replaces?

Reads only the contract files written by script 42 and A's baseline, so the
figures cannot drift from the numbers in the tables.
"""
import os, importlib.util, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
fio = _load('fio', os.path.join(HERE, '26_forecast_io.py'))

FORECAST = 'Datasets/20_FORECASTS'
INDICES  = ['SPX', 'NDX', 'UKX', 'DAX', 'NKY', 'HSI']
PAL      = ["#378ADD", "#D85A30", "#1D9E75", "#BA7517", "#7F77DD", "#8C8C8C"]
CRISIS   = {"COVID": ("2020-02-20", "2020-04-30"),
            "Rate shock": ("2022-01-03", "2022-10-12"),
            "China deval": ("2015-08-11", "2016-02-11"),
            "Q4 2018": ("2018-10-01", "2018-12-26")}

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "legend.frameon": False})

def shade(ax):
    for _, (s, e) in CRISIS.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#C44E52", alpha=0.10, zorder=0)

def main():
    os.makedirs('results/figures', exist_ok=True)
    se_tab = pd.read_csv('results/tables/41_xi_sampling_se.csv')
    se_at = dict(zip(se_tab['n_exceed'], se_tab['sd_xi']))
    nearest = lambda n: se_at[min(se_at, key=lambda x: abs(x - n))]

    # ---------------- 1. xi path, with the sampling-noise band ----------------
    fig, axes = plt.subplots(3, 2, figsize=(9.5, 7.5), sharex=True, constrained_layout=True)
    for i, c in enumerate(INDICES):
        a = axes.flat[i]
        d = pd.read_csv(f'results/tables/42_evt_diagnostics_{c}.csv', parse_dates=['Date'])
        band = np.array([nearest(n) for n in d['n_exceed']])
        a.fill_between(d['Date'], d['xi'] - band, d['xi'] + band,
                       color=PAL[i], alpha=0.15, lw=0, label='±1 sampling SE')
        a.plot(d['Date'], d['xi'], color=PAL[i], lw=1.3, label=r'$\hat\xi$')
        a.axhline(0, color='#999', lw=0.8, ls='--')
        a.set_title(f"{c}   range [{d['xi'].min():+.3f}, {d['xi'].max():+.3f}]"
                    f"   spread {d['xi'].max()-d['xi'].min():.3f}"
                    f"   SE ≈ {band.mean():.3f}", fontsize=8.5)
        a.set_ylabel(r'$\xi$', fontsize=8); a.tick_params(labelsize=7)
        if i == 0: a.legend(fontsize=7, loc='upper left')
    fig.suptitle('Expanding-window GPD shape parameter over the forecast period\n'
                 'movement within the shaded band is not distinguishable from '
                 'estimation noise', fontsize=10)
    fig.savefig('results/figures/43_xi_path.png'); plt.close(fig)

    # ---------------- 2. returns vs 1% VaR, breaches marked ----------------
    fig, axes = plt.subplots(3, 2, figsize=(9.5, 7.5), sharex=True, constrained_layout=True)
    for i, c in enumerate(INDICES):
        a = axes.flat[i]
        d = fio.read_forecasts(f'{FORECAST}/GARCH-EVT__{c}_forecasts.csv')
        d = d[d['Valid']]
        br = d['Realized'] < d['VaR_01']
        shade(a)
        a.plot(d['Date'], 100*d['Realized'], lw=0.4, color='#B8B8B8', label='return')
        a.plot(d['Date'], 100*d['VaR_01'], lw=1.1, color=PAL[i], label='99% VaR')
        a.scatter(d['Date'][br], 100*d['Realized'][br], s=13, color='#C44E52',
                  zorder=5, label=f'breach ({int(br.sum())})')
        a.set_title(f'{c}   {100*br.mean():.3f}% vs 1.00% target', fontsize=8.5)
        a.set_ylabel('%', fontsize=8); a.tick_params(labelsize=7)
        if i == 0: a.legend(fontsize=7, loc='lower left', ncol=3)
    fig.suptitle('GARCH-EVT 99% VaR and realised returns; shaded = named crisis windows',
                 fontsize=10)
    fig.savefig('results/figures/43_var_breaches.png'); plt.close(fig)

    # ---------------- 3. how much wider is the EVT tail than skew-t? ----------
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    rows = []
    for i, c in enumerate(INDICES):
        e = fio.read_forecasts(f'{FORECAST}/GARCH-EVT__{c}_forecasts.csv')
        s = fio.read_forecasts(f'{FORECAST}/GJR-skewt__{c}_forecasts.csv')
        m = e[e['Valid']].merge(s[s['Valid']][['Date', 'VaR_01']], on='Date',
                                suffixes=('_evt', '_skewt'))          # join on Date
        ratio = m['VaR_01_evt'] / m['VaR_01_skewt']
        ax.plot(m['Date'], ratio, lw=0.9, color=PAL[i], label=f'{c} (mean {ratio.mean():.3f})')
        rows.append({'index': c, 'mean_ratio': ratio.mean(), 'min': ratio.min(),
                     'max': ratio.max(), 'pct_days_wider': 100*(ratio > 1).mean()})
    ax.axhline(1.0, color='#444', ls='--', lw=1)
    ax.set_ylabel('EVT 99% VaR / skew-t 99% VaR')
    ax.set_title('Ratio of EVT to skew-t tail quantile. Above 1 = EVT is more conservative.\n'
                 'Both share the same conditional variance, so this is essentially '
                 'the tail specification.', fontsize=9)
    ax.legend(ncol=3, fontsize=8)
    fig.savefig('results/figures/43_evt_vs_skewt.png'); plt.close(fig)

    r = pd.DataFrame(rows)
    r.to_csv('results/tables/43_evt_vs_skewt_ratio.csv', index=False)
    print(r.to_string(index=False))
    print('\nwrote 3 figures and 1 table')

if __name__ == '__main__':
    main()
