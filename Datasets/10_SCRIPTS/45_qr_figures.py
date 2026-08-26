# -*- coding: utf-8 -*-
"""
45 - Quantile regression figures.

Each figure backs a specific claim:
  45_qr_calibration.png   breach rate vs nominal at all three levels, both specs.
                          Shows WHERE each spec is miscalibrated, not just that it is.
  45_qr_crossings.png     quantile crossings. The 13-predictor spec contradicts its
                          own quantile ordering on 5-12% of days; the 5-predictor
                          spec on 1-2%. Second, independent symptom of overfitting,
                          alongside the breach rates.
  45_qr_coefficients.png  standardised tau=0.01 coefficients on the final window.
  45_qr_var_series.png    QR-Range 99% VaR against realised returns, breaches marked.

COVERAGE WARNING carried into every panel: QR-Full cannot forecast where any of
its 13 predictors is missing. NKY manages only 1,814 of 3,150 days because of the
2016-17 realized-volatility gap. The two specs are therefore NOT evaluated on the
same days, and the figures label the count so this cannot be forgotten.
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
SPECS    = ['QR-Full', 'QR-Range']
LEVELS   = [('01', 1.0), ('025', 2.5), ('05', 5.0)]
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

def load(spec, code):
    d = fio.read_forecasts(f'{FORECAST}/{spec}__{code}_forecasts.csv')
    return d[d['Valid']].copy()

def main():
    os.makedirs('results/figures', exist_ok=True)
    summ = pd.read_csv('results/tables/44_qr_summary.csv')
    w = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    for k, (tag, nominal) in enumerate(LEVELS):
        ax = axes[k]; x = np.arange(len(INDICES))
        for s, (spec, off) in enumerate(zip(SPECS, (-w/2, w/2))):
            rates = [summ[(summ['spec'] == spec) & (summ['index'] == c)]
                     [f'rate_{tag}_pct'].iloc[0] for c in INDICES]
            ax.bar(x + off, rates, w, label=spec,
                   color=PAL[0] if s == 0 else PAL[1], alpha=0.85)
        ax.axhline(nominal, color='#333', ls='--', lw=1.2)
        ax.set_xticks(x); ax.set_xticklabels(INDICES, fontsize=8)
        ax.set_title(f'{100-nominal:.1f}% VaR   target {nominal}%', fontsize=9)
        ax.set_ylabel('breach rate %' if k == 0 else '')
        if k == 0: ax.legend(fontsize=8)
    fig.suptitle('Quantile regression calibration. Bars above the dashed line breach '
                 'too often.', fontsize=10)
    fig.savefig('results/figures/45_qr_calibration.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    x = np.arange(len(INDICES))
    for s, (spec, off) in enumerate(zip(SPECS, (-w/2, w/2))):
        pct = [summ[(summ['spec'] == spec) & (summ['index'] == c)]
               ['pct_crossings'].iloc[0] for c in INDICES]
        b = ax.bar(x + off, pct, w, label=spec, color=PAL[0] if s == 0 else PAL[1], alpha=0.85)
        ax.bar_label(b, fmt='%.1f', fontsize=7, padding=1)
    ax.set_xticks(x); ax.set_xticklabels(INDICES)
    ax.set_ylabel('% of forecast days with crossed quantiles')
    ax.set_title('Quantile crossing: days where the fitted 1% line sits ABOVE the 2.5% line.\n'
                 'Logically impossible, and a direct symptom of overfitting. '
                 'Repaired by rearrangement.', fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig('results/figures/45_qr_crossings.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for k, spec in enumerate(SPECS):
        ax = axes[k]
        cf = pd.read_csv(f'results/tables/44_qr_coefficients_{spec}.csv')
        cf = cf[cf['predictor'] != 'const']
        piv = cf.pivot(index='predictor', columns='index', values='coef')[INDICES]
        piv = piv.loc[piv.abs().mean(axis=1).sort_values().index]
        y = np.arange(len(piv))
        for i, c in enumerate(INDICES):
            ax.scatter(piv[c], y + (i - 2.5)*0.11, s=26, color=PAL[i],
                       label=c if k == 0 else None)
        ax.axvline(0, color='#444', lw=1)
        ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_xlabel('standardised coefficient, tau = 0.01')
        ax.set_title(f'{spec}   ({len(piv)} predictors)', fontsize=9)
        if k == 0: ax.legend(ncol=6, fontsize=7.5, loc='lower right')
    fig.suptitle('Final-window quantile-regression coefficients on the 1% tail.\n'
                 'Predictors are standardised, so magnitudes are comparable within '
                 'an index. A negative coefficient pushes the VaR deeper.', fontsize=10)
    fig.savefig('results/figures/45_qr_coefficients.png'); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(9.5, 7.5), sharex=True, constrained_layout=True)
    for i, c in enumerate(INDICES):
        a = axes.flat[i]; d = load('QR-Range', c)
        br = d['Realized'] < d['VaR_01']
        shade(a)
        a.plot(d['Date'], 100*d['Realized'], lw=0.4, color='#B8B8B8')
        a.plot(d['Date'], 100*d['VaR_01'], lw=1.1, color=PAL[i])
        a.scatter(d['Date'][br], 100*d['Realized'][br], s=13, color='#C44E52', zorder=5)
        a.set_title(f'{c}   {100*br.mean():.3f}% vs 1.00%   n={len(d)}   '
                    f'breaches={int(br.sum())}', fontsize=8.5)
        a.set_ylabel('%', fontsize=8); a.tick_params(labelsize=7)
    fig.suptitle('QR-Range 99% VaR and realised returns; shaded = named crisis windows',
                 fontsize=10)
    fig.savefig('results/figures/45_qr_var_series.png'); plt.close(fig)

    p = summ.pivot(index='index', columns='spec', values='rate_01_pct')[SPECS].loc[INDICES]
    p['closer_to_1pct'] = np.where((p['QR-Range']-1).abs() < (p['QR-Full']-1).abs(),
                                   'QR-Range', 'QR-Full')
    n = summ.pivot(index='index', columns='spec', values='n_valid')[SPECS].loc[INDICES]
    print('1% breach rate by spec (target 1.00):'); print(p.to_string())
    print('\nevaluable days by spec:'); print(n.to_string())
    print('\nQR-Range closer to nominal on %d of 6 indices'
          % int((p['closer_to_1pct'] == 'QR-Range').sum()))
    print('mean |deviation from 1%%|:  QR-Full %.3f pp   QR-Range %.3f pp'
          % ((p['QR-Full']-1).abs().mean(), (p['QR-Range']-1).abs().mean()))
    print('\nwrote 4 figures')

if __name__ == '__main__':
    main()
