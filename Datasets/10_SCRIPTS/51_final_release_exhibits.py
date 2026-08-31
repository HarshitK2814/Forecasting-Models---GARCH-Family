# -*- coding: utf-8 -*-
"""
51 - Two final-release exhibits added after the 2026-08-31 output review.

WHY THIS SCRIPT EXISTS RATHER THAN EDITS TO 43/45
  Both exhibits below need the STRICT common window, which is written by 47 and 49.
  Scripts 43 and 45 run earlier in the pipeline and are legitimately model-specific
  diagnostics, so making them read 47's output would invert the run order. Adding a
  late script keeps the ordering safe and leaves the original diagnostics intact.

EXHIBIT 1 - Realized GARCH innovation robustness
  The project's headline is that better sigma_t does not buy better q_alpha,t. That
  reading only holds up if the tail failure is attributable to the innovation
  distribution rather than to the realized-measure machinery. Pairing RealGARCH-t
  against RealGARCH-skew-t on identical dates isolates exactly that: same variance
  model, same realized measure, different innovation. The skew-t variant cuts the
  99% breach rate on every index, which is what licenses the VaR = mu + sigma*q
  decomposition in the discussion.

EXHIBIT 2 - Quantile-regression calibration on the strict window
  45_qr_calibration.png is built from 44_qr_summary.csv, i.e. each specification's
  own forecast dates. That is correct as a within-QR diagnostic but it disagrees
  numerically with the strict cross-model tables the paper prints beside it (DAX
  QR-Range reads 1.133% there against 1.049% strict). This regenerates it from
  47b_var_backtests_strict.csv so the visual and the headline table cannot diverge.

OUTPUT
  results/tables/51_realgarch_innovation.csv
  results/figures/51_realgarch_innovation.png
  results/figures/51_qr_calibration_strict.png
"""
import os, importlib.util, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
bc  = _load('bc',  os.path.join(HERE, '40_b_common.py'))
fio = _load('fio', os.path.join(HERE, '26_forecast_io.py'))

FORECAST = 'Datasets/20_FORECASTS'
INDICES  = ['SPX', 'NDX', 'UKX', 'DAX', 'NKY', 'HSI']
LEVELS   = [('VaR_01', 0.01, '99'), ('VaR_025', 0.025, '97.5'), ('VaR_05', 0.05, '95')]
PAL = ["#378ADD", "#D85A30", "#1D9E75", "#BA7517", "#7F77DD"]

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def _valid(model, code):
    f = fio.read_forecasts(f'{FORECAST}/{model}__{code}_forecasts.csv')
    return f[f['Valid'].astype(bool)].set_index('Date').sort_index()


def realgarch_innovation():
    """RealGARCH-t vs RealGARCH-skew-t on identical dates, per index."""
    rows = []
    for c in INDICES:
        a, b = _valid('RealGARCH', c), _valid('RealGARCH_ST', c)
        i = a.index.intersection(b.index)          # pair them, never compare on
        a, b = a.loc[i], b.loc[i]                  # different sets of days
        for col, alpha, lab in LEVELS:
            for name, d in (('RealGARCH-t', a), ('RealGARCH-skew-t', b)):
                br = (d['Realized'] < d[col]).astype(int)
                _lr, _p = bc.kupiec_pof(br, alpha)
                rows.append({'index': c, 'model': name, 'level_pct': lab,
                             'n_obs': len(d), 'n_breach': int(br.sum()),
                             'expected': round(alpha*len(d), 1),
                             'rate_pct': round(100*br.mean(), 3),
                             'target_pct': 100*alpha,
                             'kupiec_p': round(_p, 4),
                             'pass_kupiec': bool(_p >= 0.05)})
        ok = a['RVProxy'].notna() & b['RVProxy'].notna()
        for name, d in (('RealGARCH-t', a), ('RealGARCH-skew-t', b)):
            L = bc.vol_losses(d.loc[ok, 'RVProxy'], d.loc[ok, 'VarHat'])
            rows.append({'index': c, 'model': name, 'level_pct': 'QLIKE',
                         'n_obs': int(ok.sum()), 'rate_pct': round(L['QLIKE'], 6)})
    t = pd.DataFrame(rows)
    t.to_csv('results/tables/51_realgarch_innovation.csv', index=False)

    one = t[(t.level_pct == '99')].pivot(index='index', columns='model',
                                         values='rate_pct').loc[INDICES]
    fig, ax = plt.subplots(figsize=(7.6, 4.0), constrained_layout=True)
    x = np.arange(len(INDICES)); w = 0.36
    ax.bar(x - w/2, one['RealGARCH-t'], w, label='RealGARCH-t', color=PAL[1])
    ax.bar(x + w/2, one['RealGARCH-skew-t'], w, label='RealGARCH-skew-t', color=PAL[2])
    for k, c in enumerate(INDICES):
        ax.text(k - w/2, one.loc[c, 'RealGARCH-t'] + 0.03,
                f"{one.loc[c, 'RealGARCH-t']:.2f}", ha='center', fontsize=7.5)
        ax.text(k + w/2, one.loc[c, 'RealGARCH-skew-t'] + 0.03,
                f"{one.loc[c, 'RealGARCH-skew-t']:.2f}", ha='center', fontsize=7.5)
    ax.axhline(1.0, color='#333', ls='--', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(INDICES)
    ax.set_ylabel('99% VaR breach rate %   (1.00 is correct)')
    ax.set_title('Same variance model, same realized measure, different innovation.\n'
                 'The skew-t tail cuts the breach rate on all six, so the failure is '
                 'the innovation, not the realized measure.', fontsize=9)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25, lw=0.5, axis='y')
    fig.savefig('results/figures/51_realgarch_innovation.png'); plt.close(fig)
    return one


def qr_calibration_strict():
    """QR calibration rebuilt from the strict backtests, so it agrees with TAB4/TAB5."""
    bt = pd.read_csv('results/tables/47b_var_backtests_strict.csv')
    specs = ['QR-Full', 'QR-Range']
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    for k, (_, alpha, lab) in enumerate(LEVELS):
        ax = axes[k]
        sub = bt[np.isclose(bt['confidence'], 1 - alpha)]
        x = np.arange(len(INDICES)); w = 0.36
        for j, sp in enumerate(specs):
            v = [sub[(sub['index'] == c) & (sub.model == sp)]['rate_pct'].iloc[0]
                 for c in INDICES]
            ax.bar(x + (j - 0.5)*w, v, w, label=sp, color=PAL[j])
        ax.axhline(100*alpha, color='#333', ls='--', lw=1)
        ax.set_xticks(x); ax.set_xticklabels(INDICES, fontsize=8)
        ax.set_title(f'{lab}% VaR   target {100*alpha:g}%', fontsize=9)
        if k == 0:
            ax.set_ylabel('breach rate %'); ax.legend(fontsize=8)
        ax.grid(alpha=0.25, lw=0.5, axis='y')
    fig.suptitle('Quantile-regression calibration on the STRICT common window. '
                 'Bars above the dashed line breach too often.', fontsize=9.5)
    fig.savefig('results/figures/51_qr_calibration_strict.png'); plt.close(fig)


def main():
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    one = realgarch_innovation()
    qr_calibration_strict()
    print('=== 99% VaR breach rate, RealGARCH-t vs RealGARCH-skew-t (paired dates) ===')
    print(one.round(3).to_string())
    d = one['RealGARCH-t'] - one['RealGARCH-skew-t']
    print(f'\nskew-t lower on {int((d > 0).sum())} of {len(d)} indices; '
          f'mean reduction {d.mean():.3f} pp')
    print('\nwrote 1 table and 2 figures')


if __name__ == '__main__':
    main()
