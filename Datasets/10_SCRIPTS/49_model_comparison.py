# -*- coding: utf-8 -*-
"""
49 - Statistical model comparison and summary figures. Executive Summary 4.2/4.3.

WHAT THIS ANSWERS THAT SCRIPT 47 DOES NOT
  47 asks whether each model is individually well calibrated. 49 asks whether the
  models differ from EACH OTHER by more than sampling noise, on a proper scoring
  rule rather than on a pass/fail test.

  Pinball (quantile) loss is the proper scoring rule for a VaR forecast: it is
  minimised in expectation by the true quantile, so it cannot be gamed by simply
  widening the interval. Diebold-Mariano and the Model Confidence Set both run
  on it.

  Expect the two halves of this project to disagree. Coverage tests ask whether
  the quantile sits in the right PLACE; pinball loss asks what the forecast COSTS
  on average. A model with a too-tight VaR recovers on the ~99% of non-breach days
  what it loses on breaches. That is not a contradiction - it is precisely why
  regulators mandate coverage tests rather than average loss.

  NOTE ON n: the MCS needs one loss matrix with no gaps, so it runs on the days
  where all five models forecast. That is QR-Full's window, hence n of 1,814 to
  2,718 rather than the full sample. The DM tests are pairwise and use each pair's
  own intersection, so they keep more data.

FIGURES
  49_scorecard.png       5 models x 6 indices x 4 tests as a pass/fail grid
  49_qlike_vs_breach.png the headline: volatility accuracy against tail accuracy
  49_basel.png           Basel traffic-light zones, named in Executive Summary 4.2
  49_loss_metrics.png    why the choice of volatility loss function matters

OUTPUT
  results/tables/49_dm_pinball.csv  49_mcs.csv  49_basel.csv
"""
import os, importlib.util, itertools, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
bc  = _load('bc',  os.path.join(HERE, '40_b_common.py'))
fio = _load('fio', os.path.join(HERE, '26_forecast_io.py'))

FORECAST = 'Datasets/20_FORECASTS'
INDICES  = ['SPX', 'NDX', 'UKX', 'DAX', 'NKY', 'HSI']
MODELS   = ['GARCH-EVT', 'GJR-skewt', 'RealGARCH', 'QR-Full', 'QR-Range']
VOL_MODELS = ['GARCH-EVT', 'GJR-skewt', 'RealGARCH']
# 2026-08-29: the three variance-forecasting models, intersected per index to give the
# strict common window. Matches STRICT_MODELS in 47_evaluation.py and 48_crisis_regime.py.
STRICT_MODELS = ['GARCH-EVT', 'GJR-skewt', 'RealGARCH']
ALPHA = 0.01
PAL = ["#378ADD", "#D85A30", "#1D9E75", "#BA7517", "#7F77DD"]

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def load_all():
    """Sample-B window, same gate as scripts 47 and 48."""
    d = {}
    for m in MODELS:
        for c in INDICES:
            f = fio.read_forecasts(f'{FORECAST}/{m}__{c}_forecasts.csv')
            d[(m, c)] = f[f['Valid'].astype(bool)].set_index('Date').sort_index()
    for c in INDICES:
        ref = d[('GJR-skewt', c)].index
        for m in MODELS:
            d[(m, c)] = d[(m, c)].loc[d[(m, c)].index.intersection(ref)]
    return d


def strict_window(data):
    """Per-index intersection of the three variance models' valid dates. 2026-08-29.

    WHY BASEL NEEDS THIS
      basel_zone() classifies on n_breach GIVEN n_obs. After the walk-forward Realized
      GARCH rebuild, n_obs differs by model within an index - DAX 2,765 for RealGARCH
      against 3,267 for GJR-skewt - because the refit needs a burn-in. Comparing zones
      across a row then compares models on different sample sizes, and the zone table
      is the headline cross-model claim in this project. The same applies to the pooled
      mean pinball ranking, where each model would otherwise be averaged over its own
      set of days.

      DM and the MCS are already immune: DM intersects each pair itself, and the MCS
      needs one gap-free matrix so it already intersects all five.

      QR is excluded from the intersection for the reason 47 gives - QR-Full's 13
      predictors fail on 23-42% of days - but is carried through, cut to the strict
      index, and still reported with its own n.

    NON-DESTRUCTIVE: returns a new dict.
    """
    out, rep = {}, []
    for c in INDICES:
        idx = None
        for m in STRICT_MODELS:
            i = data[(m, c)].index
            idx = i if idx is None else idx.intersection(i)
        for m in MODELS:
            before = len(data[(m, c)])
            out[(m, c)] = data[(m, c)].loc[data[(m, c)].index.intersection(idx)]
            rep.append({'index': c, 'model': m, 'rows_unrestricted': before,
                        'rows_strict': len(out[(m, c)]),
                        'dropped_for_common_window': before - len(out[(m, c)])})
    return out, pd.DataFrame(rep)


def basel_zone(n_breach, n_obs, alpha=ALPHA):
    """Basel traffic light, generalised from the 250-day table to any n.

    The published zones are cumulative-binomial cut-points: green while the
    probability of seeing this many or fewer breaches under a correct model is
    below 95%, amber to 99.99%, red beyond. Stating it this way rather than
    hard-coding 4 and 9 keeps it valid at n = 3,243.
    """
    p = stats.binom.cdf(n_breach, n_obs, alpha)       # P(X <= n_breach)
    if p < 0.95:   return 'green'
    if p < 0.9999: return 'amber'
    return 'red'


def main():
    os.makedirs('results/tables', exist_ok=True); os.makedirs('results/figures', exist_ok=True)
    data = load_all()
    sdata, swin = strict_window(data)
    swin.to_csv('results/tables/49_strict_window.csv', index=False)

    loss = {}
    sloss = {}
    for c in INDICES:
        for m in MODELS:
            d = data[(m, c)]
            loss[(m, c)] = pd.Series(
                bc.pinball_loss(d['Realized'], d['VaR_01'], ALPHA), index=d.index)
            sd = sdata[(m, c)]
            sloss[(m, c)] = pd.Series(
                bc.pinball_loss(sd['Realized'], sd['VaR_01'], ALPHA), index=sd.index)

    dm_rows = []
    for c in INDICES:
        for m1, m2 in itertools.combinations(MODELS, 2):
            i = loss[(m1, c)].index.intersection(loss[(m2, c)].index)
            r = bc.diebold_mariano(loss[(m1, c)].loc[i], loss[(m2, c)].loc[i])
            dm_rows.append({'index': c, 'model_1': m1, 'model_2': m2, 'n': r['n'],
                            'dm_stat': r['stat'], 'p': r['p'],
                            'sig_5pct': bool(np.isfinite(r['p']) and r['p'] < 0.05)})
    dm = pd.DataFrame(dm_rows); dm.to_csv('results/tables/49_dm_pinball.csv', index=False)

    mcs_rows = []
    for c in INDICES:
        i = None
        for m in MODELS:
            i = loss[(m, c)].index if i is None else i.intersection(loss[(m, c)].index)
        r = bc.model_confidence_set({m: loss[(m, c)].loc[i].values for m in MODELS},
                                    alpha=0.10)
        mcs_rows.append({'index': c, 'n': r['n'], 'n_retained': len(r['mcs']),
                         'retained': '|'.join(r['mcs']),
                         'eliminated': '|'.join(r['eliminated']) or 'none'})
    mcs = pd.DataFrame(mcs_rows); mcs.to_csv('results/tables/49_mcs.csv', index=False)

    def basel_table(src):
        rows = []
        for c in INDICES:
            for m in MODELS:
                d = src[(m, c)]
                nb = int((d['Realized'] < d['VaR_01']).sum())
                rows.append({'index': c, 'model': m, 'n_obs': len(d), 'n_breach': nb,
                             'expected': round(ALPHA*len(d), 1),
                             'rate_pct': round(100*nb/len(d), 3),
                             'zone': basel_zone(nb, len(d))})
        return pd.DataFrame(rows)

    basel_unres = basel_table(data)
    basel_unres.to_csv('results/tables/49_basel_unrestricted.csv', index=False)
    # The strict table is the headline: zones are only comparable across a row when
    # every model was classified on the same n_obs.
    basel = basel_table(sdata)
    basel.to_csv('results/tables/49_basel.csv', index=False)

    # 47b_var_backtests_strict.csv is written by 47_evaluation.py on the same
    # strict window used here; fall back if 47 has not been re-run yet.
    _bt_strict = 'results/tables/47b_var_backtests_strict.csv'
    bt = pd.read_csv(_bt_strict if os.path.exists(_bt_strict)
                     else 'results/tables/47b_var_backtests.csv')
    bt = bt[bt['confidence'] == 0.99]
    TESTS = [('pass_kupiec', 'Kupiec'), ('pass_indep', 'Indep'),
             ('pass_cc', 'Joint CC'), ('pass_dq', 'DQ')]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    for k, (col, lab) in enumerate(TESTS):
        ax = axes[k]
        g = bt.pivot(index='index', columns='model', values=col)[MODELS].loc[INDICES]
        ax.imshow(g.astype(float).values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(range(len(INDICES))); ax.set_yticklabels(INDICES, fontsize=8)
        for i in range(len(INDICES)):
            for j in range(len(MODELS)):
                ax.text(j, i, 'P' if g.values[i, j] else 'F', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='#222')
        ax.set_title(f'{lab}   {int(g.sum().sum())}/30', fontsize=9)
    fig.suptitle('99% VaR test outcomes. P = pass at 5%. The Dynamic Quantile test, '
                 'which looks four days back, rejects far more than the others.',
                 fontsize=9.5)
    fig.savefig('results/figures/49_scorecard.png'); plt.close(fig)

    # Must be the strict window, like `bt` above: the breach rate on the y-axis of
    # 49_qlike_vs_breach is strict, so a QLIKE on the x-axis measured over a longer
    # sample would put the two axes on different samples for 8 of the 18 cells.
    _vol_strict = 'results/tables/47a_volatility_losses_strict.csv'
    vol = pd.read_csv(_vol_strict if os.path.exists(_vol_strict)
                      else 'results/tables/47a_volatility_losses.csv')
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    for j, m in enumerate(VOL_MODELS):
        q = [vol[(vol['index'] == c) & (vol['model'] == m)]['QLIKE'].iloc[0] for c in INDICES]
        b = [bt[(bt['index'] == c) & (bt['model'] == m)]['rate_pct'].iloc[0] for c in INDICES]
        ax.scatter(q, b, s=60, color=PAL[j], label=m, zorder=3)
        for x, y, c in zip(q, b, INDICES):
            ax.annotate(c, (x, y), fontsize=6.5, xytext=(4, 3),
                        textcoords='offset points', color='#555')
    ax.axhline(1.0, color='#333', ls='--', lw=1)
    ax.set_xlabel('QLIKE  (volatility accuracy, lower better)')
    ax.set_ylabel('99% VaR breach rate %  (tail accuracy, 1.00 is correct)')
    ax.set_title('The central result: the two axes rank the models in OPPOSITE orders.\n'
                 'RealGARCH forecasts variance best on all six and breaches most on five.',
                 fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, lw=0.5)
    fig.savefig('results/figures/49_qlike_vs_breach.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    cmap = {'green': '#4C9F70', 'amber': '#E0A458', 'red': '#C44E52'}
    g = basel.pivot(index='index', columns='model', values='zone')[MODELS].loc[INDICES]
    nb = basel.pivot(index='index', columns='model', values='n_breach')[MODELS].loc[INDICES]
    ax.imshow(np.zeros(g.shape), cmap='Greys', vmin=0, vmax=1, aspect='auto')
    for i in range(len(INDICES)):
        for j in range(len(MODELS)):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       color=cmap[g.values[i, j]], zorder=2))
            ax.text(j, i, f'{nb.values[i, j]}', ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white', zorder=3)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, fontsize=7.5, rotation=20, ha='right')
    ax.set_yticks(range(len(INDICES))); ax.set_yticklabels(INDICES, fontsize=8)
    ax.set_title('Basel traffic light, 99% VaR. Cell shows breach count.\n'
                 'Zones are cumulative-binomial cut-points generalised from the '
                 '250-day table.', fontsize=9)
    fig.savefig('results/figures/49_basel.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    mets = ['QLIKE', 'RMSE', 'MAE', 'MAPE']
    x = np.arange(len(INDICES)); w = 0.2
    red = {}
    for k, met in enumerate(mets):
        imp = []
        for c in INDICES:
            gg = vol[(vol['index'] == c) & (vol['model'] == 'GJR-skewt')][met].iloc[0]
            rr = vol[(vol['index'] == c) & (vol['model'] == 'RealGARCH')][met].iloc[0]
            imp.append(100*(gg-rr)/gg)
        red[met] = dict(zip(INDICES, imp))
        ax.bar(x + (k-1.5)*w, imp, w, label=met, color=PAL[k], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(INDICES)
    ax.set_ylabel('% reduction in loss, RealGARCH vs GJR-skew-t')
    # Pick the exhibit index from the data rather than hard-coding it: the widest
    # QLIKE-minus-RMSE gap among indices where the two metrics disagree on sign.
    _split = [c for c in INDICES
              if red['QLIKE'][c] > 0 > red['RMSE'][c]] or list(INDICES)
    _c = max(_split, key=lambda c: red['QLIKE'][c] - red['RMSE'][c])
    _dmtxt = ''
    _dmf = 'results/tables/47a_dm_volatility.csv'
    if os.path.exists(_dmf):
        _dm = pd.read_csv(_dmf)
        _r = _dm[(_dm['index'] == _c) & (_dm.model_1 == 'GJR-skewt')
                 & (_dm.model_2 == 'RealGARCH')]
        if len(_r) and np.isfinite(_r.iloc[0]['p']):
            _p = float(_r.iloc[0]['p'])
            _dmtxt = (f"  (DM {float(_r.iloc[0]['dm_stat']):.2f}, "
                      + (f"p<0.0001)" if _p < 1e-4 else f"p={_p:.4f})"))
    _rm = red['RMSE'][_c]
    ax.set_title('The volatility loss function is not a neutral choice.\n'
                 f'On {_c}, RMSE says RealGARCH is {abs(_rm):.1f}% '
                 f'{"worse" if _rm < 0 else "better"}; '
                 f'QLIKE says {red["QLIKE"][_c]:.1f}% better.{_dmtxt}', fontsize=9)
    ax.legend(fontsize=8, ncol=4); ax.grid(alpha=0.25, lw=0.5, axis='y')
    fig.savefig('results/figures/49_loss_metrics.png'); plt.close(fig)

    print('=== Model Confidence Set, 90%, pinball loss at 1% ===')
    print(mcs.to_string(index=False))
    print('\n=== Diebold-Mariano on pinball loss ===')
    sig = dm[dm['sig_5pct']]
    print(f'significant at 5%: {len(sig)} of {len(dm)} pairwise tests')
    print(f'expected by chance alone at 5%: {0.05*len(dm):.1f}')
    if len(sig):
        print(sig[['index', 'model_1', 'model_2', 'dm_stat', 'p']].round(4).to_string(index=False))
    print('\nmean pinball loss by model (pooled over indices, STRICT common window):')
    mp = {m: np.mean(np.concatenate([sloss[(m, c)].values for c in INDICES])) for m in MODELS}
    for m, v in sorted(mp.items(), key=lambda kv: kv[1]):
        print(f'  {m:10} {v:.6e}')
    print(f'  spread best to worst: {100*(max(mp.values())/min(mp.values())-1):.1f}%')
    mpu = {m: np.mean(np.concatenate([loss[(m, c)].values for c in INDICES])) for m in MODELS}
    print('  (unrestricted, each model on its own days: '
          + ', '.join(f'{m} {mpu[m]:.4e}' for m in sorted(mpu, key=mpu.get)) + ')')

    print('\n=== Basel traffic light, 99% VaR, STRICT common window ===')
    print(basel.pivot(index='index', columns='model', values='zone')[MODELS]
          .loc[INDICES].to_string())
    print('n_obs behind each zone:')
    print(basel.pivot(index='index', columns='model', values='n_obs')[MODELS]
          .loc[INDICES].to_string())
    print('\nzone counts (STRICT):')
    for m in MODELS:
        z = basel[basel['model'] == m]['zone'].value_counts()
        print(f"  {m:10} green {z.get('green',0)}  amber {z.get('amber',0)}  red {z.get('red',0)}")

    print('\nzone counts (unrestricted, for comparison only):')
    for m in MODELS:
        z = basel_unres[basel_unres['model'] == m]['zone'].value_counts()
        print(f"  {m:10} green {z.get('green',0)}  amber {z.get('amber',0)}  red {z.get('red',0)}")
    chg = basel.merge(basel_unres, on=['index', 'model'], suffixes=('_strict', '_unres'))
    chg = chg[chg['zone_strict'] != chg['zone_unres']]
    if len(chg):
        print('\ncells whose ZONE changed under the common window:')
        print(chg[['index', 'model', 'n_obs_unres', 'zone_unres',
                   'n_obs_strict', 'zone_strict']].to_string(index=False))
    else:
        print('\nno zone changed under the common window')
    print('\nwrote 6 tables and 4 figures')

if __name__ == '__main__':
    main()
