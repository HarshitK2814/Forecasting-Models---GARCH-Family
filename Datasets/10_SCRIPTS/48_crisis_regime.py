# -*- coding: utf-8 -*-
"""
48 - Crisis and regime analysis. Executive Summary section 4.3.

TWO REGIME DEFINITIONS, ASKING DIFFERENT QUESTIONS
  CrisisLabel        ten named windows on fixed calendar dates, identical across
                     all six indices. Captures SUDDEN ONSET.
  VolRegime_ExAnte   quartiles of trailing volatility from EXPANDING-WINDOW
                     quantiles using only data strictly before each date.
                     Captures SUSTAINED elevation. No look-ahead.
  VolRegime (ex-post, full-sample cut-points) is NEVER used - Dataset Guide
  dictionary: "correct for ex-post subsample REPORTING, NEVER as a predictor".
  Both are read through fio.eval_frame(), which is what A built it for.

COMMON EVALUATION WINDOW
  A's RealGARCH files start 2011-09 to 2012-02, not 2013-09-30, violating contract
  rule 6. Unfixed, RealGARCH is scored on up to 511 days no other model sees, and
  it alone picks up a Euro_Sovereign_Debt regime that ends before sample B begins.
  Every model is cut to A's contract-valid GJR-skewt calendar. This is NOT a
  balanced panel: coverage gaps that arise from a model legitimately being unable
  to forecast are left alone, per Dataset_Guide precaution 7.

POOLING - RESEARCHER_A_DECISIONS section 1
  A permits pooling of LOSS SERIES across indices but forbids putting two indices'
  returns on the same row and calling it the same day. Breach indicators are
  outcomes, not aligned returns, and CrisisLabel uses identical calendar dates on
  all six, so pooling breach counts by regime is legitimate.
  BUT pooled counts weight each index by its valid-day count, and those differ -
  QR-Full has 1,814 valid NKY days against 3,243 for QR-Range on SPX. The
  per-index table is written alongside and the pooled figure is never reported
  without it.

THE TRAP THIS SCRIPT IS BUILT TO AVOID
  A degradation RATIO (crisis rate / normal rate) rewards a model for having a
  HIGHER normal-regime denominator. Two models with identical crisis behaviour can
  show different ratios purely because one is worse in calm markets. The script
  reports crisis rates in LEVELS as primary and the ratio as secondary, and prints
  both so the difference cannot be quietly dropped.

OUTPUT
  results/tables/48_crisis_by_index.csv     48_crisis_pooled.csv
  results/tables/48_volregime_pooled.csv    48_degradation.csv
  results/figures/48_crisis_heatmap.png     48_degradation.png
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
MODELS   = ['GARCH-EVT', 'GJR-skewt', 'RealGARCH', 'QR-Full', 'QR-Range']
ALPHA    = 0.01
MIN_DAYS = 40          # below this a "rate" rests on 0-3 breaches and is not a rate
PAL      = ["#378ADD", "#D85A30", "#1D9E75", "#BA7517", "#7F77DD"]

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def frames():
    out = {}
    for m in MODELS:
        for c in INDICES:
            fc = fio.read_forecasts(f'{FORECAST}/{m}__{c}_forecasts.csv')
            e = fio.eval_frame(fc, c)              # brings CrisisLabel + VolRegime_ExAnte
            e = e[e['VaR_01'].notna()].copy()
            e['breach'] = (e['Realized'] < e['VaR_01']).astype(int)
            out[(m, c)] = e.set_index('Date').sort_index()
    # Restrict to sample B using A's contract-compliant GJR-skewt calendar.
    for c in INDICES:
        ref = out[('GJR-skewt', c)].index
        for m in MODELS:
            out[(m, c)] = out[(m, c)].loc[out[(m, c)].index.intersection(ref)]
    print('sample-B days per model x index:')
    print(pd.DataFrame({m: {c: len(out[(m, c)]) for c in INDICES}
                        for m in MODELS})[MODELS].to_string())
    return out


def by_regime(data, col):
    per, pooled = [], []
    for m in MODELS:
        acc = []
        for c in INDICES:
            e = data[(m, c)]
            for lab, g in e.groupby(e[col].fillna('Unlabelled')):
                per.append({'model': m, 'index': c, 'regime': lab, 'n': len(g),
                            'breaches': int(g['breach'].sum()),
                            'rate_pct': round(100 * g['breach'].mean(), 3)})
            acc.append(e[[col, 'breach']])
        a = pd.concat(acc)
        for lab, g in a.groupby(a[col].fillna('Unlabelled')):
            pooled.append({'model': m, 'regime': lab, 'n': len(g),
                           'breaches': int(g['breach'].sum()),
                           'rate_pct': round(100 * g['breach'].mean(), 3),
                           'reportable': len(g) >= MIN_DAYS})
    return pd.DataFrame(per), pd.DataFrame(pooled)


def main():
    os.makedirs('results/tables', exist_ok=True); os.makedirs('results/figures', exist_ok=True)
    data = frames()

    per_c, pool_c = by_regime(data, 'CrisisLabel')
    per_v, pool_v = by_regime(data, 'VolRegime_ExAnte')
    per_c.to_csv('results/tables/48_crisis_by_index.csv', index=False)
    pool_c.to_csv('results/tables/48_crisis_pooled.csv', index=False)
    pool_v.to_csv('results/tables/48_volregime_pooled.csv', index=False)

    deg = []
    for m in MODELS:
        s = pool_c[pool_c['model'] == m].set_index('regime')
        if 'Normal' not in s.index:
            continue
        norm = s.loc['Normal', 'rate_pct']
        cr = s[(s.index != 'Normal') & s['reportable']]
        w = (cr['rate_pct'] * cr['n']).sum() / cr['n'].sum()
        deg.append({'model': m, 'normal_rate_pct': norm, 'crisis_rate_pct': round(w, 3),
                    'ratio': round(w / norm, 3), 'crisis_days': int(cr['n'].sum())})
    deg = pd.DataFrame(deg); deg.to_csv('results/tables/48_degradation.csv', index=False)

    def show(p, title, order=None):
        t = p[p['reportable']].pivot(index='regime', columns='model', values='rate_pct')
        t = t.reindex(columns=MODELS)
        if order: t = t.reindex([o for o in order if o in t.index])
        n = p[p['reportable']].pivot(index='regime', columns='model', values='n').reindex(t.index)
        print(f'\n=== {title} (target 1.00%) ===')
        print(t.round(2).to_string())
        print('pooled days per regime:', dict(zip(t.index, n['GARCH-EVT'].astype('Int64'))))

    show(pool_c, 'pooled 99% VaR breach rate by named crisis window',
         ['Normal', 'China_Deval_Oil', 'Q4_2018_Selloff', 'COVID_Crash', 'Rate_Shock_2022'])
    show(pool_v, 'pooled 99% VaR breach rate by ex-ante volatility quartile',
         ['Calm', 'Normal', 'Stressed', 'Crisis'])

    dropped = pool_c[~pool_c['reportable']]['regime'].unique()
    print(f'\nwindows too short to report as rates (<{MIN_DAYS} pooled days): '
          f'{sorted(dropped)}')

    print('\n=== degradation: LEVELS first, ratio second ===')
    print(deg.to_string(index=False))
    print('\nA lower ratio can mean a model is WORSE in calm markets, not better in')
    print('crises. Read the crisis_rate_pct column, not the ratio, as primary.')

    t = pool_c[pool_c['reportable']].pivot(index='regime', columns='model', values='rate_pct')
    order = [o for o in ['Normal', 'China_Deval_Oil', 'Q4_2018_Selloff',
                         'COVID_Crash', 'Rate_Shock_2022'] if o in t.index]
    t = t.reindex(order)[MODELS]
    fig, ax = plt.subplots(figsize=(7.6, 3.6), constrained_layout=True)
    im = ax.imshow(t.values, cmap='RdYlGn_r', vmin=0, vmax=max(6, np.nanmax(t.values)),
                   aspect='auto')
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, fontsize=8, rotation=15)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index, fontsize=8)
    for i in range(len(t)):
        for j in range(len(MODELS)):
            v = t.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if v > 3 else '#222')
    fig.colorbar(im, ax=ax, label='breach rate %')
    ax.set_title('99% VaR breach rate by regime, pooled over six indices. Target 1.00%.',
                 fontsize=9)
    fig.savefig('results/figures/48_crisis_heatmap.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    x = np.arange(len(deg))
    axes[0].bar(x - 0.2, deg['normal_rate_pct'], 0.4, label='Normal', color='#9BB7D4')
    axes[0].bar(x + 0.2, deg['crisis_rate_pct'], 0.4, label='Crisis', color='#C44E52')
    axes[0].axhline(1.0, color='#333', ls='--', lw=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(deg['model'], fontsize=7.5, rotation=15)
    axes[0].set_ylabel('breach rate %'); axes[0].legend(fontsize=8)
    axes[0].set_title('Levels — what actually happened', fontsize=9)
    b = axes[1].bar(x, deg['ratio'], 0.55, color='#8C8C8C')
    axes[1].bar_label(b, fmt='%.2f', fontsize=7.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(deg['model'], fontsize=7.5, rotation=15)
    axes[1].set_ylabel('crisis / normal')
    axes[1].set_title('Ratio — flattered by a bad calm-market baseline', fontsize=9)
    fig.suptitle('Crisis degradation. The left panel is the evidence; the right panel '
                 'is the one that misleads.', fontsize=9.5)
    fig.savefig('results/figures/48_degradation.png'); plt.close(fig)
    print('\nwrote 4 tables and 2 figures')

if __name__ == '__main__':
    main()
