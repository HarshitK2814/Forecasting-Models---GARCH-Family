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

COMMON EVALUATION WINDOW - TWO STAGES, UPDATED 2026-08-29
  Stage 1 (sample B). Every model is cut to the contract-valid GJR-skewt calendar,
  which spans InSample_B. Before the walk-forward Realized GARCH rebuild this was
  the whole fix: RealGARCH's files started 2011-09 to 2012-02 rather than
  2013-09-30, so it alone was scored on up to 511 pre-sample-B days and alone
  picked up a Euro_Sovereign_Debt regime ending before sample B begins.

  Stage 2 (strict window). The rebuild reversed the direction of that problem.
  Walk-forward re-estimation needs a burn-in, so RealGARCH's FIRST forecast is now
  later than GJR-skewt's by a different amount on every index - DAX 2015-09-28
  against 2013-09-30, i.e. 2,765 days against 3,267. Stage 1 cannot repair this,
  because .intersection() only ever removes days.

  Left unfixed the distortion is not random. China_Deval_Oil opens in August 2015,
  so on DAX the RealGARCH series simply does not cover the start of that window -
  the devaluation days themselves - while every other model does. A pooled breach
  RATE built that way mixes a model effect with a window effect, and the missing
  days are the worst ones.

  This script therefore intersects the three VARIANCE-forecasting models' valid
  dates per index (GARCH-EVT, GJR-skewt, RealGARCH) and evaluates every model on
  that common index. It mirrors strict_window() in 47_evaluation.py deliberately,
  so the crisis tables and the headline VaR tables rest on the same definition.

  QR IS EXCLUDED FROM THE INTERSECTION, for the reason 47 gives: QR-Full's 13
  predictors fail on 23-42% of days, so including it would hand the window to QR
  and discard thousands of good observations from the other four. QR is carried
  through, cut to the strict index, and still reported with its own n - which
  remains lower than the rest. That residual gap is a genuine coverage limitation
  of the model, not a windowing artefact, and Dataset_Guide precaution 7 says to
  report it rather than balance it away.

  Both views are written. The strict tables are the ones the paper's cross-model
  comparison should quote; the unrestricted tables remain as the
  each-model-on-its-own-coverage view, and 48_window_effect.csv is the difference.

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
# 2026-08-29: the three variance-forecasting models, intersected per index to give the
# strict common window. Matches STRICT_MODELS in 47_evaluation.py - keep them in step.
STRICT_MODELS = ['GARCH-EVT', 'GJR-skewt', 'RealGARCH']
ALPHA    = 0.01
MIN_DAYS = 100         # 2026-08-29: was 40. This is a POOLED threshold, so at 40 the
                       # short windows still slipped through: Volmageddon pools to 53
                       # days (7 breaches) and Yen_Carry_Unwind to 42 (3 breaches),
                       # both marked reportable and written to the CSVs even though
                       # per index they are the 7-9 day windows that must never be
                       # quoted as rates. 100 pooled days is ~17 per index across six,
                       # which still only supports a rate loosely - the per-index table
                       # and the printed breach counts remain the primary record.
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
    # Stage 1: restrict to sample B using A's contract-compliant GJR-skewt calendar.
    for c in INDICES:
        ref = out[('GJR-skewt', c)].index
        for m in MODELS:
            out[(m, c)] = out[(m, c)].loc[out[(m, c)].index.intersection(ref)]
    print('sample-B days per model x index:')
    print(pd.DataFrame({m: {c: len(out[(m, c)]) for c in INDICES}
                        for m in MODELS})[MODELS].to_string())
    return out


def strict_frames(data):
    """Stage 2. Per-index intersection of the three variance models' valid dates.

    Mirrors strict_window() in 47_evaluation.py. QR is deliberately not part of the
    intersection - see the module docstring - but is carried through, cut to the
    strict index, and reported with its own n.

    NON-DESTRUCTIVE: returns a new dict so both views can be tabulated.
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
    rep = pd.DataFrame(rep)
    print('\nstrict-window days per model x index '
          '(intersection of %s):' % ' / '.join(STRICT_MODELS))
    print(rep.pivot(index='index', columns='model',
                    values='rows_strict')[MODELS].loc[INDICES].to_string())
    print('days dropped relative to the sample-B view:')
    print(rep.pivot(index='index', columns='model',
                    values='dropped_for_common_window')[MODELS].loc[INDICES].to_string())
    return out, rep


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


def degradation(pool_c):
    """Normal vs weighted crisis rate per model. LEVELS are primary, ratio secondary.

    THE AGGREGATE USES EVERY CRISIS DAY, INCLUDING UNREPORTABLE WINDOWS
      `reportable` gates whether an individual regime's rate may be PRINTED - a rate on
      a 7-9 day per-index window is not a rate. It must not also gate this aggregate.
      The two are different questions: the aggregate pools ~2,500 crisis days across
      regimes, where the small-sample objection does not apply, and dropping the short
      windows would remove the two most violent episodes in the sample (Volmageddon at
      13.2% and Yen_Carry_Unwind at 7.1% on the strict window) and so understate crisis
      degradation. Filtering on `reportable` here moved GARCH-EVT's crisis rate from
      2.196 to 1.885 for no defensible reason.

      n_unreportable_days records how much of the aggregate comes from windows too short
      to quote individually, so a reader can see the exposure.
    """
    deg = []
    for m in MODELS:
        s = pool_c[pool_c['model'] == m].set_index('regime')
        if 'Normal' not in s.index:
            continue
        norm = s.loc['Normal', 'rate_pct']
        cr = s[s.index != 'Normal']
        w = (cr['rate_pct'] * cr['n']).sum() / cr['n'].sum()
        short = cr[~cr['reportable']]
        deg.append({'model': m, 'normal_rate_pct': norm, 'crisis_rate_pct': round(w, 3),
                    'ratio': round(w / norm, 3), 'crisis_days': int(cr['n'].sum()),
                    'n_unreportable_days': int(short['n'].sum()),
                    'unreportable_windows': '|'.join(sorted(short.index)) or 'none'})
    return pd.DataFrame(deg)


def main():
    os.makedirs('results/tables', exist_ok=True); os.makedirs('results/figures', exist_ok=True)
    data = frames()

    # ---- unrestricted view: each model on its own sample-B coverage ----
    per_c, pool_c = by_regime(data, 'CrisisLabel')
    per_v, pool_v = by_regime(data, 'VolRegime_ExAnte')
    per_c.to_csv('results/tables/48_crisis_by_index.csv', index=False)
    pool_c.to_csv('results/tables/48_crisis_pooled.csv', index=False)
    pool_v.to_csv('results/tables/48_volregime_pooled.csv', index=False)
    deg = degradation(pool_c)
    deg.to_csv('results/tables/48_degradation.csv', index=False)

    # ---- strict view: one common window across the three variance models ----
    sdata, swin = strict_frames(data)
    swin.to_csv('results/tables/48_strict_window.csv', index=False)
    sper_c, spool_c = by_regime(sdata, 'CrisisLabel')
    sper_v, spool_v = by_regime(sdata, 'VolRegime_ExAnte')
    sper_c.to_csv('results/tables/48_crisis_by_index_strict.csv', index=False)
    spool_c.to_csv('results/tables/48_crisis_pooled_strict.csv', index=False)
    spool_v.to_csv('results/tables/48_volregime_pooled_strict.csv', index=False)
    sdeg = degradation(spool_c)
    sdeg.to_csv('results/tables/48_degradation_strict.csv', index=False)

    # ---- what the windowing alone moved ----
    eff = spool_c.merge(pool_c, on=['model', 'regime'], suffixes=('_strict', '_unrestricted'))
    eff['rate_change_pp'] = (eff['rate_pct_strict'] - eff['rate_pct_unrestricted']).round(3)
    eff['days_dropped'] = eff['n_unrestricted'] - eff['n_strict']
    eff[['model', 'regime', 'n_unrestricted', 'n_strict', 'days_dropped',
         'rate_pct_unrestricted', 'rate_pct_strict', 'rate_change_pp']].to_csv(
        'results/tables/48_window_effect.csv', index=False)

    def show(p, title, order=None):
        t = p[p['reportable']].pivot(index='regime', columns='model', values='rate_pct')
        t = t.reindex(columns=MODELS)
        if order: t = t.reindex([o for o in order if o in t.index])
        n = p[p['reportable']].pivot(index='regime', columns='model', values='n')
        n = n.reindex(t.index).reindex(columns=MODELS)
        b = p[p['reportable']].pivot(index='regime', columns='model', values='breaches')
        b = b.reindex(t.index).reindex(columns=MODELS)
        print(f'\n=== {title} (target 1.00%) ===')
        print(t.round(2).to_string())
        # Per-model day counts, not one shared figure: on the unrestricted view the
        # denominators genuinely differ by model, and printing only GARCH-EVT's would
        # imply a common window that does not exist.
        print('pooled DAYS behind each rate:')
        print(n.astype('Int64').to_string())
        print('pooled BREACHES behind each rate:')
        print(b.astype('Int64').to_string())

    CRISES = ['Normal', 'China_Deval_Oil', 'Q4_2018_Selloff', 'COVID_Crash', 'Rate_Shock_2022']
    QUARTS = ['Calm', 'Normal', 'Stressed', 'Crisis']

    print('\n' + '=' * 78)
    print('UNRESTRICTED VIEW - each model on its own sample-B coverage')
    print('denominators differ by model; not a like-for-like cross-model comparison')
    print('=' * 78)
    show(pool_c, 'pooled 99% VaR breach rate by named crisis window', CRISES)
    show(pool_v, 'pooled 99% VaR breach rate by ex-ante volatility quartile', QUARTS)

    dropped = pool_c[~pool_c['reportable']]['regime'].unique()
    print(f'\nwindows too short to report as rates (<{MIN_DAYS} pooled days): '
          f'{sorted(dropped)}')

    print('\n=== degradation, unrestricted: LEVELS first, ratio second ===')
    print(deg.to_string(index=False))

    print('\n' + '=' * 78)
    print('STRICT VIEW - one common window per index, the three variance models')
    print('intersected. THESE ARE THE CROSS-MODEL COMPARISON TABLES FOR THE PAPER.')
    print('=' * 78)
    show(spool_c, 'STRICT pooled 99% VaR breach rate by named crisis window', CRISES)
    show(spool_v, 'STRICT pooled 99% VaR breach rate by ex-ante volatility quartile', QUARTS)

    sdropped = spool_c[~spool_c['reportable']]['regime'].unique()
    print(f'\nwindows too short to report as rates (<{MIN_DAYS} pooled days): '
          f'{sorted(sdropped)}')

    print('\n=== degradation, STRICT: LEVELS first, ratio second ===')
    print(sdeg.to_string(index=False))
    print('\nA lower ratio can mean a model is WORSE in calm markets, not better in')
    print('crises. Read the crisis_rate_pct column, not the ratio, as primary.')

    print('\n=== what the common window alone moved (largest 12 by |pp|) ===')
    e = eff[eff['reportable_strict'] & eff['reportable_unrestricted']].copy()
    e['abs'] = e['rate_change_pp'].abs()
    print(e.sort_values('abs', ascending=False)
           .head(12)[['model', 'regime', 'n_unrestricted', 'n_strict',
                      'rate_pct_unrestricted', 'rate_pct_strict', 'rate_change_pp']]
           .to_string(index=False))

    # Figures show the STRICT view, matching the tables the paper quotes.
    t = spool_c[spool_c['reportable']].pivot(index='regime', columns='model', values='rate_pct')
    order = [o for o in ['Normal', 'China_Deval_Oil', 'Q4_2018_Selloff',
                         'COVID_Crash', 'Rate_Shock_2022'] if o in t.index]
    t = t.reindex(order)[MODELS]
    fig, ax = plt.subplots(figsize=(7.6, 3.6), constrained_layout=True)
    vmax = max(6, np.nanmax(t.values))
    im = ax.imshow(t.values, cmap='RdYlGn_r', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, fontsize=8, rotation=15)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index, fontsize=8)
    for i in range(len(t)):
        for j in range(len(MODELS)):
            v = t.values[i, j]
            if np.isfinite(v):
                # RdYlGn_r passes through pale yellow around the middle of the
                # range (v ~ 3-4 here), and pale yellow is light -- a raw value
                # threshold (v > 3) put white text on exactly those cells and
                # made them unreadable. Read the cell's actual rendered color
                # and pick white/black from its luminance instead.
                r, g, b, _ = im.cmap(im.norm(v))
                lum = 0.299*r + 0.587*g + 0.114*b
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if lum < 0.5 else '#222')
    fig.colorbar(im, ax=ax, label='breach rate %')
    ax.set_title('99% VaR breach rate by regime, pooled over six indices, common window. Target 1.00%.',
                 fontsize=9)
    fig.savefig('results/figures/48_crisis_heatmap.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    x = np.arange(len(sdeg))
    axes[0].bar(x - 0.2, sdeg['normal_rate_pct'], 0.4, label='Normal', color='#9BB7D4')
    axes[0].bar(x + 0.2, sdeg['crisis_rate_pct'], 0.4, label='Crisis', color='#C44E52')
    axes[0].axhline(1.0, color='#333', ls='--', lw=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(sdeg['model'], fontsize=7.5, rotation=15)
    axes[0].set_ylabel('breach rate %'); axes[0].legend(fontsize=8)
    axes[0].set_title('Levels — what actually happened', fontsize=9)
    b = axes[1].bar(x, sdeg['ratio'], 0.55, color='#8C8C8C')
    axes[1].bar_label(b, fmt='%.2f', fontsize=7.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(sdeg['model'], fontsize=7.5, rotation=15)
    axes[1].set_ylabel('crisis / normal')
    axes[1].set_title('Ratio — flattered by a bad calm-market baseline', fontsize=9)
    fig.suptitle('Crisis degradation. The left panel is the evidence; the right panel '
                 'is the one that misleads.', fontsize=9.5)
    fig.savefig('results/figures/48_degradation.png'); plt.close(fig)
    print('\nwrote 10 tables and 2 figures '
      '(strict tables are the ones the paper quotes)')

if __name__ == '__main__':
    main()
