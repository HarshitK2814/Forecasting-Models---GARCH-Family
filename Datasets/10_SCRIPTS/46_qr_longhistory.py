# -*- coding: utf-8 -*-
"""
46 - QR-Range on the full 1990+ daily history. ROBUSTNESS ONLY.

STATUS OF THIS SCRIPT - read before using any number it produces
  RESEARCHER_A_DECISIONS.md section 3 restricts the out-of-sample evaluation
  window to sample B (2013-09-30 onward) so all four models are graded on the
  same days. Realized GARCH cannot start earlier: free intraday history begins
  2011, and 2013 for the DAX.

  QR-Range needs only five predictors, all available from 1990, so it CAN be
  evaluated across the GFC. That is exactly why the result must be quarantined:

      "Do not claim the models were tested through the GFC - they were not,
       on sample B."

  A also names the only legitimate form for a pre-2013 result: "a daily-only
  robustness table (GARCH-EVT / QR on the full 1990+ history, no Realized GARCH
  comparator) - flagged as a possible robustness-check addition, not built by
  default." That is what this is.

  This script deliberately writes NO contract forecast file. Nothing in
  20_FORECASTS/ comes from here, so these numbers cannot leak into the main
  comparison, the Diebold-Mariano tests or the Model Confidence Set.

COVERAGE WARNING
  LogIV starts at different dates per index, so reach varies enormously:
  SPX from 1992, DAX from 2008, the other four only from 2011-2014. Only SPX
  and DAX see the GFC at all. Do not describe this as a six-index GFC result.
"""
import os, warnings, importlib.util, numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt, matplotlib as mpl
from statsmodels.tools.sm_exceptions import IterationLimitWarning

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
bc = _load('bc', os.path.join(HERE, '40_b_common.py'))

BASE    = 'Datasets/01_ANALYSIS_READY'
INDICES = ['SPX', 'NDX', 'UKX', 'DAX', 'NKY', 'HSI']
PREDS   = ['RangePct', 'NegReturn', 'LogIV', 'DXY_ret', 'TermSpread_diff']
LEVELS  = [('01', 0.01), ('025', 0.025), ('05', 0.05)]
MIN_TRAIN, REFIT_EVERY = 750, 10
PAL = ["#378ADD", "#D85A30", "#1D9E75", "#BA7517", "#7F77DD", "#8C8C8C"]

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "legend.frameon": False})


def run_index(code):
    a = pd.read_csv(f'{BASE}/{code}_analysis.csv', parse_dates=['Date'],
                    low_memory=False).set_index('Date').sort_index()
    X = a[PREDS].shift(1)                     # precaution 3, same lag as script 44
    d = X.copy(); d['_y'] = a['Return']; d = d.dropna()

    out, cached, scaler = [], {}, None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IterationLimitWarning)
        for t in range(MIN_TRAIN, len(d)):
            if not cached or (t - MIN_TRAIN) % REFIT_EVERY == 0:
                tr = d[PREDS].iloc[:t]
                mu, sd = tr.mean(), tr.std().replace(0, 1.0); scaler = (mu, sd)
                trs = sm.add_constant((tr - mu) / sd, has_constant='add')
                for tag, tau in LEVELS:
                    try:
                        cached[tag] = sm.QuantReg(d['_y'].iloc[:t], trs).fit(q=tau, max_iter=5000)
                    except Exception:
                        cached.pop(tag, None)
            if len(cached) < len(LEVELS):
                continue
            mu, sd = scaler
            row = sm.add_constant((d[PREDS].iloc[[t]] - mu) / sd, has_constant='add')
            row = row.reindex(columns=list(cached.values())[0].params.index, fill_value=0.0)
            rec = {'Date': d.index[t], 'actual': float(d['_y'].iloc[t])}
            for tag, _ in LEVELS:
                rec[f'raw_{tag}'] = float(cached[tag].predict(row).iloc[0])
            out.append(rec)

    f = pd.DataFrame(out).set_index('Date')
    raw = f[[f'raw_{t}' for t, _ in LEVELS]].values
    n_cross = int((np.diff(raw, axis=1) < 0).any(axis=1).sum())
    srt = np.sort(raw, axis=1)                # rearrangement, as in script 44
    for j, (tag, _) in enumerate(LEVELS):
        f[f'VaR_{tag}'] = srt[:, j]
    f['CrisisLabel'] = a['CrisisLabel'].reindex(f.index)
    f['InSample_B'] = a['InSample_B'].reindex(f.index).astype(bool)
    return f, n_cross


def main():
    os.makedirs('results/tables', exist_ok=True); os.makedirs('results/figures', exist_ok=True)
    crisis_rows, summ, store = [], [], {}

    for code in INDICES:
        f, n_cross = run_index(code)
        store[code] = f
        br = (f['actual'] < f['VaR_01'])
        pre = f[~f['InSample_B']]
        pre_br = (pre['actual'] < pre['VaR_01'])
        inb = f[f['InSample_B']]
        lr, p = bc.kupiec_pof(br.astype(int).values, 0.01)
        summ.append({'index': code, 'n': len(f), 'first': str(f.index.min().date()),
                     'n_pre2013': len(pre), 'crossings_pct': round(100*n_cross/len(f), 2),
                     'rate_full_pct': round(100*br.mean(), 3),
                     'rate_pre2013_pct': round(100*pre_br.mean(), 3),
                     'rate_sampleB_pct': round(100*(inb['actual'] < inb['VaR_01']).mean(), 3),
                     'kupiec_p_full': round(p, 4)})
        for lab, g in f.groupby(f['CrisisLabel'].fillna('Unlabelled')):
            b = (g['actual'] < g['VaR_01'])
            crisis_rows.append({'index': code, 'crisis': lab, 'n': len(g),
                                'breaches': int(b.sum()),
                                'rate_pct': round(100*b.mean(), 2),
                                'in_sample_B': bool(g['InSample_B'].any())})
        print(f'{code}: n={len(f)} from {f.index.min().date()}  pre-2013={len(pre)}  '
              f'full={100*br.mean():.3f}%  pre-2013={100*pre_br.mean():.3f}%  '
              f'sampleB={100*(inb["actual"]<inb["VaR_01"]).mean():.3f}%')

    s = pd.DataFrame(summ); s.to_csv('results/tables/46_qr_longhistory_summary.csv', index=False)
    cr = pd.DataFrame(crisis_rows)
    cr.to_csv('results/tables/46_qr_longhistory_crisis.csv', index=False)

    piv = cr.pivot_table(index='crisis', columns='index', values='rate_pct')
    order = [c for c in ['Normal', 'Asian_Crisis_LTCM', 'DotCom_Bust', 'GFC',
                         'Euro_Sovereign_Debt', 'China_Deval_Oil', 'Volmageddon',
                         'Q4_2018_Selloff', 'COVID_Crash', 'Rate_Shock_2022',
                         'Yen_Carry_Unwind'] if c in piv.index]
    piv = piv.loc[order, INDICES]
    print('\n1% breach rate by crisis window (target 1.00):')
    print(piv.round(2).to_string())
    n_piv = cr.pivot_table(index='crisis', columns='index', values='n').loc[order, INDICES]
    print('\ndays per window - rates on short windows rest on 1-3 breaches:')
    print(n_piv.astype('Int64').to_string())
    print('\n' + s.to_string(index=False))

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True, constrained_layout=True)
    for i, c in enumerate(INDICES):
        ax = axes.flat[i]; f = store[c]
        b = f['actual'] < f['VaR_01']
        ax.axvspan(f.index.min(), pd.Timestamp('2013-09-30'), color='#999', alpha=0.10, zorder=0)
        ax.plot(f.index, 100*f['actual'], lw=0.3, color='#C4C4C4')
        ax.plot(f.index, 100*f['VaR_01'], lw=0.8, color=PAL[i])
        ax.scatter(f.index[b], 100*f['actual'][b], s=7, color='#C44E52', zorder=5)
        pre = f[~f['InSample_B']]; inb = f[f['InSample_B']]
        ax.set_title(f'{c}   pre-2013 {100*(pre["actual"]<pre["VaR_01"]).mean():.2f}%  |  '
                     f'sample B {100*(inb["actual"]<inb["VaR_01"]).mean():.2f}%', fontsize=8.5)
        ax.set_ylabel('%', fontsize=8); ax.tick_params(labelsize=7)
    fig.suptitle('ROBUSTNESS ONLY - QR-Range 99% VaR on full daily history.\n'
                 'Grey region is OUTSIDE sample B: no other model is evaluated there, '
                 'so these days are not part of the model comparison.', fontsize=9.5)
    fig.savefig('results/figures/46_qr_longhistory.png'); plt.close(fig)
    print('\nwrote 2 tables and 1 figure. NO contract file written - by design.')

if __name__ == '__main__':
    main()
