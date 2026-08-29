# -*- coding: utf-8 -*-
"""
41 - EVT threshold diagnostics on GENUINE GARCH residuals.

WHY THIS SCRIPT EXISTS
  Dataset_Guide.pdf precaution 12: the shipped 95th-97.5th percentile
  recommendation (Guide section 10) was computed on ROLLING-STANDARDISED RETURNS,
  a stand-in used because no GARCH had been fitted at the EDA stage.
  McNeil-Frey (2000) applies the GPD to genuine GARCH residuals. This redoes the
  diagnostic on 06_REALIZED_MEASURES/<CODE>_std_resid.csv, which
  RESEARCHER_A_SCOPE.md section 1 names as the stage-2 input.

DECISIONS RECORDED HERE (both were left to Researcher B in writing)
  1. Stage-1 filter: AR(1)-GJR-GARCH-skew-t. FEATURE_SETS.csv names this spec.
     RESEARCHER_A_SCOPE.md section 1 notes EGARCH-skewt has marginally lower AIC
     on all six but did NOT substitute it, because swapping the plan's named
     model is B's call. Decision: KEEP GJR - the plan names it and the AIC gap is
     marginal. Stated, not implied.
  2. Threshold: chosen on exceedance count and fit quality, not convention alone.
     The sampling-SE table makes "no clean plateau" a calibrated statement.
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, importlib.util
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location('bc', os.path.join(HERE, '40_b_common.py'))
bc = importlib.util.module_from_spec(_s); _s.loader.exec_module(bc)

RESID   = 'Datasets/06_REALIZED_MEASURES'
INDICES = ['SPX', 'NDX', 'UKX', 'DAX', 'NKY', 'HSI']
QGRID   = [0.90, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.99]
CHOSEN  = 0.95
PAL     = ["#378ADD", "#D85A30", "#1D9E75", "#BA7517", "#7F77DD", "#8C8C8C"]

# Guide section 10: xi on rolling-standardised RETURNS - the stand-in being replaced
GUIDE_XI = {
    0.90:  {'DAX':0.07067,'HSI':0.2144,'NDX':0.09342,'NKY':0.1331,'SPX':0.1402,'UKX':0.1157},
    0.925: {'DAX':0.09920,'HSI':0.2258,'NDX':0.08779,'NKY':0.1319,'SPX':0.1704,'UKX':0.1141},
    0.95:  {'DAX':0.16320,'HSI':0.2071,'NDX':0.11850,'NKY':0.1534,'SPX':0.1753,'UKX':0.1429},
    0.96:  {'DAX':0.16810,'HSI':0.1808,'NDX':0.09846,'NKY':0.2019,'SPX':0.1436,'UKX':0.1349},
    0.97:  {'DAX':0.18700,'HSI':0.1964,'NDX':0.13690,'NKY':0.1960,'SPX':0.1739,'UKX':0.1545},
    0.975: {'DAX':0.22210,'HSI':0.1957,'NDX':0.15860,'NKY':0.2768,'SPX':0.1810,'UKX':0.1257},
    0.98:  {'DAX':0.24510,'HSI':0.2268,'NDX':0.17350,'NKY':0.2521,'SPX':0.2296,'UKX':0.1710},
    0.99:  {'DAX':0.20950,'HSI':0.1716,'NDX':0.22850,'NKY':0.1833,'SPX':0.3805,'UKX':0.1899},
}

def load_losses(code):
    """Left-tail loss scale: L = -StdResid, positive = bad."""
    r = pd.read_csv(f'{RESID}/{code}_std_resid.csv', parse_dates=['Date']).set_index('Date')
    return -r['StdResid'].dropna().values

def xi_sampling_se(n_exceed, xi_true=0.15, beta=0.58, reps=600, seed=bc.SEED):
    """Simulated SD of xi-hat at a given exceedance count. This is what turns
    'xi drifts with the threshold' into a statement with a scale attached."""
    rng = np.random.default_rng(seed)
    out = [bc.fit_gpd(stats.genpareto.rvs(xi_true, scale=beta, size=n_exceed,
                                          random_state=rng), 0.0)['xi'] for _ in range(reps)]
    a = np.array(out)
    return {'n_exceed': n_exceed, 'sd_xi': float(a.std()),
            'pct_negative': float(100*(a < 0).mean())}

def main():
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    rows = []
    for code in INDICES:
        L = load_losses(code)
        for q in QGRID:
            g = bc.fit_gpd(L, q); gof = bc.gpd_gof(L, g)
            rows.append({'index': code, 'q': q, 'u': g['u'], 'xi': g['xi'],
                         'beta': g['beta'], 'n_exceed': g['n_exceed'],
                         'n_total': g['n_total'], 'converged': g['converged'],
                         'ks_p': gof['ks_p'], 'ad_stat': gof['ad_stat'],
                         'ad_reject_5pct': gof['ad_reject_5pct']})
    tab = pd.DataFrame(rows)
    tab.to_csv('results/tables/41_threshold_stability.csv', index=False)

    cmp = pd.DataFrame([{'index': r['index'], 'q': r['q'], 'xi_residuals': r['xi'],
                         'xi_guide_standin': GUIDE_XI.get(r['q'], {}).get(r['index'], np.nan),
                         'diff': r['xi'] - GUIDE_XI.get(r['q'], {}).get(r['index'], np.nan)}
                        for _, r in tab.iterrows()])
    cmp.to_csv('results/tables/41_threshold_vs_guide.csv', index=False)

    # ---- EVT exceedance-dependence diagnostic (execution-plan item 12) --------------------
    dep_rows = []
    for code in INDICES:
        L = load_losses(code)
        u = float(tab[(tab['index'] == code) & (tab['q'] == CHOSEN)]['u'].iloc[0])
        d = bc.exceedance_dependence(L, u)
        d['index'] = code
        dep_rows.append(d)
    dep = pd.DataFrame(dep_rows)[['index', 'u', 'n_obs', 'n_exceed', 'lb_stat', 'lb_p',
                                   'runs_observed', 'runs_z', 'runs_p', 'theta_ferro_segers']]
    dep.to_csv('results/tables/41_exceedance_dependence.csv', index=False)
    print('\n=== EVT exceedance-dependence diagnostic (threshold q=%.3f) ===' % CHOSEN)
    print('  index    LB(10) p   runs p    theta(Ferro-Segers)')
    for _, r in dep.iterrows():
        flag = '  <-- clustered (theta<<1 or p<0.05)' if (r['theta_ferro_segers'] < 0.7 or
               (np.isfinite(r['lb_p']) and r['lb_p'] < 0.05)) else ''
        print(f"  {r['index']:<6}  {r['lb_p']:.4f}     {r['runs_p']:.4f}    "
              f"{r['theta_ferro_segers']:.3f}{flag}")

    counts = sorted({int(round(c/10)*10) for c in tab['n_exceed']})
    se = pd.DataFrame([xi_sampling_se(c) for c in counts if c >= 20])
    se.to_csv('results/tables/41_xi_sampling_se.csv', index=False)
    se_at = dict(zip(se['n_exceed'], se['sd_xi']))
    nearest_se = lambda n: se_at[min(se_at, key=lambda x: abs(x - n))]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for i, code in enumerate(INDICES):
        s = tab[tab['index'] == code]
        band = np.array([nearest_se(n) for n in s['n_exceed']])
        ax.plot(s['q'], s['xi'], marker='o', ms=3.5, lw=1.4, color=PAL[i], label=code)
        ax.fill_between(s['q'], s['xi']-band, s['xi']+band, color=PAL[i], alpha=0.08, lw=0)
    ax.axvline(CHOSEN, color='#444', ls='--', lw=1); ax.axhline(0, color='#999', lw=0.8)
    ax.set_xlabel('POT threshold quantile q'); ax.set_ylabel(r'GPD shape $\xi$')
    ax.set_title('GPD shape vs threshold, genuine GARCH residuals\n'
                 r'shaded band = $\pm 1$ sampling SE at that exceedance count', fontsize=9)
    ax.legend(ncol=6, fontsize=8, frameon=False)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig('results/figures/41_threshold_stability.png', dpi=300); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6), constrained_layout=True)
    for i, code in enumerate(INDICES):
        a = axes.flat[i]; L = load_losses(code)
        g = bc.fit_gpd(L, CHOSEN); theo, emp = bc.gpd_qq(L, g)
        lim = max(theo.max(), emp.max())*1.03
        a.plot([0, lim], [0, lim], color='#999', lw=1, ls='--')
        a.scatter(theo, emp, s=7, color=PAL[i], alpha=0.65, edgecolor='none')
        gof = bc.gpd_gof(L, g)
        a.set_title(f"{code}   $\\xi$={g['xi']:.3f}  n={g['n_exceed']}  KS p={gof['ks_p']:.2f}",
                    fontsize=8.5)
        a.set_xlabel('theoretical', fontsize=8); a.set_ylabel('empirical', fontsize=8)
        for sp in ('top', 'right'): a.spines[sp].set_visible(False)
        a.grid(alpha=0.25, lw=0.5); a.tick_params(labelsize=7)
    fig.suptitle(f'GPD QQ plots of exceedances, q={CHOSEN}, GARCH residuals', fontsize=10)
    fig.savefig('results/figures/41_qq_gpd.png', dpi=300); plt.close(fig)

    print('=== xi on GENUINE GARCH residuals (Guide section 10 stand-in in brackets) ===')
    piv = tab.pivot(index='q', columns='index', values='xi')[INDICES]
    print('  q      ' + ' '.join(f'{c:>16}' for c in INDICES))
    for q in QGRID:
        print(f'  {q:<6} ' + ' '.join(f'{piv.loc[q, c]:+.3f} [{GUIDE_XI[q][c]:+.3f}]'
                                      for c in INDICES))
    print('\n=== at chosen threshold q=%.3f ===' % CHOSEN)
    s = tab[tab['q'] == CHOSEN].set_index('index')
    for c in INDICES:
        print(f"  {c}: u={s.loc[c,'u']:.4f}  xi={s.loc[c,'xi']:+.4f}  "
              f"n_exceed={int(s.loc[c,'n_exceed'])}  KS p={s.loc[c,'ks_p']:.3f}  "
              f"AD reject={bool(s.loc[c,'ad_reject_5pct'])}")
    print('\n=== sampling SE of xi (simulated, true xi=0.15) ===')
    for _, r in se.iterrows():
        print(f"  n_exceed={int(r['n_exceed']):5}  SD(xi)={r['sd_xi']:.4f}  "
              f"P(xi<0)={r['pct_negative']:.1f}%")
    print('\nwrote 4 tables and 2 figures')

if __name__ == '__main__':
    main()
