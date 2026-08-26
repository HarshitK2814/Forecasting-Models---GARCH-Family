# -*- coding: utf-8 -*-
"""
50 - Reproducibility audit. Executive Summary deliverable, week 14.

WHAT AN AUDIT IS FOR
  Not to assert that the results are correct - the individual scripts do that -
  but to make the assertion CHECKABLE by someone who did not run them. Every test
  below re-derives something from the committed artefacts rather than trusting a
  number printed earlier in a notebook.

  Two of these checks caught real errors before the work was committed:
  the pooled-day count exposed RealGARCH's contract-rule-6 window violation, and
  the Kupiec recomputation confirmed the published p-values were not stale.

WHAT IT DOES NOT CHECK
  It cannot prove the absence of look-ahead in A's upstream fits, only in the
  handoff (OriginDate < Date) and in B's own use of it. It cannot detect a
  conceptually wrong model that is internally consistent. It verifies plumbing,
  not economics.

EXIT
  Prints a PASS/FAIL line per check and writes results/tables/50_audit.csv.
  Non-zero exit status if anything fails, so it can gate a CI run later.
"""
import os, sys, glob, re, importlib.util, numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
bc  = _load('bc',  os.path.join(HERE, '40_b_common.py'))
fio = _load('fio', os.path.join(HERE, '26_forecast_io.py'))

FORECAST = 'Datasets/20_FORECASTS'
BASE     = 'Datasets/01_ANALYSIS_READY'
INDICES  = ['SPX', 'NDX', 'UKX', 'DAX', 'NKY', 'HSI']
B_MODELS = ['GARCH-EVT', 'QR-Full', 'QR-Range']       # files B is responsible for

# A_SCOPE.md section 3, the rolling engine's own published counts
A_SCOPE = {'SPX': 3243, 'NDX': 3243, 'UKX': 3258, 'DAX': 3267, 'NKY': 3150, 'HSI': 3172}
# PROJECT_STATE, recorded before this rebuild - an independent earlier run.
# GARCH-EVT's PRIOR_EVT was updated 2026-08-26 after 42_garch_evt.py switched from
# 27_baseline_garch.py's full-sample (look-ahead) residuals to
# 34_causal_evt_residuals.py's walk-forward-consistent ones (code review of PR #1).
# This is a METHOD change, not noise: only NDX's count actually moved (46 -> 49);
# the other five indices are unchanged, which is the expected size of the effect -
# see 34_causal_evt_residuals.py's own diagnostics for why (early-history residuals
# differ from the full-sample fit; recent ones barely do). The value this check
# guards is unchanged: THIS run should reproduce THIS method's own prior run
# bit-for-bit, catching non-determinism, not measuring against the old method.
PRIOR_EVT = {'SPX': 39, 'NDX': 49, 'UKX': 41, 'DAX': 41, 'NKY': 38, 'HSI': 22}
PRIOR_RG  = {'SPX': 73, 'NDX': 72, 'UKX': 63, 'DAX': 52, 'NKY': 58, 'HSI': 36}

results = []
def check(section, name, ok, detail=''):
    results.append({'section': section, 'check': name, 'pass': bool(ok), 'detail': detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ''))


def main():
    os.makedirs('results/tables', exist_ok=True)
    src = {os.path.basename(f): open(f).read() for f in sorted(glob.glob(f'{HERE}/4*.py'))}
    code = '\n'.join(src.values())
    # strip comments and docstrings before searching for banned identifiers,
    # otherwise a warning ABOUT a column reads as a use OF it
    live = re.sub(r'#.*|"""(.*?)"""', '', code, flags=re.S)

    print('=== 1. ENVIRONMENT ===')
    import statsmodels, scipy, numpy, pandas
    env = {'python': '.'.join(map(str, sys.version_info[:3])),
           'statsmodels': statsmodels.__version__, 'scipy': scipy.__version__,
           'numpy': numpy.__version__, 'pandas': pandas.__version__}
    print('  running:', env)
    if os.path.exists('requirements_B.txt'):
        pins = dict(re.findall(r'^(\w+)==([\d.]+)', open('requirements_B.txt').read(), re.M))
        drift = {k: (pins[k], env[k]) for k in pins if k in env and pins[k] != env[k]}
        check('env', 'running versions match requirements_B.txt', not drift, str(drift))
    else:
        check('env', 'requirements_B.txt present', False, 'missing')

    print('\n=== 2. CONTRACT COMPLIANCE ===')
    bad = []
    for m in B_MODELS:
        for c in INDICES:
            try: fio.read_forecasts(f'{FORECAST}/{m}__{c}_forecasts.csv', strict=True)
            except Exception as e: bad.append(f'{m}/{c}: {str(e)[:60]}')
    check('contract', 'all 18 B files pass validate(strict=True)', not bad, str(bad[:2]))

    lk = sum(int((pd.read_csv(f'{FORECAST}/{m}__{c}_forecasts.csv',
                              parse_dates=['Date', 'OriginDate'])
                  .pipe(lambda d: d['OriginDate'] >= d['Date'])).sum())
             for m in B_MODELS for c in INDICES)
    check('contract', 'OriginDate < Date on every row', lk == 0, f'violations={lk}')

    prob = []
    for m in B_MODELS:
        for c in INDICES:
            v = fio.read_forecasts(f'{FORECAST}/{m}__{c}_forecasts.csv')
            v = v[v['Valid']]
            if not (v['VaR_01'] <= v['VaR_025']).all(): prob.append(f'{m}/{c}:01>025')
            if not (v['VaR_025'] <= v['VaR_05']).all(): prob.append(f'{m}/{c}:025>05')
            if not (v['VaR_05'] < 0).all():             prob.append(f'{m}/{c}:VaR>=0')
            e = v[v['ES_01'].notna()]
            if len(e) and not (e['ES_01'] <= e['VaR_01']).all(): prob.append(f'{m}/{c}:ES>VaR')
    check('contract', 'VaR ordering and ES <= VaR hold everywhere', not prob, str(prob[:2]))

    print('\n=== 3. DATASET GUIDE PRECAUTIONS ===')
    check('precaution', '1  never reads 07_PANEL_INTERMEDIATE', '07_PANEL' not in code)
    check('precaution', '3  forecasting lag applied and asserted',
          code.count('.shift(1)') >= 2 and 'lag not applied' in code)
    check('precaution', '4  no winsorizing, trimming or de-jumping',
          not re.search(r'winsor|trim|de-?jump', live, re.I))
    check('precaution', '5  LogRS_neg never used as a predictor', 'LogRS_neg' not in live)
    check('precaution', '6  US10Y_pct / TermSpread_pct levels never used',
          not re.search(r"'(US10Y_pct|TermSpread_pct)'", live))
    check('precaution', '7  BalancedRV_B not used for per-index work',
          'BalancedRV_B' not in live)
    check('precaution', "   VolRegime (ex-post) never used, only VolRegime_ExAnte",
          "'VolRegime'" not in live and 'VolRegime_ExAnte' in code)
    # LogRV itself must be NaN outside RV_Valid. LogRV_w/m are TRAILING means and
    # legitimately exist on an invalid day, so they are excluded from this test.
    a = pd.read_csv(f'{BASE}/NKY_analysis.csv', parse_dates=['Date'], low_memory=False)
    leak = int(a.loc[~a['RV_Valid'].astype(bool), ['LogRV', 'RSV_Ratio', 'VRP']].notna().sum().sum())
    check('precaution', '2  LogRV/RSV_Ratio/VRP NaN outside RV_Valid (NKY)', leak == 0,
          f'leaks={leak}')

    print('\n=== 4. AGREEMENT WITH RESEARCHER A ===')
    got = {c: int(fio.read_forecasts(f'{FORECAST}/GARCH-EVT__{c}_forecasts.csv')['Valid'].sum())
           for c in INDICES}
    check('handoff', 'GARCH-EVT row counts match A_SCOPE section 3', got == A_SCOPE, str(got))
    w = pd.read_csv('results/tables/47_common_window.csv')
    others = int((w[w['model'] != 'RealGARCH']['dropped_outside_sampleB'] > 0).sum())
    check('handoff', 'only RealGARCH lost rows to the sample-B gate', others == 0,
          f'others={others}')

    print('\n=== 5. INTERNAL CONSISTENCY ===')
    bt = pd.read_csv('results/tables/47b_var_backtests.csv')
    b1 = bt[bt['confidence'] == 0.99]
    d = (100 * b1['n_breach'] / b1['n_obs'] - b1['rate_pct']).abs().max()
    check('consistency', 'breach rate equals n_breach / n_obs', d < 0.001, f'max dev={d:.1e}')
    rec = [bc.kupiec_pof(np.r_[np.ones(int(r.n_breach)),
                               np.zeros(int(r.n_obs - r.n_breach))], 0.01)[1]
           for r in b1.itertuples()]
    dv = float(np.max(np.abs(np.array(rec) - b1['kupiec_p'].values)))
    check('consistency', 'Kupiec p recomputable from counts alone', dv < 1e-3, f'max dev={dv:.1e}')
    ba = pd.read_csv('results/tables/49_basel.csv')
    j = b1.merge(ba, on=['index', 'model'], suffixes=('_47', '_49'))
    check('consistency', 'scripts 47 and 49 agree on all breach counts',
          (j['n_breach_47'] == j['n_breach_49']).all(), f'{len(j)} cells')

    print('\n=== 6. GUARDS AGAINST KNOWN TRAPS ===')
    vol = pd.read_csv('results/tables/47a_volatility_losses.csv')
    check('guard', 'QR absent from every volatility-loss table',
          not vol['model'].str.startswith('QR').any())
    check('guard', "QR Spec flags the reconstructed SigmaHat",
          all('RECONSTRUCTED' in pd.read_csv(f'{FORECAST}/{m}__{c}_forecasts.csv')['Spec'].iloc[0]
              for m in ['QR-Full', 'QR-Range'] for c in INDICES))
    dm = pd.read_csv('results/tables/47a_dm_volatility.csv')
    ident = dm[dm['better'] == 'identical_by_construction']
    check('guard', 'EVT vs skew-t DM returns identity, not rounding noise',
          len(ident) == 6, f'{len(ident)}/6 index pairs')
    check('guard', 'long-history run wrote no contract file',
          not glob.glob(f'{FORECAST}/*long*'))
    check('guard', '_SYNTHETIC placeholders never read', '_SYNTHETIC' not in live)

    print('\n=== 7. DETERMINISM ===')
    rngs = [os.path.basename(f) for f in glob.glob(f'{HERE}/4*.py')
            if 'default_rng' in open(f).read() and 'seed' not in open(f).read().lower()]
    check('determinism', 'every RNG call is seeded', not rngs, str(rngs))
    L = -pd.read_csv('Datasets/06_REALIZED_MEASURES/SPX_std_resid.csv')['StdResid'].dropna().values
    f1, f2 = bc.fit_gpd(L, 0.95), bc.fit_gpd(L, 0.95)
    check('determinism', 'GPD fit is bit-identical on repeat', f1['xi'] == f2['xi'])
    z1 = bc.acerbi_szekely_z2(np.random.default_rng(1).standard_normal(500)*0.01,
                              np.full(500, -0.023), np.full(500, -0.030), 0.01, n_sim=200)
    z2 = bc.acerbi_szekely_z2(np.random.default_rng(1).standard_normal(500)*0.01,
                              np.full(500, -0.023), np.full(500, -0.030), 0.01, n_sim=200)
    check('determinism', 'seeded Acerbi-Szekely reproduces exactly', z1['p'] == z2['p'])

    print('\n=== 8. INDEPENDENT PRIOR RUN ===')
    e = b1[b1['model'] == 'GARCH-EVT'].set_index('index')['n_breach'].astype(int).to_dict()
    r = b1[b1['model'] == 'RealGARCH'].set_index('index')['n_breach'].astype(int).to_dict()
    check('replication', 'GARCH-EVT breach counts match the earlier run', e == PRIOR_EVT, str(e))
    check('replication', 'RealGARCH breach counts match the earlier run', r == PRIOR_RG, str(r))

    print('\n=== 9. ARTEFACTS ===')
    nt, nf = len(glob.glob('results/tables/*.csv')), len(glob.glob('results/figures/*.png'))
    check('artefacts', 'tables written', nt >= 26, f'{nt}')
    check('artefacts', 'figures written', nf >= 15, f'{nf}')

    df = pd.DataFrame(results)
    df.to_csv('results/tables/50_audit.csv', index=False)
    n_fail = int((~df['pass']).sum())
    print('\n' + '=' * 64)
    print(f'{len(df) - n_fail}/{len(df)} checks passed')
    if n_fail:
        print('FAILED:')
        for _, r_ in df[~df['pass']].iterrows():
            print(f"  {r_['section']}: {r_['check']}  {r_['detail']}")
    else:
        print('AUDIT CLEAN - results reproduce and every documented rule is respected')
    print('=' * 64)
    return 1 if n_fail else 0

if __name__ == '__main__':
    sys.exit(main())
