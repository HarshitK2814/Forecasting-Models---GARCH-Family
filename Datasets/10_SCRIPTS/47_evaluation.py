# -*- coding: utf-8 -*-
"""
47 - Evaluation metrics. Executive Summary section 4.2.

THREE QUESTIONS, THREE TABLES, DELIBERATELY NOT MIXED
  47a volatility accuracy   RMSE, MAE, QLIKE against the realized proxy
  47b VaR backtests         Kupiec, Christoffersen independence and joint CC, DQ
  47c expected shortfall    Acerbi-Szekely Z2 and the realised/predicted ratio

WHICH MODELS ARE ELIGIBLE FOR WHICH TABLE - this is not a detail
  Volatility accuracy requires a genuine conditional variance forecast.
    GJR-skewt   yes
    RealGARCH   yes
    GARCH-EVT   yes, but it REUSES A's SigmaHat, so its variance forecast is
                IDENTICAL to GJR-skewt's by construction. QLIKE cannot separate
                them and a Diebold-Mariano test between them returns NaN.
    QR-Full     NO. Its SigmaHat is reconstructed from fitted quantiles purely to
    QR-Range    satisfy the forecast contract (see script 44). It is not a
                volatility forecast. QLIKE on it would return a plausible and
                meaningless number, so QR is EXCLUDED from table 47a by name.

  Expected shortfall requires an ES forecast.
    GARCH-EVT   yes        RealGARCH   yes
    GJR-skewt   no, ES_01 is empty in A's file
    QR          no, quantile regression produces no ES

COMMON EVALUATION WINDOW - THE FIX THIS SCRIPT EXISTS TO APPLY
  A's RealGARCH files violate contract rule 6 ("Rows span each index InSample_B
  days"). They start 2011-09 to 2012-02 rather than 2013-09-30, giving RealGARCH
  up to 511 days that NO other model is evaluated on. validate() does not check
  the window, so nothing caught it. Symptom that exposed it: RealGARCH showed
  21,681 pooled index-days against a maximum of 19,333, and was the only model
  with a Euro_Sovereign_Debt regime - a crisis that ends before sample B begins.
  Every model is therefore cut to A's GJR-skewt calendar, which is contract-valid
  and spans InSample_B, BEFORE any metric is computed. See common_window() for why
  this is NOT an all-model intersection.

PAIRED COMPARISONS
  Diebold-Mariano needs both loss series on the SAME days. Models have different
  valid coverage - QR-Full manages only 1,814 of NKY's 3,150 days - so every
  pairwise test is run on the intersection of the two models' valid dates, and the
  n used is reported.

OUTPUT
  results/tables/47_common_window.csv
  results/tables/47a_volatility_losses.csv   47a_dm_volatility.csv
  results/tables/47b_var_backtests.csv
  results/tables/47c_es_backtests.csv
"""
import os, importlib.util, itertools, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
bc  = _load('bc',  os.path.join(HERE, '40_b_common.py'))
fio = _load('fio', os.path.join(HERE, '26_forecast_io.py'))

FORECAST = 'Datasets/20_FORECASTS'
INDICES  = ['SPX', 'NDX', 'UKX', 'DAX', 'NKY', 'HSI']
MODELS   = ['GARCH-EVT', 'GJR-skewt', 'RealGARCH', 'QR-Full', 'QR-Range']
VOL_MODELS = ['GARCH-EVT', 'GJR-skewt', 'RealGARCH']       # QR excluded by design
ES_MODELS  = ['GARCH-EVT', 'RealGARCH']
LEVELS   = [('01', 0.01), ('025', 0.025), ('05', 0.05)]


def load(model, code):
    d = fio.read_forecasts(f'{FORECAST}/{model}__{code}_forecasts.csv')
    return d[d['Valid'].astype(bool)].set_index('Date').sort_index()


def common_window(data):
    """Restrict every model to the sample-B window, using A's contract-compliant
    GJR-skewt file as the reference calendar.

    NOT an all-model intersection. Forcing every model onto the days where all
    five can forecast would hand the window to QR-Full, whose 13 predictors fail
    on 23-42% of days, and would discard thousands of good observations from the
    other four. Dataset_Guide precaution 7 is explicit that per-index work should
    use each index's own valid days rather than a balanced panel.

    What IS fixed here is the genuine violation: RealGARCH's file starts 2011-09
    to 2012-02 instead of 2013-09-30, so it alone carried up to 511 pre-sample-B
    days. Those are removed. Coverage differences that arise from a model
    legitimately being unable to forecast are left alone and reported instead.
    """
    rep = []
    for c in INDICES:
        ref = data[('GJR-skewt', c)].index          # spans InSample_B, contract-valid
        for m in MODELS:
            before = len(data[(m, c)])
            data[(m, c)] = data[(m, c)].loc[data[(m, c)].index.intersection(ref)]
            rep.append({'index': c, 'model': m, 'rows_before': before,
                        'rows_after': len(data[(m, c)]),
                        'dropped_outside_sampleB': before - len(data[(m, c)])})
    return data, pd.DataFrame(rep)


def main():
    os.makedirs('results/tables', exist_ok=True)
    data = {(m, c): load(m, c) for m in MODELS for c in INDICES}
    data, win = common_window(data)
    win.to_csv('results/tables/47_common_window.csv', index=False)
    print('=== rows removed for falling outside sample B ===')
    print(win.pivot(index='index', columns='model', values='dropped_outside_sampleB')[MODELS]
          .loc[INDICES].to_string())
    print('rows retained per model x index:')
    print(win.pivot(index='index', columns='model', values='rows_after')[MODELS]
          .loc[INDICES].to_string())

    # ---------------------------------------------------- 47a volatility
    vol_rows = []
    for c in INDICES:
        for m in VOL_MODELS:
            d = data[(m, c)]
            ok = d['RVProxy'].notna()          # QLIKE needs the realized proxy
            L = bc.vol_losses(d.loc[ok, 'RVProxy'], d.loc[ok, 'VarHat'])
            vol_rows.append({'index': c, 'model': m, **L})
    vol = pd.DataFrame(vol_rows)
    vol.to_csv('results/tables/47a_volatility_losses.csv', index=False)

    dm_rows = []
    for c in INDICES:
        for m1, m2 in itertools.combinations(VOL_MODELS, 2):
            a, b = data[(m1, c)], data[(m2, c)]
            idx = a.index.intersection(b.index)               # paired on Date
            idx = idx[a.loc[idx, 'RVProxy'].notna() & b.loc[idx, 'RVProxy'].notna()]
            # Identity guard. GARCH-EVT reuses A's SigmaHat, so its variance
            # forecast IS GJR-skewt's. The two files round to 10 significant
            # digits independently, leaving relative differences around 1e-9.
            # Left alone, DM would compute a statistic on that rounding noise
            # and report a winner.
            rel = ((a.loc[idx, 'VarHat'] - b.loc[idx, 'VarHat']).abs()
                   / b.loc[idx, 'VarHat']).max()
            if rel < 1e-6:
                dm_rows.append({'index': c, 'model_1': m1, 'model_2': m2,
                                'n': len(idx), 'dm_stat': np.nan, 'p': np.nan,
                                'better': 'identical_by_construction'})
                continue
            l1 = bc.qlike_series(a.loc[idx, 'RVProxy'], a.loc[idx, 'VarHat'])
            l2 = bc.qlike_series(b.loc[idx, 'RVProxy'], b.loc[idx, 'VarHat'])
            r = bc.diebold_mariano(l1, l2)
            dm_rows.append({'index': c, 'model_1': m1, 'model_2': m2, 'n': r['n'],
                            'dm_stat': r['stat'], 'p': r['p'],
                            'better': (np.nan if not np.isfinite(r['stat'])
                                       else (m1 if r['stat'] < 0 else m2))})
    dm = pd.DataFrame(dm_rows)
    dm.to_csv('results/tables/47a_dm_volatility.csv', index=False)

    # ---------------------------------------------------- 47b VaR backtests
    bt_rows = []
    for c in INDICES:
        for m in MODELS:
            d = data[(m, c)]
            for tag, alpha in LEVELS:
                bt_rows.append(bc.backtest(d['Realized'], d[f'VaR_{tag}'], alpha, m, c))
    bt = pd.DataFrame(bt_rows)
    bt.to_csv('results/tables/47b_var_backtests.csv', index=False)

    # ---------------------------------------------------- 47c expected shortfall
    es_rows = []
    for c in INDICES:
        for m in ES_MODELS:
            d = data[(m, c)]
            for tag, alpha in [('01', 0.01), ('025', 0.025)]:
                ok = d[f'ES_{tag}'].notna()
                if ok.sum() == 0:
                    continue
                r = bc.es_ratio(d.loc[ok, 'Realized'], d.loc[ok, f'VaR_{tag}'],
                                d.loc[ok, f'ES_{tag}'])
                z = bc.acerbi_szekely_z2(d.loc[ok, 'Realized'], d.loc[ok, f'VaR_{tag}'],
                                         d.loc[ok, f'ES_{tag}'], alpha, n_sim=3000)
                # mean excess beyond VaR, in units of the VaR level. Carried
                # because the ES ratio ALONE is not comparable across models:
                # a model that breaches twice as often breaches on milder days,
                # which flatters its ratio without its ES being better.
                br = d.loc[ok, 'Realized'] < d.loc[ok, f'VaR_{tag}']
                exc = ((d.loc[ok, 'Realized'][br] - d.loc[ok, f'VaR_{tag}'][br])
                       / d.loc[ok, f'VaR_{tag}'][br].abs()).mean()
                es_rows.append({'index': c, 'model': m, 'level': tag, **r,
                                'mean_excess': float(exc),
                                'Z2': z['Z2'], 'Z2_p': z['p']})
    es = pd.DataFrame(es_rows)
    es.to_csv('results/tables/47c_es_backtests.csv', index=False)

    # ---------------------------------------------------- console
    print('\n=== 47a  QLIKE, lower is better (QR excluded: no volatility forecast) ===')
    print(vol.pivot(index='index', columns='model', values='QLIKE')[VOL_MODELS]
          .loc[INDICES].round(4).to_string())
    print('\nDiebold-Mariano on QLIKE (negative stat = model_1 better):')
    for _, r in dm.iterrows():
        st = 'NaN (identical forecasts)' if not np.isfinite(r['dm_stat']) \
             else f"{r['dm_stat']:+.2f}  p={r['p']:.4f}  -> {r['better']}"
        print(f"  {r['index']}  {r['model_1']:10} vs {r['model_2']:10} n={r['n']:5}  {st}")

    print('\n=== 47b  99% VaR: breach rate and four tests ===')
    b1 = bt[bt['confidence'] == 0.99]
    print(b1.pivot(index='index', columns='model', values='rate_pct')[MODELS]
          .loc[INDICES].round(3).to_string())
    print('\npass counts out of 6 indices, 99% VaR:')
    for m in MODELS:
        s = b1[b1['model'] == m]
        print(f"  {m:10} Kupiec {int(s['pass_kupiec'].sum())}/6   "
              f"indep {int(s['pass_indep'].sum())}/6   "
              f"joint CC {int(s['pass_cc'].sum())}/6   "
              f"DQ {int(s['pass_dq'].sum())}/6")

    print('\n=== 47c  Expected shortfall, realised / predicted on breach days ===')
    e1 = es[es['level'] == '01']
    print(e1.pivot(index='index', columns='model', values='ratio')[ES_MODELS]
          .loc[INDICES].round(3).to_string())
    print('\n(ratio > 1 means realised tail losses ran DEEPER than the model predicted)')
    print('\nCAUTION: do not rank models on this ratio. Mean excess beyond VaR:')
    print(e1.pivot(index='index', columns='model', values='mean_excess')[ES_MODELS]
          .loc[INDICES].round(3).to_string())
    print('A model with more breaches breaches on MILDER days, which flatters its ratio.')
    print('breaches behind each ratio:')
    print(e1.pivot(index='index', columns='model', values='n_breach')[ES_MODELS]
          .loc[INDICES].to_string())
    print('\nwrote 5 tables')

if __name__ == '__main__':
    main()
