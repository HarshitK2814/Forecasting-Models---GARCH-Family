# -*- coding: utf-8 -*-
"""
42 - GARCH-EVT two-stage conditional VaR and ES, all six indices.

DESIGN
  Stage 1 is NOT re-fitted here. It is the project baseline AR(1)-GJR-GARCH-skew-t,
  whose out-of-sample SigmaHat and standardised residuals arrive through
  20_FORECASTS/GJR-skewt__<CODE>_forecasts.csv and
  06_REALIZED_MEASURES/<CODE>_std_resid.csv.
  Stage 2 fits a GPD to the left tail of those residuals.

  Consequence: GARCH-EVT and the skew-t baseline share the SAME conditional
  variance (both use A's SigmaHat) and very nearly the same conditional mean.
  Mu here is reconstructed from the residual file, which is A's FULL-SAMPLE fit,
  whereas A's forecast file uses the expanding-window fit. Measured gap: zero
  mean, sd 0.001-0.008 sigma, max 0.05 sigma, against a tail effect of
  0.019-0.044 sigma. Unbiased and an order smaller, so the comparison still
  isolates the tail specification, but it is not exact and is disclosed as such.
  The shared SigmaHat also means the two have IDENTICAL volatility forecasts, so
  QLIKE cannot separate them and a Diebold-Mariano test on volatility loss returns
  NaN. That is correct behaviour, not a bug.

WINDOW DECISION - expanding, and why it differs from stage 1
  A 1,000-day rolling window leaves roughly 50 exceedances at q=0.95. Script 41
  measured the sampling SD of xi at that count: 0.135, with 18% of fits returning
  a NEGATIVE xi even when the true xi is +0.15. A negative xi implies a bounded
  tail, which is not credible for equity returns. At 300-460 exceedances the SD
  falls to about 0.056 and negative fits essentially vanish.
  Tail estimation is data-hungry in a way variance estimation is not, so the two
  stages warrant different window schemes. Stage 1 keeps A's expanding-window,
  21-day-refit scheme; stage 2 uses an expanding window over all residual history
  available at OriginDate. This asymmetry is deliberate and is disclosed.

LOOK-AHEAD
  The GPD for target date t is fitted only on residuals dated <= OriginDate.
  The contract guarantees OriginDate < Date and validate() rejects violations,
  so the inclusive slice is safe.

OUTPUT
  20_FORECASTS/GARCH-EVT__<CODE>_forecasts.csv   contract format, write_forecasts()
  results/tables/42_evt_diagnostics_<CODE>.csv   xi path, n_exceed, threshold per date
  results/tables/42_evt_summary.csv              one row per index
"""
import os, importlib.util, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
bc  = _load('bc',  os.path.join(HERE, '40_b_common.py'))
fio = _load('fio', os.path.join(HERE, '26_forecast_io.py'))

RESID    = 'Datasets/06_REALIZED_MEASURES'
FORECAST = 'Datasets/20_FORECASTS'
INDICES  = ['SPX', 'NDX', 'UKX', 'DAX', 'NKY', 'HSI']
THRESH_Q = 0.95          # chosen in script 41 on fit quality and exceedance count
MIN_OBS  = 500           # minimum residual history before a GPD fit is attempted
SPEC     = 'GARCH-EVT-GPD-q95-expanding-on-GJRskewt'

LEVELS = [('01', 0.99), ('025', 0.975), ('05', 0.95)]


def rolling_evt(code):
    """One index. Returns (forecast frame in contract shape, diagnostics frame)."""
    r = pd.read_csv(f'{RESID}/{code}_std_resid.csv', parse_dates=['Date']
                    ).set_index('Date').sort_index()
    # A fits AR(1)-GJR; recover that conditional mean from the identity
    #   Return = Mu + StdResid * CondVol
    r['Mu'] = r['Return'] - r['StdResid'] * r['CondVol']

    base = fio.read_forecasts(f'{FORECAST}/GJR-skewt__{code}_forecasts.csv')
    base = base.sort_values('Date').reset_index(drop=True)

    z_all, mu_all = r['StdResid'].dropna(), r['Mu']
    fc_rows, diag_rows = [], []

    for _, row in base.iterrows():
        date, origin = row['Date'], row['OriginDate']
        rec = {'Date': date, 'OriginDate': origin, 'Horizon': 1,
               'SigmaHat': row['SigmaHat'], 'VarHat': row['SigmaHat'] ** 2,
               'Realized': row['Realized'], 'RVProxy': row['RVProxy'],
               'Valid': False, 'Reason': ''}
        for tag, _q in LEVELS:
            rec[f'VaR_{tag}'] = np.nan
            rec[f'ES_{tag}'] = np.nan

        # ---- reasons a row can be present but not evaluable. Contract rule 5:
        #      keep the row, never drop it, or paired-length tests break silently.
        if not bool(row['Valid']):
            rec['Reason'] = row.get('Reason') or 'baseline_row_invalid'
            fc_rows.append(rec); continue
        if pd.isna(origin):
            rec['Reason'] = 'no_origin_date'; fc_rows.append(rec); continue

        hist = z_all.loc[:origin]          # OriginDate < Date, enforced by the contract
        if len(hist) < MIN_OBS:
            rec['Reason'] = f'residual_history_{len(hist)}_below_{MIN_OBS}'
            fc_rows.append(rec); continue
        try:
            g = bc.fit_gpd(-hist.values, THRESH_Q)     # loss scale: L = -z
        except ValueError as e:
            rec['Reason'] = f'gpd_fit_failed:{e}'; fc_rows.append(rec); continue

        sigma = float(row['SigmaHat'])
        mu = mu_all.get(date, np.nan)
        if not np.isfinite(mu):
            rec['Reason'] = 'mu_unavailable_on_target_date'
            fc_rows.append(rec); continue

        for tag, q in LEVELS:
            rec[f'VaR_{tag}'] = mu - sigma * bc.gpd_var(g, q)
            rec[f'ES_{tag}'] = mu - sigma * bc.gpd_es(g, q)
        rec['Valid'] = True
        fc_rows.append(rec)
        diag_rows.append({'Date': date, 'OriginDate': origin, 'xi': g['xi'],
                          'beta': g['beta'], 'u': g['u'], 'n_exceed': g['n_exceed'],
                          'n_hist': g['n_total'], 'converged': g['converged'],
                          'sigma': sigma, 'mu': mu})

    return pd.DataFrame(fc_rows), pd.DataFrame(diag_rows)


def main():
    os.makedirs('results/tables', exist_ok=True)
    summary = []
    for code in INDICES:
        fc, diag = rolling_evt(code)
        path = fio.write_forecasts(fc, 'GARCH-EVT', code, spec=SPEC)   # validates on write
        diag.to_csv(f'results/tables/42_evt_diagnostics_{code}.csv', index=False)

        v = fc[fc['Valid']]
        br = (v['Realized'] < v['VaR_01'])
        summary.append({'index': code, 'n_rows': len(fc), 'n_valid': int(fc['Valid'].sum()),
                        'n_invalid': int((~fc['Valid']).sum()),
                        'xi_min': diag['xi'].min(), 'xi_max': diag['xi'].max(),
                        'xi_negative_days': int((diag['xi'] < 0).sum()),
                        'n_exceed_min': int(diag['n_exceed'].min()),
                        'n_exceed_max': int(diag['n_exceed'].max()),
                        'breach_01': int(br.sum()),
                        'rate_01_pct': round(100 * br.mean(), 3)})
        print(f'{code}: wrote {os.path.basename(path)}  '
              f'valid={int(fc["Valid"].sum())}/{len(fc)}  '
              f'xi=[{diag["xi"].min():+.4f},{diag["xi"].max():+.4f}]  '
              f'neg_xi_days={int((diag["xi"]<0).sum())}  '
              f'1%_breach={int(br.sum())} ({100*br.mean():.3f}%)')

    s = pd.DataFrame(summary)
    s.to_csv('results/tables/42_evt_summary.csv', index=False)
    print('\n' + s.to_string(index=False))

    for code in INDICES:
        d = fio.read_forecasts(f'{FORECAST}/GARCH-EVT__{code}_forecasts.csv')
        assert d['Valid'].sum() > 0
    print('\nall six GARCH-EVT files re-read and re-validated through the contract')

if __name__ == '__main__':
    main()
