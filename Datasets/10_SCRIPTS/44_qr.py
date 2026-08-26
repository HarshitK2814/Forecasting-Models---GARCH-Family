# -*- coding: utf-8 -*-
"""
44 - Quantile regression VaR, two specifications, all six indices.

WHAT THIS MODEL IS
  Conditional quantiles of next-day return estimated DIRECTLY from predictors.
  No volatility equation, no distributional assumption. That is the point of
  including it: GARCH-EVT and Realized GARCH both route through a conditional
  variance, so QR is the one specification whose failures cannot be blamed on
  the variance model.

THE FORECASTING LAG - Dataset_Guide precaution 3
  Nothing in 01_ANALYSIS_READY is pre-lagged; every predictor is dated at the
  close of day t. This script applies .shift(1) so the row targeting return(t)
  carries predictors observed at t-1. Failing to do this produces spectacular
  results and invalidates the paper. An assertion below re-checks it.

SigmaHat - A KNOWN GAP WITH THE FORECAST CONTRACT
  26_forecast_io.py lists SigmaHat and VarHat as REQUIRED_NUMERIC and validate()
  rejects NaN on valid rows. Quantile regression produces no conditional standard
  deviation, so there is nothing honest to put there.
  Resolution: SigmaHat is RECONSTRUCTED from the fitted quantiles under a Gaussian
  shape assumption, sigma = (VaR_05 - VaR_01) / (z_05 - z_01), and the Spec string
  says so. It exists only to satisfy the schema.
  *** QR's SigmaHat IS NOT A VOLATILITY FORECAST. Never score it with QLIKE, MSE
  or any other volatility loss function. eval_frame() will happily compute a QLIKE
  from it and the number will be meaningless. ***

QUANTILE CROSSING
  Each tau is estimated by a separate regression, so fitted quantiles can cross
  (VaR_01 above VaR_025). The contract requires VaR_01 <= VaR_025 <= VaR_05 < 0.
  Crossings are repaired by sorting the three fitted values (Chernozhukov,
  Fernandez-Val and Galichon rearrangement) and the count is reported, not hidden.

EVALUATION WINDOW - RESEARCHER_A_DECISIONS section 3
  Output is reindexed onto A's GJR-skewt forecast dates, so all four models share
  one identical Date index and paired tests need no alignment step. Estimation may
  reach back to 1990 (QR-Range) but evaluation is sample B, like every other model.
  The pre-2013 QR-Range run is a SEPARATE daily-only robustness table, script 46.

SPECIFICATIONS
  QR-Full  13 predictors. Binding constraint is the realized block, which starts
           2012-01, NOT CreditStress (2007-04) as an earlier note recorded.
  QR-Range 5 predictors, all available from 1990. Deliberately excluded:
           LogRS_neg (VIF ~95, collinear with LogRV by identity), US10Y_pct and
           TermSpread_pct in levels (non-stationary), VolRegime (look-ahead).

OUTPUT
  20_FORECASTS/QR-Full__<CODE>_forecasts.csv    contract format
  20_FORECASTS/QR-Range__<CODE>_forecasts.csv   contract format
  results/tables/44_qr_summary.csv              coverage, crossings, breach rates
  results/tables/44_qr_coefficients_<SPEC>.csv  final-window coefficients and t-stats
"""
import os, warnings, importlib.util, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import IterationLimitWarning

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
fio = _load('fio', os.path.join(HERE, '26_forecast_io.py'))

BASE     = 'Datasets/01_ANALYSIS_READY'
FORECAST = 'Datasets/20_FORECASTS'
INDICES  = ['SPX', 'NDX', 'UKX', 'DAX', 'NKY', 'HSI']

QR_CORE  = ['LogRV', 'LogRV_w', 'LogRV_m', 'LogIV', 'VRP', 'NegReturn', 'RangePct']
QR_ASYM  = ['RSV_Ratio', 'JumpShare', 'RSkew']
QR_MACRO = ['TermSpread_diff', 'CreditStress', 'DXY_ret']
SPECS = {
    'QR-Full':  QR_CORE + QR_ASYM + QR_MACRO,
    'QR-Range': ['RangePct', 'NegReturn', 'LogIV', 'DXY_ret', 'TermSpread_diff'],
}
LEVELS      = [('01', 0.01), ('025', 0.025), ('05', 0.05)]
MIN_TRAIN   = 750
REFIT_EVERY = 10
Z = {tag: stats.norm.ppf(tau) for tag, tau in LEVELS}


def build_frame(a, preds):
    """Lagged design matrix. Row dated t carries predictors observed at t-1."""
    X = a[preds].shift(1)
    d = X.copy()
    d['_y'] = a['Return']
    d = d.dropna()
    if len(d) > 5:                      # the lag, re-checked rather than trusted
        t = d.index[-1]
        prev = a.index[a.index.get_loc(t) - 1]
        assert np.isclose(d.loc[t, preds[0]], a.loc[prev, preds[0]], equal_nan=True), \
            'forecasting lag not applied - predictors are contemporaneous'
    return d


def rolling_qr(d, preds):
    """Expanding-window QR. Standardisation uses TRAINING-window statistics only;
    full-sample statistics would leak future information into every forecast."""
    Xraw, yv, n = d[preds], d['_y'], len(d)
    out, cached, scaler = [], {}, None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IterationLimitWarning)
        for t in range(MIN_TRAIN, n):
            if not cached or (t - MIN_TRAIN) % REFIT_EVERY == 0:
                tr = Xraw.iloc[:t]
                mu, sd = tr.mean(), tr.std().replace(0, 1.0)
                scaler = (mu, sd)
                trs = sm.add_constant((tr - mu) / sd, has_constant='add')
                for tag, tau in LEVELS:
                    try:
                        cached[tag] = sm.QuantReg(yv.iloc[:t], trs).fit(q=tau, max_iter=5000)
                    except Exception:
                        cached.pop(tag, None)
            if len(cached) < len(LEVELS) or scaler is None:
                continue
            mu, sd = scaler
            row = sm.add_constant((Xraw.iloc[[t]] - mu) / sd, has_constant='add')
            row = row.reindex(columns=list(cached.values())[0].params.index, fill_value=0.0)
            rec = {'Date': d.index[t]}
            for tag, _ in LEVELS:
                rec[f'raw_{tag}'] = float(cached[tag].predict(row).iloc[0])
            out.append(rec)
    return pd.DataFrame(out).set_index('Date'), cached, scaler


def main():
    os.makedirs('results/tables', exist_ok=True)
    summary, coef_rows = [], {k: [] for k in SPECS}

    for spec_name, preds in SPECS.items():
        for code in INDICES:
            a = pd.read_csv(f'{BASE}/{code}_analysis.csv', parse_dates=['Date'],
                            low_memory=False).set_index('Date').sort_index()
            d = build_frame(a, preds)
            fitted, cached, scaler = rolling_qr(d, preds)

            raw = fitted[[f'raw_{t}' for t, _ in LEVELS]].values
            n_cross = int((np.diff(raw, axis=1) < 0).any(axis=1).sum())
            srt = np.sort(raw, axis=1)                 # ascending: 01 <= 025 <= 05
            for j_, (tag, _) in enumerate(LEVELS):
                fitted[f'VaR_{tag}'] = srt[:, j_]

            fitted['SigmaHat'] = (fitted['VaR_05'] - fitted['VaR_01']) / (Z['05'] - Z['01'])

            base = fio.read_forecasts(f'{FORECAST}/GJR-skewt__{code}_forecasts.csv')
            base = base.sort_values('Date').reset_index(drop=True)
            j = base[['Date', 'OriginDate', 'Realized', 'RVProxy']].merge(
                fitted.reset_index(), on='Date', how='left')

            j['Horizon'] = 1
            ok = j['SigmaHat'].notna() & (j['SigmaHat'] > 0) & (j['VaR_05'] < 0)
            j['Valid'] = ok
            j['Reason'] = np.where(j['SigmaHat'].isna(), 'predictors_unavailable',
                          np.where(~ok, 'degenerate_quantile_fit', ''))
            j['VarHat'] = j['SigmaHat'] ** 2
            for tag, _ in LEVELS:
                j.loc[~ok, f'VaR_{tag}'] = np.nan
            j.loc[~ok, ['SigmaHat', 'VarHat']] = np.nan
            j['ES_01'] = np.nan          # QR produces no expected shortfall
            j['ES_025'] = np.nan

            fio.write_forecasts(j, spec_name, code,
                                spec=f'{spec_name}-expanding-min{MIN_TRAIN}-'
                                     f'refit{REFIT_EVERY}-SigmaHatRECONSTRUCTED')
            v = j[j['Valid']]
            br = (v['Realized'] < v['VaR_01'])
            summary.append({'spec': spec_name, 'index': code,
                            'n_rows': len(j), 'n_valid': int(ok.sum()),
                            'n_invalid': int((~ok).sum()),
                            'first_forecast': str(fitted.index.min().date()),
                            'n_crossings': n_cross,
                            'pct_crossings': round(100 * n_cross / max(len(fitted), 1), 2),
                            'median_sigmahat': round(float(v['SigmaHat'].median()), 6),
                            'breach_01': int(br.sum()),
                            'rate_01_pct': round(100 * br.mean(), 3),
                            'rate_025_pct': round(100 * (v['Realized'] < v['VaR_025']).mean(), 3),
                            'rate_05_pct': round(100 * (v['Realized'] < v['VaR_05']).mean(), 3)})
            print(f'{spec_name:9} {code}: valid={int(ok.sum())}/{len(j)}  '
                  f'crossings={n_cross} ({100*n_cross/max(len(fitted),1):.2f}%)  '
                  f'1% breach={int(br.sum())} ({100*br.mean():.3f}%)')

            if '01' in cached:
                r = cached['01']
                for nm, cf, tv, pv in zip(r.params.index, r.params, r.tvalues, r.pvalues):
                    coef_rows[spec_name].append({'index': code, 'tau': 0.01,
                                                 'predictor': nm, 'coef': cf,
                                                 't': tv, 'p': pv})

    s = pd.DataFrame(summary)
    s.to_csv('results/tables/44_qr_summary.csv', index=False)
    for k, v in coef_rows.items():
        pd.DataFrame(v).to_csv(f'results/tables/44_qr_coefficients_{k}.csv', index=False)
    print('\n' + s.to_string(index=False))

    for spec_name in SPECS:
        for code in INDICES:
            fio.read_forecasts(f'{FORECAST}/{spec_name}__{code}_forecasts.csv')
    print('\nall twelve QR files re-read and re-validated through the contract')

if __name__ == '__main__':
    main()
