# -*- coding: utf-8 -*-
"""
ROBUSTNESS CHECKS — Researcher A, plan item "Robustness checks", 16h.

Five checks, each answering a specific question a reviewer will ask:

  1. SUB-SAMPLE STABILITY. Are the GJR-skewt parameters stable pre- vs post-COVID, or did the
     2020 shock permanently reweight the persistence/leverage estimates? Split at 2020-02-20
     (the crisis-window start already used everywhere else - see CRISIS_PERIODS.csv), refit
     the primary spec on each half independently, and compare.

  2. INNOVATION DISTRIBUTION. Does the choice of Normal vs Student-t vs skew-t change the
     answer, or just the likelihood? Pulled from 27_baseline_garch.py's own full comparison
     (garch_baseline_params.csv) rather than refit here - the fits already exist. Reports the
     AIC ranking and how much the persistence estimate moves across distributions.

  3. SAMPLING-FREQUENCY SENSITIVITY (5-min RV vs 10/15/30-min). The EDA's own microstructure-
     noise diagnostic (`NoiseRatio_1v5`) motivated using 5-min RV rather than 1-min. This check
     asks the coarser-grid question: does the Hansen-Lunde scale factor - and hence the
     Realized-GARCH-relevant "how much does session RV understate close-to-close variance" -
     move materially across 5/10/15/30-min sampling. A stable scale factor across frequencies
     is evidence the 5-min choice is not doing quiet, unacknowledged work.

  4. WINDOW-LENGTH / REFIT-CADENCE SENSITIVITY. The rolling engine (29_*.py) refits every 21
     trading days. This check refits the same primary spec at 63-day (quarterly) cadence on
     one index (SPX, as the reference market) and compares the resulting forecast series -
     if 21-day and 63-day cadences give materially different SigmaHat, the cadence choice is
     not a free implementation detail and must be reported as such.

  5. NKY MISSING-RV ROBUSTNESS. Does the realized-information volatility result depend on
     including NKY, or specifically on the days its realized measure was recursion-imputed
     (the 2016-17 feed outage, and the causal Hansen-Lunde scale factor's own warm-up window)?
     Splits RealGARCH QLIKE three ways: all six markets, five markets excluding NKY, and
     within NKY, observed-RV days vs recursion-imputed days.

OUTPUTS
  08_VALIDATION/robustness_subsample_stability.csv
  08_VALIDATION/robustness_distribution_comparison.csv
  08_VALIDATION/robustness_frequency_sensitivity.csv
  08_VALIDATION/robustness_refit_cadence_sensitivity.csv
  08_VALIDATION/robustness_nky_missing_rv.csv
  08_VALIDATION/ROBUSTNESS_SUMMARY.md
  11_LOGS/phase22_robustness.log
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from arch import arch_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
RVDIR = os.path.join(ROOT, '06_REALIZED_MEASURES')
VAL = os.path.join(ROOT, '08_VALIDATION')
LOG = os.path.join(ROOT, '11_LOGS')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]
COVID_SPLIT = pd.Timestamp('2020-02-20')   # matches CRISIS_PERIODS.csv COVID_Crash start

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
fio = importlib.import_module('26_forecast_io')


def fit_gjr_skewt(returns100):
    am = arch_model(returns100, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="skewt")
    return am.fit(disp="off", show_warning=False)


# ---------------------------------------------------------------------------
# 1. sub-sample stability
# ---------------------------------------------------------------------------
def check_subsample_stability():
    rows = []
    for code in CODES:
        a = pd.read_csv(os.path.join(ANA, f'{code}_analysis.csv'), parse_dates=['Date'], low_memory=False)
        r = a.set_index('Date')['Return'].dropna() * 100.0
        pre, post = r[r.index < COVID_SPLIT], r[r.index >= COVID_SPLIT]
        if len(pre) < 500 or len(post) < 500:
            continue
        rf, rp = fit_gjr_skewt(pre), fit_gjr_skewt(post)
        for label, res, n in [("pre_COVID", rf, len(pre)), ("post_COVID", rp, len(post))]:
            p = res.params
            persistence = p.get("alpha[1]", np.nan) + 0.5 * p.get("gamma[1]", 0.0) + p.get("beta[1]", np.nan)
            rows.append(dict(Code=code, Period=label, N=n, LogLik=res.loglikelihood,
                              alpha=p.get("alpha[1]", np.nan), gamma_asym=p.get("gamma[1]", np.nan),
                              beta=p.get("beta[1]", np.nan), Persistence=persistence,
                              eta_dof=p.get("eta", np.nan), lam_skew=p.get("lambda", np.nan)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(VAL, 'robustness_subsample_stability.csv'), index=False)
    return df


# ---------------------------------------------------------------------------
# 2. innovation distribution — read the comparison already computed in phase 19
# ---------------------------------------------------------------------------
def check_distribution_comparison():
    path = os.path.join(VAL, 'garch_baseline_params.csv')
    if not os.path.exists(path):
        print("  garch_baseline_params.csv not found - run 27_baseline_garch.py first")
        return pd.DataFrame()
    p = pd.read_csv(path)
    rows = []
    for code, g in p.groupby('Code'):
        g = g.set_index('Spec')
        if not {'GARCH-normal', 'GARCH-t', 'GJR-skewt'}.issubset(g.index):
            continue
        best = g['AIC'].idxmin()
        rows.append(dict(
            Code=code,
            AIC_Normal=g.loc['GARCH-normal', 'AIC'], AIC_t=g.loc['GARCH-t', 'AIC'],
            AIC_GJR_skewt=g.loc['GJR-skewt', 'AIC'],
            DeltaAIC_t_vs_Normal=g.loc['GARCH-normal', 'AIC'] - g.loc['GARCH-t', 'AIC'],
            DeltaAIC_GJRskewt_vs_t=g.loc['GARCH-t', 'AIC'] - g.loc['GJR-skewt', 'AIC'],
            BestSpec=best,
            Persistence_Normal=g.loc['GARCH-normal', 'Persistence'] if 'Persistence' in g else np.nan,
            Persistence_GJRskewt=g.loc['GJR-skewt', 'Persistence'] if 'Persistence' in g else np.nan,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(VAL, 'robustness_distribution_comparison.csv'), index=False)
    return df


# ---------------------------------------------------------------------------
# 3. sampling-frequency sensitivity: Hansen-Lunde scale factor at 5/10/15/30 min
# ---------------------------------------------------------------------------
def check_frequency_sensitivity():
    rows = []
    for code in CODES:
        path = os.path.join(RVDIR, f'{code}_RV_daily.csv')
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path, parse_dates=['Date'])
        r2 = d['CloseToClose_LogRet'] ** 2
        row = dict(Code=code)
        for freq in ['5min', '10min', '15min', '30min']:
            col = f'RV_{freq}'
            if col not in d:
                continue
            m = d[col].notna() & (d[col] > 0) & r2.notna()
            c = float(r2[m].sum() / d.loc[m, col].sum())
            row[f'ScaleFactor_{freq}'] = c
            row[f'N_{freq}'] = int(m.sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    for freq in ['10min', '15min', '30min']:
        col = f'ScaleFactor_{freq}'
        if col in df and 'ScaleFactor_5min' in df:
            df[f'PctDiff_{freq}_vs_5min'] = 100 * (df[col] - df['ScaleFactor_5min']) / df['ScaleFactor_5min']
    df.to_csv(os.path.join(VAL, 'robustness_frequency_sensitivity.csv'), index=False)
    return df


# ---------------------------------------------------------------------------
# 4. refit-cadence sensitivity — SPX only, primary spec, 21-day vs 63-day
# ---------------------------------------------------------------------------
def check_refit_cadence(code="SPX"):
    engine = importlib.import_module('29_rolling_forecast_engine')
    rows = []
    series = {}
    for cadence in [21, 63]:
        engine.REFIT_EVERY = cadence
        df, refits, elapsed = engine.run_index_spec(code, "GJR-skewt")
        series[cadence] = df.set_index('Date')['SigmaHat']
        rows.append(dict(Code=code, Cadence=cadence, NRefits=refits['RefitDate'].nunique(),
                          MeanSigmaHat=df['SigmaHat'].mean(), Elapsed_s=elapsed))
    engine.REFIT_EVERY = 21  # restore default
    joined = pd.concat(series, axis=1, join='inner')
    joined.columns = [f'Sigma_{c}d' for c in joined.columns]
    rel_diff = (joined['Sigma_63d'] - joined['Sigma_21d']).abs() / joined['Sigma_21d']
    summary = pd.DataFrame(rows)
    summary['MeanAbsRelDiff_63v21'] = float(rel_diff.mean())
    summary['MaxAbsRelDiff_63v21'] = float(rel_diff.max())
    summary['Corr_63v21'] = float(joined['Sigma_21d'].corr(joined['Sigma_63d']))
    summary.to_csv(os.path.join(VAL, 'robustness_refit_cadence_sensitivity.csv'), index=False)
    return summary


# ---------------------------------------------------------------------------
# 5. NKY missing-RV robustness (execution-plan item 10, 2026-08-29 addition).
#    "All six markets" vs "five markets excl. NKY" vs, within NKY itself,
#    observed-RV days vs recursion-imputed days (the 2016-17 feed outage, plus the causal
#    Hansen-Lunde warm-up window). Turns a footnote into a table with QLIKE numbers attached.
# ---------------------------------------------------------------------------
def check_nky_missing_rv(model="RealGARCH"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    bc = importlib.import_module('40_b_common')
    per_index = {}
    for code in CODES:
        path = os.path.join(fio.FCDIR, f'{model}__{code}_forecasts.csv')
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path, parse_dates=['Date'])
        d = d[d['Valid'] == True]
        per_index[code] = d

    rows = []
    six = pd.concat(per_index.values(), ignore_index=True) if per_index else pd.DataFrame()
    if len(six):
        L = bc.vol_losses(six['RVProxy'], six['VarHat'])
        rows.append(dict(Comparison="all_six_markets", N_obs=L['n'], QLIKE=L['QLIKE'], RMSE=L['RMSE']))
    five = pd.concat([v for k, v in per_index.items() if k != "NKY"], ignore_index=True) if len(per_index) > 1 else pd.DataFrame()
    if len(five):
        L = bc.vol_losses(five['RVProxy'], five['VarHat'])
        rows.append(dict(Comparison="five_markets_excl_NKY", N_obs=L['n'], QLIKE=L['QLIKE'], RMSE=L['RMSE']))

    if "NKY" in per_index:
        n = per_index["NKY"]
        imputed = n['Reason'].astype(str).eq('RV_imputed_in_recursion')
        for label, mask in [("NKY_observed_RV_days", ~imputed), ("NKY_imputed_recursion_days", imputed)]:
            sub = n[mask]
            if len(sub) < 20:
                rows.append(dict(Comparison=label, N_obs=len(sub), QLIKE=np.nan, RMSE=np.nan))
                continue
            L = bc.vol_losses(sub['RVProxy'], sub['VarHat'])
            rows.append(dict(Comparison=label, N_obs=L['n'], QLIKE=L['QLIKE'], RMSE=L['RMSE']))
        rows.append(dict(Comparison="NKY_pct_days_imputed",
                          N_obs=len(n), QLIKE=np.nan, RMSE=np.nan,
                          Note=f"{100*imputed.mean():.1f}%"))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(VAL, 'robustness_nky_missing_rv.csv'), index=False)
    return df


def main():
    os.makedirs(VAL, exist_ok=True)
    os.makedirs(LOG, exist_ok=True)
    t0 = time.time()
    log_lines = [f"phase22 robustness checks — {pd.Timestamp.now()}", ""]

    print("1/4 sub-sample stability (pre/post COVID, GJR-skewt) ...")
    r1 = check_subsample_stability()
    print(r1.to_string(index=False))
    log_lines.append("subsample stability:\n" + r1.to_string(index=False))

    print("\n2/4 innovation distribution comparison ...")
    r2 = check_distribution_comparison()
    print(r2.to_string(index=False) if len(r2) else "  (skipped - run 27 first)")
    log_lines.append("\ndistribution comparison:\n" + (r2.to_string(index=False) if len(r2) else "skipped"))

    print("\n3/4 sampling-frequency sensitivity (Hansen-Lunde scale factor) ...")
    r3 = check_frequency_sensitivity()
    print(r3.to_string(index=False))
    log_lines.append("\nfrequency sensitivity:\n" + r3.to_string(index=False))

    print("\n4/5 refit-cadence sensitivity (SPX, 21d vs 63d) ...")
    r4 = check_refit_cadence()
    print(r4.to_string(index=False))
    log_lines.append("\nrefit cadence sensitivity:\n" + r4.to_string(index=False))

    print("\n5/5 NKY missing-RV robustness (RealGARCH QLIKE: 6mkts / 5mkts-excl-NKY / NKY split) ...")
    r5 = check_nky_missing_rv()
    print(r5.to_string(index=False) if len(r5) else "  (skipped - run 28_realized_garch.py first)")
    log_lines.append("\nNKY missing-RV robustness:\n" + (r5.to_string(index=False) if len(r5) else "skipped"))

    md = ["# Robustness checks — summary", "",
          f"Run {pd.Timestamp.now():%Y-%m-%d %H:%M}. Researcher A, plan item \"Robustness checks\".", "",
          "## 1. Sub-sample stability (pre/post COVID-19, GJR-skewt)",
          "COVID split at 2020-02-20 (crisis-window start). Compare `Persistence` (alpha + "
          "0.5*gamma + beta) and the skew parameter `lam_skew` across the two halves.", "",
          r1.to_markdown(index=False) if len(r1) else "(no data)", "",
          "## 2. Innovation distribution (Normal vs Student-t vs skew-t)",
          "AIC differences of 2+ units are conventionally decisive. `DeltaAIC_t_vs_Normal` "
          "and `DeltaAIC_GJRskewt_vs_t` should both be positive and large if heavy tails and "
          "asymmetry are real features of the data, not overfitting.", "",
          r2.to_markdown(index=False) if len(r2) else "(run 27_baseline_garch.py first)", "",
          "## 3. Sampling-frequency sensitivity (Hansen-Lunde scale factor, 5/10/15/30-min RV)",
          "If `PctDiff_*_vs_5min` is small (a few percent), the choice of 5-min sampling for "
          "the primary realized-measure series is not doing unacknowledged work relative to "
          "coarser, even-lower-noise alternatives.", "",
          r3.to_markdown(index=False) if len(r3) else "(no data)", "",
          "## 4. Refit-cadence sensitivity (SPX, GJR-skewt, 21-day vs 63-day expanding refit)",
          "`Corr_63v21` close to 1 and a small `MeanAbsRelDiff_63v21` mean the rolling engine's "
          "REFIT_EVERY=21 choice is not materially different from a coarser, cheaper cadence - "
          "i.e. the 21-day default is a compute-cost choice, not a result-changing one.", "",
          r4.to_markdown(index=False) if len(r4) else "(no data)", "",
          "## 5. NKY missing-RV robustness (RealGARCH, QLIKE against RVProxy)",
          "Does the realized-information result depend on including NKY, or on the days its "
          "realized measure was recursion-imputed (2016-17 feed outage; causal Hansen-Lunde "
          "warm-up window)? `NKY_pct_days_imputed` is informational (Note column), not a loss.", "",
          r5.to_markdown(index=False) if len(r5) else "(run 28_realized_garch.py first)", ""]
    with open(os.path.join(VAL, 'ROBUSTNESS_SUMMARY.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(md))

    with open(os.path.join(LOG, 'phase22_robustness.log'), 'w') as f:
        f.write("\n".join(log_lines) + f"\n\ndone in {time.time()-t0:.1f}s\n")

    print(f"\nDone in {time.time()-t0:.1f}s. See 08_VALIDATION/ROBUSTNESS_SUMMARY.md")


if __name__ == "__main__":
    main()
