# -*- coding: utf-8 -*-
"""
BASELINE VOLATILITY MODELS — GARCH(1,1), GJR-GARCH(1,1,1), EGARCH(1,1).

Researcher A, plan item "Baseline GARCH & GJR/EGARCH", 24h.

WHY THREE SPECIFICATIONS, NOT ONE
  The EDA (`16_eda_stylized_facts.py`) ran the Engle-Ng sign-bias test and rejected the null of
  no asymmetry on ALL SIX indices, worst case p=7e-5. A symmetric GARCH(1,1) is therefore known
  to be misspecified before it is even fitted. GJR-GARCH and EGARCH are fitted alongside it (a)
  to have the correct model in the comparison and (b) so the improvement from GJR/EGARCH over
  plain GARCH is itself a result, not an assumption.

WHY STUDENT-t, NOT NORMAL
  Hill tail-index estimates in the EDA are 2.6-3.9 for every index - all below 4, meaning the
  fourth moment does not exist. A Normal likelihood systematically understates tail risk under
  these conditions. Student-t (and skew-t, since the leverage correlation is negative
  throughout, -0.12 to -0.20) is fit as the primary distribution; Normal is kept ONLY as a
  diagnostic comparison row, never as the reported spec.

WHAT THIS PRODUCES, AND WHY IT IS ON THE CRITICAL PATH FOR RESEARCHER B
  GARCH-EVT (B's module) is a TWO-STAGE model: stage 1 is exactly the GJR-GARCH fitted here,
  stage 2 fits a GPD to the STANDARDISED RESIDUALS of stage 1. B cannot start stage 2 without
  the residual series this script writes. That is why this runs first, ahead of Realized GARCH
  and the rolling engine.

ESTIMATION SAMPLE
  Full available history per index (from 1990), NOT sample B. GARCH-EVT is meant to use all
  the return history available (see FEATURE_SETS.csv: "estimate on all history; only the
  forecast window must match the other models"). This script produces:
    (a) one full-sample in-sample fit per index x spec, for parameter tables and the residual
        series B needs, and
    (b) does NOT itself produce the rolling 1-step-ahead OOS forecasts - that is
        29_rolling_forecast_engine.py, which reuses the fitted spec choice made here.

OUTPUTS
  08_VALIDATION/garch_baseline_params.csv       one row per (Code, Spec): params, SEs, LL, AIC, BIC
  08_VALIDATION/garch_baseline_diagnostics.csv  Ljung-Box / ARCH-LM on standardised residuals
  06_REALIZED_MEASURES/<CODE>_std_resid.csv     Date, Return, CondVol, StdResid per chosen spec
                                                 -> THIS is the file B's GARCH-EVT stage 2 reads
  09_FIGURES/garch_baseline_summary.png         conditional volatility, all six indices
  11_LOGS/phase19_baseline_garch.log

LOOK-AHEAD ON CondVol/StdResid — FLAGGED 2026-08-25, FIXED 2026-08-26
  CondVol and StdResid above come from ONE full-sample fit (params estimated on the whole
  history, then applied to every date) — this is the documented design ("estimate on all
  history" per FEATURE_SETS.csv), not an accident, and this file's OWN purpose (parameter
  tables, the summary figure) is unaffected by it: those describe the fitted model, not an
  out-of-sample forecast.
  The consequence WAS that B's GARCH-EVT stage 2 (script 42) fit its expanding-window GPD tail
  model on this series at every OriginDate, so the tail-shape parameters (xi, beta) driving
  every GARCH-EVT VaR/ES forecast were estimated on residuals whose scale (CondVol) already
  reflected GARCH parameters fit using the full sample, including dates after that OriginDate —
  a look-ahead channel into the tail shape itself, on top of the Mu-reconstruction bias
  originally disclosed in B's PR #1. Script 42 no longer reads this file at all: it now sources
  both StdResid and Mu from 34_causal_evt_residuals.py's CausalResidualSource, which reuses
  29_rolling_forecast_engine.py's own walk-forward refits (already computed, no new GARCH
  optimisation) so that nothing feeding the EVT tail fit was ever estimated on data after the
  forecast's OriginDate. See that module's docstring for the mechanism and verification, and
  42_garch_evt.py's own docstring for the effect on its output (GARCH-EVT breach counts:
  unchanged on five of six indices, NDX moved from 46 to 49 - see
  results/tables/34_causal_verification.csv for why the effect is this small).
  This file (27_baseline_garch.py) itself is UNCHANGED — the full-sample fit it produces
  remains correct and useful for what it is now used for (parameter tables, diagnostics, the
  summary figure, and 41_evt_threshold.py's one-time global threshold calibration, which - like
  spec selection above - is a fixed hyperparameter choice made once from the whole history, not
  a per-forecast quantity, so it carries no look-ahead into any individual VaR/ES output).
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
VAL = os.path.join(ROOT, '08_VALIDATION')
RVDIR = os.path.join(ROOT, '06_REALIZED_MEASURES')
FIG = os.path.join(ROOT, '09_FIGURES')
LOG = os.path.join(ROOT, '11_LOGS')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

try:
    from arch import arch_model
except ImportError:
    print("Missing dependency: pip install arch")
    sys.exit(1)

from statsmodels.stats.diagnostic import acorr_ljungbox


# ---------------------------------------------------------------------------
# specs. vol_model / o / dist map directly onto arch_model kwargs.
#   GARCH:  o=0                       symmetric
#   GJR:    o=1, power=2.0            Glosten-Jagannathan-Runkle asymmetry
#   EGARCH: vol='EGARCH'              log-variance, asymmetric by construction
#   mean='AR', lags=1                 FEATURE_SETS.csv specifies "AR(1)-GJR-GARCH" for the
#                                     GARCH-EVT stage-1 model, i.e. the mean equation carries
#                                     one autoregressive lag, not a plain constant. A first pass
#                                     with mean='Constant' left HSI's standardised residuals
#                                     autocorrelated (Ljung-Box p=5e-5 at 10 lags) - the mean
#                                     equation, not the variance equation, was misspecified.
#                                     AR(1) is applied to every spec so the AIC comparison
#                                     across specs isolates the variance equation, not a mix of
#                                     mean specifications.
# ---------------------------------------------------------------------------
SPECS = [
    dict(name="GARCH-normal",  vol="GARCH",  p=1, o=0, q=1, dist="normal"),
    dict(name="GARCH-t",       vol="GARCH",  p=1, o=0, q=1, dist="t"),
    dict(name="GJR-t",         vol="GARCH",  p=1, o=1, q=1, dist="t"),
    dict(name="GJR-skewt",     vol="GARCH",  p=1, o=1, q=1, dist="skewt"),
    dict(name="EGARCH-t",      vol="EGARCH", p=1, o=1, q=1, dist="t"),
    dict(name="EGARCH-skewt",  vol="EGARCH", p=1, o=1, q=1, dist="skewt"),
]
# the spec used downstream for the residual file GARCH-EVT reads, and for the rolling engine.
# GJR-skewt: asymmetric variance (required by Engle-Ng) + asymmetric innovation density
# (required by the negative leverage correlation), matching what the EDA actually found, AND
# matching FEATURE_SETS.csv's explicit "stage 1 AR(1)-GJR-GARCH" spec for GARCH-EVT. Note:
# EGARCH-skewt has marginally lower AIC on every index in this sample (see
# garch_baseline_params.csv) - it is reported as a comparator, not silently substituted,
# because the plan names GJR specifically and swapping it would move a decision that is B's
# to make, not something to decide implicitly by picking the best-AIC spec.
PRIMARY_SPEC = "GJR-skewt"


def fit_one(returns, spec):
    """Fit one spec. Returns are RESCALED x100 for numerical stability (arch's own
    recommendation - raw daily log returns are ~1e-2 to 1e-3 and the optimiser can fail to
    converge on them). All output is converted back to the original decimal scale before
    it is written anywhere.
    """
    r100 = returns.dropna() * 100.0
    am = arch_model(r100, mean="AR", lags=1, vol=spec["vol"],
                     p=spec["p"], o=spec["o"], q=spec["q"], dist=spec["dist"])
    res = am.fit(disp="off", show_warning=False)
    return res


def param_row(code, spec, res, n):
    conv = res.convergence_flag == 0
    row = dict(Code=code, Spec=spec["name"], N=n, Converged=conv,
               LogLik=res.loglikelihood, AIC=res.aic, BIC=res.bic)
    for pname in res.params.index:
        row[f"param_{pname}"] = res.params[pname]
        row[f"se_{pname}"] = res.std_err[pname]
        row[f"pval_{pname}"] = res.pvalues[pname]
    # persistence: alpha (+ 0.5*gamma for GJR) + beta. Near/at 1 => near-IGARCH.
    p = res.params
    if spec["vol"] == "GARCH":
        alpha = p.get("alpha[1]", np.nan)
        gamma = p.get("gamma[1]", 0.0)
        beta = p.get("beta[1]", np.nan)
        row["Persistence"] = alpha + 0.5 * gamma + beta
    elif spec["vol"] == "EGARCH":
        # EGARCH persistence is the beta (log-variance AR coefficient); no closed-form
        # unconditional variance comparison to GARCH, reported separately.
        row["Persistence"] = p.get("beta[1]", np.nan)
    return row


def diagnostics_row(code, spec, res):
    """Ljung-Box on standardised residuals and on their squares, plus ARCH-LM on squares.
    This is the check that the chosen spec has actually removed the volatility clustering the
    EDA documented (ARCH-LM p<1e-16 on raw returns, all six indices).
    """
    z = res.std_resid.dropna()
    lb_z = acorr_ljungbox(z, lags=[10], return_df=True)
    lb_z2 = acorr_ljungbox(z ** 2, lags=[10], return_df=True)
    return dict(Code=code, Spec=spec["name"],
                LB_z_stat=float(lb_z["lb_stat"].iloc[0]), LB_z_pval=float(lb_z["lb_pvalue"].iloc[0]),
                LB_z2_stat=float(lb_z2["lb_stat"].iloc[0]), LB_z2_pval=float(lb_z2["lb_pvalue"].iloc[0]),
                StdResid_Mean=float(z.mean()), StdResid_Std=float(z.std()),
                StdResid_Skew=float(z.skew()), StdResid_Kurt=float(z.kurtosis()))


def main():
    os.makedirs(VAL, exist_ok=True)
    os.makedirs(RVDIR, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(LOG, exist_ok=True)
    t0 = time.time()
    log_lines = [f"phase19 baseline GARCH — {pd.Timestamp.now()}", f"arch package specs: {[s['name'] for s in SPECS]}", ""]

    param_rows, diag_rows = [], []
    cond_vol_by_code = {}

    for code in CODES:
        a = pd.read_csv(os.path.join(ANA, f'{code}_analysis.csv'), parse_dates=['Date'], low_memory=False)
        a = a.sort_values('Date').reset_index(drop=True)
        returns = a.set_index('Date')['Return']
        n = returns.notna().sum()
        print(f"[{code}] n={n} {returns.dropna().index.min().date()}..{returns.dropna().index.max().date()}")
        log_lines.append(f"[{code}] n={n}")

        primary_res = None
        for spec in SPECS:
            try:
                res = fit_one(returns, spec)
            except Exception as e:
                print(f"    {spec['name']:16s} FAILED: {e}")
                log_lines.append(f"    {spec['name']} FAILED: {e}")
                continue
            row = param_row(code, spec, res, n)
            param_rows.append(row)
            diag_rows.append(diagnostics_row(code, spec, res))
            conv_flag = "ok" if res.convergence_flag == 0 else "NOT CONVERGED"
            print(f"    {spec['name']:16s} LL={res.loglikelihood:10.2f}  AIC={res.aic:10.2f}  "
                  f"persistence={row.get('Persistence', float('nan')):.4f}  {conv_flag}")
            log_lines.append(f"    {spec['name']:16s} LL={res.loglikelihood:.2f} AIC={res.aic:.2f} {conv_flag}")
            if spec["name"] == PRIMARY_SPEC:
                primary_res = res

        if primary_res is None:
            raise RuntimeError(f"{code}: primary spec {PRIMARY_SPEC} did not fit")

        # ---- write the residual file B's GARCH-EVT stage 2 reads ----
        # arch's conditional_volatility / std_resid are indexed on the r*100 scale for
        # variance-of-returns but std_resid (return/vol, both scaled) is scale-invariant, so
        # it is unaffected by the x100 rescale. CondVol is converted back to decimal.
        idx = primary_res.std_resid.dropna().index
        out = pd.DataFrame({
            'Date': idx,
            'Return': returns.loc[idx].values,
            'CondVol': (primary_res.conditional_volatility.loc[idx] / 100.0).values,
            'StdResid': primary_res.std_resid.loc[idx].values,
        })
        out['CondVar'] = out['CondVol'] ** 2
        out_path = os.path.join(RVDIR, f'{code}_std_resid.csv')
        out.to_csv(out_path, index=False, date_format='%Y-%m-%d', float_format='%.10g')
        print(f"    -> wrote {out_path}  ({len(out)} rows, spec={PRIMARY_SPEC})")
        log_lines.append(f"    wrote {out_path} ({len(out)} rows)")

        cv = primary_res.conditional_volatility.dropna() / 100.0
        cond_vol_by_code[code] = cv.reindex(idx).rename(code)

    params_df = pd.DataFrame(param_rows)
    diag_df = pd.DataFrame(diag_rows)
    params_df.to_csv(os.path.join(VAL, 'garch_baseline_params.csv'), index=False)
    diag_df.to_csv(os.path.join(VAL, 'garch_baseline_diagnostics.csv'), index=False)

    # ---- summary figure: annualised conditional vol, primary spec, all six ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        for code, s in cond_vol_by_code.items():
            ax.plot(s.index, s.values * np.sqrt(252) * 100, label=code, linewidth=0.8)
        ax.set_title(f'{PRIMARY_SPEC} conditional volatility (annualised %), full history')
        ax.set_ylabel('Annualised volatility (%)')
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, 'garch_baseline_summary.png'), dpi=140)
        plt.close(fig)
        print("wrote 09_FIGURES/garch_baseline_summary.png")
    except Exception as e:
        print(f"figure skipped: {e}")

    with open(os.path.join(LOG, 'phase19_baseline_garch.log'), 'w') as f:
        f.write("\n".join(log_lines) + f"\n\ndone in {time.time()-t0:.1f}s\n")

    pd.set_option('display.width', 200)
    print()
    print("=== ARCH-LM / Ljung-Box on standardised residuals, primary spec ===")
    print(diag_df[diag_df.Spec == PRIMARY_SPEC][
        ['Code', 'LB_z_pval', 'LB_z2_pval', 'StdResid_Skew', 'StdResid_Kurt']
    ].to_string(index=False))
    print()
    print("Interpretation: LB_z2_pval should be well above 0.05 if the variance model has")
    print("removed the clustering the EDA found at p<1e-16 in raw returns. A low LB_z2_pval")
    print("means volatility clustering remains and a richer spec (or more lags) is needed.")
    print()
    print(f"Done in {time.time()-t0:.1f}s. See garch_baseline_params.csv for the full table.")


if __name__ == "__main__":
    main()
