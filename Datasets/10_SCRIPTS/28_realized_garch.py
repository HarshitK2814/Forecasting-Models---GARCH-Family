# -*- coding: utf-8 -*-
"""
REALIZED GARCH(1,1), log-linear specification — Hansen, Huang & Shek (2012),
"Realized GARCH: A complete model of returns and realized measures of volatility",
Journal of Applied Econometrics 27(6), 877-906.

Researcher A, plan item "Realized GARCH", 24h. Not available in the `arch` package (confirmed:
`arch` ships GARCH/EGARCH/FIGARCH/APARCH/HARCH, no realized-measure variant) - implemented here
by direct quasi-maximum-likelihood, following the paper's own estimation approach.

THE MODEL (three equations, as in the paper)
    return:      r_t = mu + sqrt(h_t) * z_t                    z_t standardised Student-t(nu)
    GARCH:       log(h_t) = omega + beta*log(h_{t-1}) + gamma*log(x_{t-1})
    measurement: log(x_t) = xi + phi*log(h_t) + tau1*z_t + tau2*(z_t^2 - 1) + sigma_u * eps_t

  x_t is the realized measure (RV_Scaled - see below for why the scaled series, not raw
  session RV). tau1*z_t + tau2*(z_t^2-1) is the "leverage function": it lets the realized
  measure respond asymmetrically to the sign of the return shock, which is exactly the
  Engle-Ng asymmetry the EDA found in every index.

WHY x_t = RV_Scaled, NOT RV
  RV as computed in `01_ANALYSIS_READY` is the SESSION-ONLY realized variance - it excludes
  the overnight return by construction. The Realized GARCH measurement equation, however,
  models h_t, which is the conditional variance of r_t = the CLOSE-TO-CLOSE return. Feeding
  session-only RV into a close-to-close model means x_t is systematically 1.7x-3.0x smaller
  than what it is measuring (Hansen-Lunde scale factors, see SCALE_FACTORS.csv) - the model
  would then have to force xi or phi to absorb a scale mismatch that differs BY INDEX, which
  contaminates the leverage and persistence parameters. RV_Scaled = ScaleFactor_HL * RV
  removes this by construction; see RESEARCHER_A_DECISIONS.md and precaution list.

THE NKY GAP (2016-17) - MISSING x_t, NOT MISSING r_t
  The exchange close (r_t) is valid throughout; only the realized measure has a 2016-17 hole
  (`RV_Valid` False - see the session-classification finding in EDA_REPORT.md). This model
  needs x_{t-1} to update h_t. On a day where x_{t-1} is missing, h_{t-1} itself - the model's
  own last conditional-variance estimate - is used as a stand-in for the missing realized
  measure. This is not the paper's own procedure (the original paper doesn't confront a
  feed-outage this long); it is a fallback documented here as one, and it is the natural one:
  the measurement equation says E[x_t | h_t] = h_t up to the leverage terms and Jensen's
  correction, so substituting h_{t-1} for a missing x_{t-1} keeps the recursion self-consistent
  rather than injecting a fabricated number into the LIKELIHOOD (the measurement-equation
  likelihood term is simply SKIPPED, not evaluated, on days where x_t is missing - only the
  GARCH recursion gets the imputed value, the objective function never sees it as data).
  Forecast rows in this window are still Valid=True (the return is real) but carry
  Reason="RV_imputed_in_recursion" so B can isolate them for a robustness table.

ESTIMATION
  Quasi-MLE via scipy.optimize.minimize (L-BFGS-B), joint log-likelihood = return density
  (Student-t) + measurement density (Gaussian, as in the original paper - u_t is the
  regression-style error of a log-linear equation, not the return itself, so a Gaussian
  measurement error is the standard and defensible choice; this is a plan robustness-check
  candidate, not asserted as the only right answer).

  Estimation sample: from each index's realized-measure start date (own history), not
  restricted to sample B - matching the "estimate on all history" convention already set for
  GARCH-EVT. NKY has no valid RV before 2011-09, so its Realized GARCH necessarily starts
  later than its return history.

OUTPUTS
  08_VALIDATION/realized_garch_params.csv         one row per Code: all 9 parameters + SEs (numeric Hessian), LL, AIC, BIC
  06_REALIZED_MEASURES/<CODE>_realized_garch_fit.csv   Date, Return, RVProxy, CondVar, CondVol, StdResid, RV_Imputed
  20_FORECASTS/RealGARCH__<CODE>_forecasts.csv    contract-format 1-step-ahead forecasts (see 26_forecast_io.py)
  09_FIGURES/realized_garch_fit.png
  11_LOGS/phase20_realized_garch.log
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
VAL = os.path.join(ROOT, '08_VALIDATION')
RVDIR = os.path.join(ROOT, '06_REALIZED_MEASURES')
FIG = os.path.join(ROOT, '09_FIGURES')
LOG = os.path.join(ROOT, '11_LOGS')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
fio = importlib.import_module('26_forecast_io')

LEVELS = fio.LEVELS  # [("01",0.01), ("025",0.025), ("05",0.05)]

# parameter order for the optimiser vector theta
PNAMES = ["mu", "omega", "beta", "gamma", "xi", "phi", "tau1", "tau2", "log_sigma_u2", "log_nu_m2"]


def unpack(theta):
    mu, omega, beta, gamma, xi, phi, tau1, tau2, log_su2, log_num2 = theta
    sigma_u2 = np.exp(log_su2)
    nu = 2.0 + np.exp(log_num2)   # nu > 2 always, needed for finite variance
    return mu, omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2, nu


def negloglik(theta, r100, x100, h1):
    """r100: return*100 (percent). x100: realized measure on the matching percent^2 scale,
    NaN where missing. h1: fixed starting conditional variance (percent^2), not estimated -
    conditioning on it is standard practice to avoid an unidentified extra free parameter.
    """
    mu, omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2, nu = unpack(theta)
    if not (0 < beta < 1.2 and -0.5 < gamma < 2.0 and sigma_u2 > 1e-10):
        return 1e12
    n = len(r100)
    h = np.empty(n)
    h[0] = h1
    logx_used = np.where(np.isnan(x100), np.nan, np.log(np.maximum(x100, 1e-12)))
    # recursion: log(h_t) needs log(x_{t-1}); substitute log(h_{t-1}) when x_{t-1} is missing
    for t in range(1, n):
        lx_prev = logx_used[t - 1] if np.isfinite(logx_used[t - 1]) else np.log(max(h[t - 1], 1e-12))
        lh = omega + beta * np.log(max(h[t - 1], 1e-12)) + gamma * lx_prev
        lh = np.clip(lh, -30, 30)
        h[t] = np.exp(lh)
        if h[t] <= 0 or not np.isfinite(h[t]):
            return 1e12

    z = (r100 - mu) / np.sqrt(h)
    # standardised Student-t log density (unit variance): scale z by sqrt(nu/(nu-2))
    c = np.sqrt(nu / (nu - 2.0))
    zt = z * c
    ll_ret = (stats.t.logpdf(zt, df=nu) + np.log(c) - 0.5 * np.log(h)).sum()
    # -0.5*log(h): Jacobian of r = mu + sqrt(h) z  ->  dz/dr = 1/sqrt(h)

    obs = np.isfinite(logx_used)
    if obs.sum() < 10:
        return 1e12
    mean_x = xi + phi * np.log(h[obs]) + tau1 * z[obs] + tau2 * (z[obs] ** 2 - 1.0)
    resid_x = logx_used[obs] - mean_x
    ll_meas = stats.norm.logpdf(resid_x, loc=0.0, scale=np.sqrt(sigma_u2)).sum()

    total = ll_ret + ll_meas
    if not np.isfinite(total):
        return 1e12
    return -total


def fit_index(code):
    a = pd.read_csv(os.path.join(ANA, f'{code}_analysis.csv'), parse_dates=['Date'], low_memory=False)
    a = a.sort_values('Date').reset_index(drop=True)
    # own realized-measure history: from the first RV_Valid observation onward, keep every
    # subsequent day's return even where RV_Valid is later False (the NKY gap) - do NOT drop
    # those rows, that is precisely the missing-x_t case this model is built to handle.
    first_valid = a.index[a['RV_Valid'].astype(bool)]
    if len(first_valid) == 0:
        raise RuntimeError(f"{code}: no valid RV observations at all")
    start = first_valid.min()
    sub = a.iloc[start:].reset_index(drop=True)
    sub = sub[sub['Return'].notna()].reset_index(drop=True)

    r100 = sub['Return'].values * 100.0
    rv_scaled = np.where(sub['RV_Valid'].astype(bool), sub['RV_Scaled'], np.nan)
    x100 = rv_scaled * 10000.0  # variance scales as return^2 -> x100 units match r100^2

    h1 = float(np.nanvar(r100[:250])) if np.isfinite(np.nanvar(r100[:250])) else float(np.var(r100))
    h1 = max(h1, 1e-6)

    # starting values: informed by simple moments rather than arbitrary constants
    log_x_obs = np.log(np.maximum(x100[np.isfinite(x100)], 1e-12))
    log_h_naive = np.log(np.maximum(h1, 1e-12))
    theta0 = np.array([
        np.mean(r100), 0.05, 0.55, 0.35,
        float(np.mean(log_x_obs) - log_h_naive), 1.0,
        -0.05, 0.03,
        np.log(max(float(np.var(log_x_obs)) * 0.3, 1e-4)),
        np.log(6.0),
    ])

    res = optimize.minimize(negloglik, theta0, args=(r100, x100, h1),
                             method='Nelder-Mead',
                             options=dict(maxiter=8000, xatol=1e-7, fatol=1e-7))
    # polish with L-BFGS-B from the Nelder-Mead optimum - Nelder-Mead is robust to the awkward
    # likelihood surface at the start, L-BFGS-B refines faster once near the optimum
    res2 = optimize.minimize(negloglik, res.x, args=(r100, x100, h1),
                              method='Nelder-Mead',
                              options=dict(maxiter=4000, xatol=1e-9, fatol=1e-9))
    theta = res2.x if res2.fun < res.fun else res.x
    nll = min(res.fun, res2.fun)

    # numeric Hessian -> standard errors, via central differences on negloglik
    k = len(theta)
    eps = 1e-4
    H = np.zeros((k, k))
    f0 = negloglik(theta, r100, x100, h1)
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i] = eps * max(abs(theta[i]), 1.0)
            ej[j] = eps * max(abs(theta[j]), 1.0)
            fpp = negloglik(theta + ei + ej, r100, x100, h1)
            fpm = negloglik(theta + ei - ej, r100, x100, h1)
            fmp = negloglik(theta - ei + ej, r100, x100, h1)
            fmm = negloglik(theta - ei - ej, r100, x100, h1)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * ei[i] * ej[j] + 1e-300)
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)

    mu, omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2, nu = unpack(theta)
    n = len(r100)
    kpar = k
    aic = 2 * kpar + 2 * nll
    bic = kpar * np.log(n) + 2 * nll

    # ---- reconstruct the full h_t series and standardised residuals at the optimum ----
    logx_used = np.where(np.isfinite(x100), np.log(np.maximum(x100, 1e-12)), np.nan)
    h = np.empty(n)
    h[0] = h1
    rv_imputed = np.zeros(n, dtype=bool)
    for t in range(1, n):
        if np.isfinite(logx_used[t - 1]):
            lx_prev = logx_used[t - 1]
        else:
            lx_prev = np.log(max(h[t - 1], 1e-12))
            rv_imputed[t] = True
        lh = omega + beta * np.log(max(h[t - 1], 1e-12)) + gamma * lx_prev
        h[t] = np.exp(np.clip(lh, -30, 30))
    z = (r100 - mu) / np.sqrt(h)

    params = dict(zip(PNAMES, theta))
    param_row = dict(Code=code, N=n, LogLik=-nll, AIC=aic, BIC=bic,
                      mu_pct=mu, omega=omega, beta=beta, gamma=gamma,
                      xi=xi, phi=phi, tau1=tau1, tau2=tau2,
                      sigma_u2=sigma_u2, nu=nu,
                      Persistence_beta_plus_gammaphi=beta + gamma * phi,
                      RV_Imputed_Days=int(rv_imputed.sum()),
                      Converged=bool(res2.success or res.success))
    for i, name in enumerate(PNAMES):
        param_row[f"se_{name}"] = se[i]

    fit_df = pd.DataFrame({
        'Date': sub['Date'].values,
        'Return': sub['Return'].values,
        'RVProxy': rv_scaled,
        'CondVar_pct2': h,
        'CondVol_pct': np.sqrt(h),
        'CondVar': h / 10000.0,
        'CondVol': np.sqrt(h) / 100.0,
        'StdResid': z,
        'RV_Imputed': rv_imputed,
    })
    return param_row, fit_df, (mu, omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2, nu)


def build_forecast_file(code, fit_df, nu, mu):
    """One-step-ahead forecast at each date t, using CondVol(t) - which, by the recursion,
    is built from information available through t-1 (x_{t-1} and h_{t-1}). This is the same
    "feasible 1-step-ahead" convention used in 27_baseline_garch.py: OriginDate = previous
    trading day, Date = the day the sigma/VaR apply to.
    """
    df = fit_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['OriginDate'] = df['Date'].shift(1)

    sigma = df['CondVol'].values  # decimal scale
    c = np.sqrt(nu / (nu - 2.0))
    mu_dec = mu / 100.0

    out = pd.DataFrame({'Date': df['Date'], 'OriginDate': df['OriginDate']})
    out['SigmaHat'] = sigma
    out['VarHat'] = sigma ** 2
    for k, tau in LEVELS:
        tq = stats.t.ppf(tau, df=nu) / c   # standardised (unit-variance) t quantile
        out[f'VaR_{k}'] = mu_dec + sigma * tq
    for k, tau in [("01", 0.01), ("025", 0.025)]:
        tq = stats.t.ppf(tau, df=nu)  # non-standardised t quantile, for the ES formula
        es_std_t = -(nu + tq ** 2) / ((nu - 1) * tau) * stats.t.pdf(tq, df=nu)
        es_unit_var = es_std_t / c
        out[f'ES_{k}'] = mu_dec + sigma * es_unit_var
    out['Realized'] = df['Return'].values
    out['RVProxy'] = df['RVProxy'].values
    out['Valid'] = out['OriginDate'].notna()
    out['Reason'] = np.where(df['RV_Imputed'].values, 'RV_imputed_in_recursion', '')
    out.loc[~out['Valid'], 'Reason'] = 'no_prior_day_in_estimation_sample'
    # drop the very first row (no OriginDate) rather than carry an invalid, NaN-numeric row -
    # validate() requires the required numerics to be non-NaN even on Valid=False rows only
    # via the schema being present; simplest correct behaviour is to keep it and let SigmaHat
    # carry the h1-based startup value even though it has no OriginDate, marking Valid False.
    out.loc[~out['Valid'], ['VaR_01', 'VaR_025', 'VaR_05', 'ES_01', 'ES_025']] = np.nan
    return out


def main():
    os.makedirs(VAL, exist_ok=True)
    os.makedirs(RVDIR, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(LOG, exist_ok=True)
    os.makedirs(fio.FCDIR, exist_ok=True)
    t0 = time.time()
    rows = []
    log_lines = [f"phase20 Realized GARCH — {pd.Timestamp.now()}", ""]
    fit_series = {}

    for code in CODES:
        print(f"[{code}] fitting Realized GARCH(1,1) log-linear, Student-t returns ...")
        row, fit_df, theta = fit_index(code)
        rows.append(row)
        mu, omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2, nu = theta
        print(f"    N={row['N']}  LL={row['LogLik']:.2f}  AIC={row['AIC']:.2f}  "
              f"beta={beta:.3f} gamma={gamma:.3f} phi={phi:.3f} "
              f"tau1={tau1:.4f} tau2={tau2:.4f} nu={nu:.2f}  "
              f"persistence(beta+gamma*phi)={row['Persistence_beta_plus_gammaphi']:.4f}  "
              f"RV_imputed_days={row['RV_Imputed_Days']}  conv={row['Converged']}")
        log_lines.append(f"[{code}] {row}")

        fit_path = os.path.join(RVDIR, f'{code}_realized_garch_fit.csv')
        fit_df.to_csv(fit_path, index=False, date_format='%Y-%m-%d', float_format='%.10g')
        print(f"    -> wrote {fit_path}")

        fc = build_forecast_file(code, fit_df, nu, mu)
        path = fio.write_forecasts(fc, model="RealGARCH", code=code,
                                    spec="RealGARCH11-logRVscaled-studentt")
        print(f"    -> wrote {path} (contract-validated)")
        fit_series[code] = fit_df.set_index(pd.to_datetime(fit_df['Date']))['CondVol'] * np.sqrt(252) * 100

    params_df = pd.DataFrame(rows)
    params_df.to_csv(os.path.join(VAL, 'realized_garch_params.csv'), index=False)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        for code, s in fit_series.items():
            ax.plot(s.index, s.values, label=code, linewidth=0.8)
        ax.set_title('Realized GARCH(1,1) conditional volatility (annualised %)')
        ax.set_ylabel('Annualised volatility (%)')
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, 'realized_garch_fit.png'), dpi=140)
        plt.close(fig)
        print("wrote 09_FIGURES/realized_garch_fit.png")
    except Exception as e:
        print(f"figure skipped: {e}")

    with open(os.path.join(LOG, 'phase20_realized_garch.log'), 'w') as f:
        f.write("\n".join(log_lines) + f"\n\ndone in {time.time()-t0:.1f}s\n")

    pd.set_option('display.width', 220)
    print()
    print(params_df[['Code', 'N', 'LogLik', 'AIC', 'beta', 'gamma', 'phi', 'tau1', 'tau2',
                      'nu', 'Persistence_beta_plus_gammaphi', 'RV_Imputed_Days']].to_string(index=False))
    print(f"\nDone in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
