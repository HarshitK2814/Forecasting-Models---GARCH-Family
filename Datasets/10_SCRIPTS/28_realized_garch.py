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
                                                                 [or skew-t(eta,lambda), RGARCH-ST]
    GARCH:       log(h_t) = omega + beta*log(h_{t-1}) + gamma*log(x_{t-1})
    measurement: log(x_t) = xi + phi*log(h_t) + tau1*z_t + tau2*(z_t^2 - 1) + sigma_u * eps_t

  x_t is the realized measure (RV_Scaled_Causal - see below). tau1*z_t + tau2*(z_t^2-1) is the
  "leverage function": it lets the realized measure respond asymmetrically to the sign of the
  return shock, which is exactly the Engle-Ng asymmetry the EDA found in every index.

WHY x_t = RV_Scaled_Causal, NOT raw RV
  RV as computed in `01_ANALYSIS_READY` is the SESSION-ONLY realized variance - it excludes
  the overnight return by construction. The Realized GARCH measurement equation, however,
  models h_t, which is the conditional variance of r_t = the CLOSE-TO-CLOSE return. Feeding
  session-only RV into a close-to-close model means x_t is systematically 1.7x-3.0x smaller
  than what it is measuring (Hansen-Lunde scale factors) - the model would then have to force
  xi or phi to absorb a scale mismatch that differs BY INDEX, which contaminates the leverage
  and persistence parameters. RV_Scaled = ScaleFactor_HL * RV removes this by construction; see
  RESEARCHER_A_DECISIONS.md and precaution list.

  RV_Scaled (single full-sample constant c_hl) is look-ahead: an observation dated 2014 was
  scaled using a factor estimated partly from 2026 data. RV_Scaled_Causal fixes this - c_t is
  an EXPANDING factor using only observations strictly before date t (see 15_build_analysis_
  dataset.py DECISION 5). This script uses RV_Scaled_Causal exclusively for every model fit
  (walk-forward and the archived full-sample comparison alike).

WALK-FORWARD PARAMETER ESTIMATION (2026-08-29 fix)
  Previously this script fit theta ONCE on the full sample and reported the daily-recursive
  1-step-ahead series from that single fixed-parameter fit - a genuine "in-sample GARCH"
  convention (state h_t only ever uses x_{t-1}, h_{t-1}, so the RECURSION is look-ahead-free,
  but the PARAMETER VALUES used throughout the whole 2012-2026 sample were fit partly on data
  from 2026). Because Realized GARCH was the volatility-loss winner in every headline QLIKE
  comparison, this was exactly the look-ahead channel the paper's central figure depended on.

  Fixed here by mirroring 29_rolling_forecast_engine.py's design for the GJR benchmark:
  parameters are re-estimated every REFIT_EVERY trading days on an EXPANDING window, and the
  h_t recursion is updated daily between refits using only the most recently fitted theta and
  the real, newly observed x_{t-1}. Annual refitting (REFIT_EVERY=252) is used as the minimum
  defensible cadence named in the execution plan; quarterly was evaluated as computationally
  infeasible in an interactive session (~85s/full-sample optimisation x ~4x the refits x 6
  indices), so annual is the delivered cadence and is disclosed as such, not silently upgraded.

  model="RealGARCH" (the name every downstream evaluation/comparison script reads) now points
  at this walk-forward series - the plan's instruction is that a corrected result REPLACES the
  original rather than being filed as an inconvenient robustness check. The original full-
  sample-parameter fit is still written, under model="RealGARCH_FullSample_INSAMPLE", purely so
  the "did the QLIKE advantage survive the fair rerun" comparison in the paper has a citable
  source; it must never be used as a headline result.

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
  GARCH recursion gets the imputed value, the objective function never sees it as data). The
  same mechanism now also absorbs the RV_Scaled_Causal warm-up window (first ~60 valid RV
  observations of each index's own history, before the causal scale factor is trustworthy).
  Forecast rows in this window are still Valid=True (the return is real) but carry
  Reason="RV_imputed_in_recursion" so B can isolate them for a robustness table.

ESTIMATION
  Quasi-MLE via scipy.optimize.minimize (Nelder-Mead then polish), joint log-likelihood =
  return density (Student-t, or skew-t for RGARCH-ST) + measurement density (Gaussian, as in
  the original paper - u_t is the regression-style error of a log-linear equation, not the
  return itself, so a Gaussian measurement error is the standard and defensible choice).

OUTPUTS
  08_VALIDATION/realized_garch_params.csv          full-sample archive fit: all params + SEs (numeric Hessian), LL, AIC, BIC
  08_VALIDATION/realized_garch_refit_log.csv        walk-forward: every refit date, params, N used
  06_REALIZED_MEASURES/<CODE>_realized_garch_fit.csv   Date, Return, RVProxy, CondVar, CondVol, StdResid, RV_Imputed (walk-forward series)
  20_FORECASTS/RealGARCH__<CODE>_forecasts.csv                     contract-format walk-forward forecasts (PRIMARY, see 26_forecast_io.py)
  20_FORECASTS/RealGARCH_FullSample_INSAMPLE__<CODE>_forecasts.csv contract-format full-sample-parameter forecasts (comparison only, NOT a result)
  20_FORECASTS/RealGARCH_ST__<CODE>_forecasts.csv                  skew-t innovation, walk-forward (robustness only)
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
CODES = os.environ.get("REALGARCH_CODES", "SPX,NDX,UKX,DAX,NKY,HSI").split(",")

REFIT_EVERY = 252   # ~1 trading year; see docstring "WALK-FORWARD PARAMETER ESTIMATION"
BURN_IN = 500        # ~2 trading years before the first walk-forward fit; no forecast rows before this

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
fio = importlib.import_module('26_forecast_io')

LEVELS = fio.LEVELS  # [("01",0.01), ("025",0.025), ("05",0.05)]

# parameter order for the optimiser vector theta (Student-t innovation)
PNAMES = ["mu", "omega", "beta", "gamma", "xi", "phi", "tau1", "tau2", "log_sigma_u2", "log_nu_m2"]
# skew-t innovation: two extra params replace log_nu_m2 (arch.univariate.distribution.SkewStudent)
PNAMES_ST = ["mu", "omega", "beta", "gamma", "xi", "phi", "tau1", "tau2", "log_sigma_u2",
             "logit_eta", "atanh_lambda"]


def unpack(theta):
    mu, omega, beta, gamma, xi, phi, tau1, tau2, log_su2, log_num2 = theta
    sigma_u2 = np.exp(log_su2)
    nu = 2.0 + np.exp(log_num2)   # nu > 2 always, needed for finite variance
    return mu, omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2, nu


def unpack_st(theta):
    mu, omega, beta, gamma, xi, phi, tau1, tau2, log_su2, logit_eta, atanh_lam = theta
    sigma_u2 = np.exp(log_su2)
    # eta (skew-t df) in (2.05, 60); lambda (skew) in (-0.99, 0.99) - both via smooth bijections
    eta = 2.05 + 57.95 / (1.0 + np.exp(-logit_eta))
    lam = 0.99 * np.tanh(atanh_lam)
    return mu, omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2, eta, lam


def _h_recursion(r100, x100, h1, omega, beta, gamma):
    n = len(r100)
    h = np.empty(n)
    h[0] = h1
    logx_used = np.where(np.isnan(x100), np.nan, np.log(np.maximum(x100, 1e-12)))
    rv_imputed = np.zeros(n, dtype=bool)
    for t in range(1, n):
        if np.isfinite(logx_used[t - 1]):
            lx_prev = logx_used[t - 1]
        else:
            lx_prev = np.log(max(h[t - 1], 1e-12))
            rv_imputed[t] = True
        lh = omega + beta * np.log(max(h[t - 1], 1e-12)) + gamma * lx_prev
        lh = np.clip(lh, -30, 30)
        h[t] = np.exp(lh)
        if h[t] <= 0 or not np.isfinite(h[t]):
            return None, None
    return h, rv_imputed


def negloglik(theta, r100, x100, h1):
    """r100: return*100 (percent). x100: realized measure on the matching percent^2 scale,
    NaN where missing. h1: fixed starting conditional variance (percent^2), not estimated -
    conditioning on it is standard practice to avoid an unidentified extra free parameter.
    """
    mu, omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2, nu = unpack(theta)
    if not (0 < beta < 1.2 and -0.5 < gamma < 2.0 and sigma_u2 > 1e-10):
        return 1e12
    h, _ = _h_recursion(r100, x100, h1, omega, beta, gamma)
    if h is None:
        return 1e12

    z = (r100 - mu) / np.sqrt(h)
    c = np.sqrt(nu / (nu - 2.0))
    zt = z * c
    ll_ret = (stats.t.logpdf(zt, df=nu) + np.log(c) - 0.5 * np.log(h)).sum()

    logx_used = np.where(np.isnan(x100), np.nan, np.log(np.maximum(x100, 1e-12)))
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


def negloglik_st(theta, r100, x100, h1, skewstudent):
    """Skew-t return equation (RGARCH-ST robustness variant), otherwise identical to negloglik."""
    mu, omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2, eta, lam = unpack_st(theta)
    if not (0 < beta < 1.2 and -0.5 < gamma < 2.0 and sigma_u2 > 1e-10):
        return 1e12
    h, _ = _h_recursion(r100, x100, h1, omega, beta, gamma)
    if h is None:
        return 1e12

    resid = r100 - mu
    ll_ret = skewstudent.loglikelihood([eta, lam], resid, h, individual=True).sum()
    z = resid / np.sqrt(h)

    logx_used = np.where(np.isnan(x100), np.nan, np.log(np.maximum(x100, 1e-12)))
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


def _theta0(r100, x100, h1, skew=False):
    log_x_obs = np.log(np.maximum(x100[np.isfinite(x100)], 1e-12))
    log_h_naive = np.log(np.maximum(h1, 1e-12))
    base = [np.mean(r100), 0.05, 0.55, 0.35,
            float(np.mean(log_x_obs) - log_h_naive) if log_x_obs.size else 0.0, 1.0,
            -0.05, 0.03,
            np.log(max(float(np.var(log_x_obs)) * 0.3, 1e-4)) if log_x_obs.size else -2.0]
    if skew:
        return np.array(base + [0.0, 0.0])   # logit_eta=0 -> eta~=31; atanh_lambda=0 -> lambda=0
    return np.array(base + [np.log(6.0)])


def fit_theta(r100, x100, h1, skew=False, skewstudent=None, maxiter1=8000, maxiter2=4000):
    """Two-stage Nelder-Mead fit (robust-then-polish) on the given slice. No SE/Hessian here -
    those are only computed for the archived full-sample fit; walk-forward refits (order ~80
    across the 6 indices at annual cadence) would be too slow with a 10x10 numeric Hessian each."""
    theta0 = _theta0(r100, x100, h1, skew=skew)
    fn = negloglik_st if skew else negloglik
    args = (r100, x100, h1, skewstudent) if skew else (r100, x100, h1)
    res = optimize.minimize(fn, theta0, args=args, method='Nelder-Mead',
                             options=dict(maxiter=maxiter1, xatol=1e-7, fatol=1e-7))
    res2 = optimize.minimize(fn, res.x, args=args, method='Nelder-Mead',
                              options=dict(maxiter=maxiter2, xatol=1e-9, fatol=1e-9))
    theta = res2.x if res2.fun < res.fun else res.x
    nll = min(res.fun, res2.fun)
    converged = bool(res2.success or res.success)
    return theta, nll, converged


def load_series(code):
    """Own realized-measure history: from the first RV_Valid observation onward, keep every
    subsequent day's return even where RV_Valid is later False (the NKY gap) - do NOT drop
    those rows, that is precisely the missing-x_t case this model is built to handle. Uses the
    CAUSAL Hansen-Lunde scale (RV_Scaled_Causal), not the full-sample constant (RV_Scaled)."""
    a = pd.read_csv(os.path.join(ANA, f'{code}_analysis.csv'), parse_dates=['Date'], low_memory=False)
    a = a.sort_values('Date').reset_index(drop=True)
    first_valid = a.index[a['RV_Valid'].astype(bool)]
    if len(first_valid) == 0:
        raise RuntimeError(f"{code}: no valid RV observations at all")
    start = first_valid.min()
    sub = a.iloc[start:].reset_index(drop=True)
    sub = sub[sub['Return'].notna()].reset_index(drop=True)

    r100 = sub['Return'].values * 100.0
    rv_valid_causal = sub['RV_Valid'].astype(bool) & sub['ScaleFactor_HL_Causal'].notna()
    rv_scaled = np.where(rv_valid_causal, sub['RV_Scaled_Causal'], np.nan)
    x100 = rv_scaled * 10000.0  # variance scales as return^2 -> x100 units match r100^2
    return sub, r100, x100


def fit_index(code):
    """Full-sample-parameter fit (comparison/archive only - see module docstring). Uses the
    same causal-scaled x_t as the walk-forward fit; only the PARAMETER estimation window
    differs (whole sample vs. expanding-refit)."""
    sub, r100, x100 = load_series(code)
    h1 = float(np.nanvar(r100[:250])) if np.isfinite(np.nanvar(r100[:250])) else float(np.var(r100))
    h1 = max(h1, 1e-6)

    theta, nll, converged = fit_theta(r100, x100, h1)

    k = len(theta)
    eps = 1e-4
    H = np.zeros((k, k))
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
    aic = 2 * k + 2 * nll
    bic = k * np.log(n) + 2 * nll

    h, rv_imputed = _h_recursion(r100, x100, h1, omega, beta, gamma)
    z = (r100 - mu) / np.sqrt(h)

    param_row = dict(Code=code, N=n, LogLik=-nll, AIC=aic, BIC=bic,
                      mu_pct=mu, omega=omega, beta=beta, gamma=gamma,
                      xi=xi, phi=phi, tau1=tau1, tau2=tau2,
                      sigma_u2=sigma_u2, nu=nu,
                      Persistence_beta_plus_gammaphi=beta + gamma * phi,
                      RV_Imputed_Days=int(rv_imputed.sum()),
                      Converged=bool(converged))
    for i, name in enumerate(PNAMES):
        param_row[f"se_{name}"] = se[i]

    fit_df = pd.DataFrame({
        'Date': sub['Date'].values, 'Return': sub['Return'].values, 'RVProxy': x100 / 10000.0,
        'CondVar_pct2': h, 'CondVol_pct': np.sqrt(h), 'CondVar': h / 10000.0,
        'CondVol': np.sqrt(h) / 100.0, 'StdResid': z, 'RV_Imputed': rv_imputed,
        'Mu_pct': mu, 'Nu': nu,
    })
    return param_row, fit_df


def rolling_fit_index(code, refit_every=REFIT_EVERY, burn_in=BURN_IN, skew=False, skewstudent=None):
    """Walk-forward: expanding-window refit every `refit_every` trading days; conditional
    variance is updated every day from the real x_{t-1}, using whichever theta was most
    recently fitted as of that day - mirrors 29_rolling_forecast_engine.py for GJR."""
    sub, r100, x100 = load_series(code)
    n = len(r100)
    burn_in = min(burn_in, max(250, n // 5))
    h1 = float(np.nanvar(r100[:250])) if n >= 250 else float(np.var(r100))
    h1 = max(h1, 1e-6)
    logx_used = np.where(np.isfinite(x100), np.log(np.maximum(x100, 1e-12)), np.nan)

    h = np.empty(n)
    h[0] = h1
    mu_arr = np.full(n, np.nan)
    nu_arr = np.full(n, np.nan)      # for skew=True, holds eta; lam_arr holds lambda
    lam_arr = np.full(n, np.nan)
    rv_imputed = np.zeros(n, dtype=bool)
    theta = None
    refit_rows = []

    def _params(theta):
        if skew:
            return unpack_st(theta)
        m, o, b, g, x_, p, t1, t2, su2, nu = unpack(theta)
        return m, o, b, g, x_, p, t1, t2, su2, nu, np.nan

    for t in range(1, n):
        if t >= burn_in and (theta is None or (t - burn_in) % refit_every == 0):
            theta, nll, conv = fit_theta(r100[:t], x100[:t], h1, skew=skew, skewstudent=skewstudent)
            mu, omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2, nu, lam = _params(theta)
            refit_rows.append(dict(Code=code, Spec='RealGARCH-ST' if skew else 'RealGARCH',
                                    RefitIndex=t, RefitDate=str(sub['Date'].iloc[t - 1].date()),
                                    N_used=t, mu=mu, omega=omega, beta=beta, gamma=gamma,
                                    xi=xi, phi=phi, tau1=tau1, tau2=tau2, sigma_u2=sigma_u2,
                                    nu=nu, lam=lam, NLL=nll, Converged=conv))
        if theta is None:
            h[t] = h1
            continue
        mu, omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2, nu, lam = _params(theta)
        if np.isfinite(logx_used[t - 1]):
            lx_prev = logx_used[t - 1]
        else:
            lx_prev = np.log(max(h[t - 1], 1e-12))
            rv_imputed[t] = True
        lh = omega + beta * np.log(max(h[t - 1], 1e-12)) + gamma * lx_prev
        h[t] = np.exp(np.clip(lh, -30, 30))
        mu_arr[t] = mu
        nu_arr[t] = nu
        lam_arr[t] = lam

    z = (r100 - mu_arr) / np.sqrt(h)
    fit_df = pd.DataFrame({
        'Date': sub['Date'].values, 'Return': sub['Return'].values, 'RVProxy': x100 / 10000.0,
        'CondVar_pct2': h, 'CondVol_pct': np.sqrt(h), 'CondVar': h / 10000.0,
        'CondVol': np.sqrt(h) / 100.0, 'StdResid': z, 'RV_Imputed': rv_imputed,
        'Mu_pct': mu_arr, 'Nu': nu_arr, 'Lambda': lam_arr,
    })
    # no forecast for the burn-in window - no fitted model was available yet
    fit_df = fit_df.iloc[burn_in:].reset_index(drop=True)
    refit_log = pd.DataFrame(refit_rows)
    return fit_df, refit_log


def build_forecast_file(code, fit_df, skew=False):
    """One-step-ahead forecast at each date t, using CondVol(t) - which, by the recursion,
    is built from information available through t-1 (x_{t-1} and h_{t-1}), AND from theta that
    was fitted using only data through t-1 (walk-forward). Mu/Nu/Lambda are per-row because
    walk-forward refits change them over time."""
    df = fit_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['OriginDate'] = df['Date'].shift(1)

    sigma = df['CondVol'].values
    mu_dec = df['Mu_pct'].values / 100.0
    nu = df['Nu'].values

    out = pd.DataFrame({'Date': df['Date'], 'OriginDate': df['OriginDate']})
    out['SigmaHat'] = sigma
    out['VarHat'] = sigma ** 2

    if skew:
        from arch.univariate.distribution import SkewStudent
        sk = SkewStudent()
        lam = df['Lambda'].values
        for k, tau in LEVELS:
            tq = np.array([sk.ppf([tau], [nu[i], lam[i]])[0] if np.isfinite(nu[i]) else np.nan
                            for i in range(len(nu))])
            out[f'VaR_{k}'] = mu_dec + sigma * tq
        # ES via Monte Carlo per unique (nu,lambda) pair - skew-t has no closed-form ES
        rng = np.random.default_rng(20260829)
        cache = {}
        for k, tau in [("01", 0.01), ("025", 0.025)]:
            es_vals = np.full(len(nu), np.nan)
            for i in range(len(nu)):
                if not np.isfinite(nu[i]):
                    continue
                key = (round(nu[i], 3), round(lam[i], 3))
                if key not in cache:
                    sim = sk.simulate([key[0], key[1]])(100000)
                    cache[key] = np.sort(sim)
                sims = cache[key]
                cut = sims[int(tau * len(sims))]
                es_vals[i] = sims[sims <= cut].mean()
            out[f'ES_{k}'] = mu_dec + sigma * es_vals
    else:
        c = np.sqrt(nu / (nu - 2.0))
        for k, tau in LEVELS:
            tq = stats.t.ppf(tau, df=nu) / c
            out[f'VaR_{k}'] = mu_dec + sigma * tq
        for k, tau in [("01", 0.01), ("025", 0.025)]:
            tq = stats.t.ppf(tau, df=nu)
            es_std_t = -(nu + tq ** 2) / ((nu - 1) * tau) * stats.t.pdf(tq, df=nu)
            es_unit_var = es_std_t / c
            out[f'ES_{k}'] = mu_dec + sigma * es_unit_var

    out['Realized'] = df['Return'].values
    out['RVProxy'] = df['RVProxy'].values
    out['Valid'] = out['OriginDate'].notna()
    out['Reason'] = np.where(df['RV_Imputed'].values, 'RV_imputed_in_recursion', '')
    out.loc[~out['Valid'], 'Reason'] = 'no_prior_day_in_estimation_sample'
    out.loc[~out['Valid'], ['VaR_01', 'VaR_025', 'VaR_05', 'ES_01', 'ES_025']] = np.nan
    return out


def main():
    os.makedirs(VAL, exist_ok=True)
    os.makedirs(RVDIR, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(LOG, exist_ok=True)
    os.makedirs(fio.FCDIR, exist_ok=True)
    t0 = time.time()
    archive_rows = []
    all_refits = []
    log_lines = [f"phase20 Realized GARCH — {pd.Timestamp.now()}", ""]
    fit_series = {}

    from arch.univariate.distribution import SkewStudent
    skewstudent = SkewStudent()

    for code in CODES:
        t1 = time.time()
        print(f"[{code}] walk-forward Realized GARCH(1,1), log-linear, Student-t returns (PRIMARY) ...")
        wf_df, refit_log = rolling_fit_index(code)
        all_refits.append(refit_log)
        n_refits = len(refit_log)
        print(f"    {len(wf_df)} forecasts, {n_refits} refits, {time.time()-t1:.1f}s")
        log_lines.append(f"[{code}] walk-forward n={len(wf_df)} refits={n_refits}")

        fit_path = os.path.join(RVDIR, f'{code}_realized_garch_fit.csv')
        wf_df.to_csv(fit_path, index=False, date_format='%Y-%m-%d', float_format='%.10g')
        print(f"    -> wrote {fit_path}")

        fc = build_forecast_file(code, wf_df)
        path = fio.write_forecasts(fc, model="RealGARCH", code=code,
                                    spec=f"RealGARCH11-logRVscaled-causal-studentt-walkforward-refit{REFIT_EVERY}")
        print(f"    -> wrote {path} (contract-validated, PRIMARY)")
        fit_series[code] = wf_df.set_index(pd.to_datetime(wf_df['Date']))['CondVol'] * np.sqrt(252) * 100

        print(f"[{code}] full-sample-parameter Realized GARCH (comparison/archive only) ...")
        t2 = time.time()
        row, full_df = fit_index(code)
        archive_rows.append(row)
        print(f"    N={row['N']}  LL={row['LogLik']:.2f}  AIC={row['AIC']:.2f}  "
              f"beta={row['beta']:.3f} gamma={row['gamma']:.3f} phi={row['phi']:.3f}  "
              f"({time.time()-t2:.1f}s)")
        fc_full = build_forecast_file(code, full_df)
        fio.write_forecasts(fc_full, model="RealGARCH_FullSample_INSAMPLE", code=code,
                             spec="RealGARCH11-logRVscaled-causal-studentt-FULLSAMPLE-NOT-A-RESULT")
        print(f"    -> wrote archive comparison forecast file")

        print(f"[{code}] walk-forward Realized GARCH-ST (skew-t innovation, robustness only) ...")
        t3 = time.time()
        wf_st_df, refit_log_st = rolling_fit_index(code, skew=True, skewstudent=skewstudent)
        all_refits.append(refit_log_st)
        fc_st = build_forecast_file(code, wf_st_df, skew=True)
        fio.write_forecasts(fc_st, model="RealGARCH_ST", code=code,
                             spec=f"RealGARCH11-logRVscaled-causal-skewt-walkforward-refit{REFIT_EVERY}")
        print(f"    -> wrote RealGARCH_ST forecast file ({time.time()-t3:.1f}s, {len(refit_log_st)} refits)")

        # ---- incremental persistence, per index, not just at the very end ----
        # 2026-08-29 fix: this run was killed partway through once already, and the refit-
        # parameter log (unlike the forecast files) was only ever written after the full
        # CODES loop finished, so 4 indices' worth of refit history was silently lost. Append
        # after every index instead, so a kill only ever costs the index in flight.
        refit_log_path = os.path.join(VAL, 'realized_garch_refit_log.csv')
        pd.concat([refit_log, refit_log_st], ignore_index=True).to_csv(
            refit_log_path, mode='a', header=not os.path.exists(refit_log_path), index=False)
        params_path = os.path.join(VAL, 'realized_garch_params.csv')
        pd.DataFrame([row]).to_csv(params_path, mode='a', header=not os.path.exists(params_path),
                                    index=False)
        print(f"    -> appended refit log + params for {code}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        for code, s in fit_series.items():
            ax.plot(s.index, s.values, label=code, linewidth=0.8)
        ax.set_title('Realized GARCH(1,1) walk-forward conditional volatility (annualised %)')
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
    # params_df is read back from disk (appended incrementally per index above), not held in
    # memory - so this summary reflects every index run so far, across resumed invocations too.
    params_path = os.path.join(VAL, 'realized_garch_params.csv')
    if os.path.exists(params_path):
        params_df = pd.read_csv(params_path)
        print(params_df[['Code', 'N', 'LogLik', 'AIC', 'beta', 'gamma', 'phi', 'tau1', 'tau2',
                          'nu', 'Persistence_beta_plus_gammaphi', 'RV_Imputed_Days']].to_string(index=False))
    print(f"\nDone in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
