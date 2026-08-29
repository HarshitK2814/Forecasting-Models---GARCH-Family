# -*- coding: utf-8 -*-
"""
Researcher B - shared estimation and evaluation functions (scripts 41-48).

CONVENTIONS (match RESEARCHER_A_DECISIONS.md section 2)
  - Decimal log returns, never percent.
  - VaR and ES are SIGNED and NEGATIVE. A breach is  Realized < VaR.
  - Inside EVT code only, work on the LOSS scale: L = -z, positive = bad.
    Convert back with  VaR = mu - sigma * gpd_var(...).
  - Diebold-Mariano uses d = loss1 - loss2. NEGATIVE stat = model 1 better.
  - Align on Date, never on row position.
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize

SEED = 42

# ------------------------------------------------------------- EVT dependence diagnostic
def exceedance_dependence(losses, u, max_lag=10):
    """2026-08-29 addition (execution-plan item 12, EVT dependence diagnostics).

    Marginal GPD goodness-of-fit (gpd_gof below) says nothing about whether threshold
    exceedances I_t = 1{L_t > u} are temporally independent. POT/EVT implicitly leans on an
    exceedance process that is close to a Poisson process in the limit; a genuinely clustered
    tail (volatility persists, so a bad day is often followed by another bad day) violates
    that and biases VaR/ES coverage even when xi/beta fit the exceedance sizes well.

    Returns three diagnostics on the exceedance indicator:
      lb_stat/lb_p    Ljung-Box test for autocorrelation in I_t at `max_lag` lags.
      runs_z/runs_p   Wald-Wolfowitz runs test (fewer runs than expected under independence
                      = clustering).
      theta_ferro_segers   Ferro & Segers (2003) intervals estimator of the extremal index
                      theta in [0,1]. theta=1: no clustering (each exceedance is its own
                      "cluster"). theta<1: exceedances cluster; 1/theta is the expected
                      cluster size. This is the number the plan asks be added if declustering
                      needs to be discussed as a robustness check or a limitation.
    """
    L = np.asarray(losses, float)
    I = (L > u).astype(int)
    n_exceed = int(I.sum())
    out = {'n_obs': len(I), 'n_exceed': n_exceed, 'u': float(u)}
    if n_exceed < 10:
        out.update(lb_stat=np.nan, lb_p=np.nan, runs_observed=np.nan, runs_z=np.nan,
                    runs_p=np.nan, theta_ferro_segers=np.nan)
        return out

    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(I, lags=[max_lag], return_df=True)
    out['lb_stat'] = float(lb['lb_stat'].iloc[0])
    out['lb_p'] = float(lb['lb_pvalue'].iloc[0])

    # Wald-Wolfowitz runs test on the binary exceedance sequence: fewer runs than expected
    # under independence is clustering.
    runs = int(1 + np.sum(I[1:] != I[:-1]))
    n1 = int(I.sum()); n0 = len(I) - n1
    if n1 > 0 and n0 > 0 and (n1 + n0) > 1:
        mean_r = 2.0 * n1 * n0 / (n1 + n0) + 1
        var_r = (2.0 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / (((n1 + n0) ** 2) * (n1 + n0 - 1))
        z = (runs - mean_r) / np.sqrt(var_r) if var_r > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    else:
        z, p = np.nan, np.nan
    out['runs_observed'] = runs
    out['runs_z'] = float(z) if np.isfinite(z) else np.nan
    out['runs_p'] = float(p) if np.isfinite(p) else np.nan

    # Ferro-Segers (2003) intervals estimator of the extremal index
    idx = np.where(I == 1)[0]
    T = np.diff(idx)  # inter-exceedance times, in observations, length n_exceed-1
    N = len(T)
    if N < 2:
        theta = np.nan
    elif T.max() <= 2:
        num = 2.0 * (np.sum(T)) ** 2
        den = N * np.sum(T.astype(float) ** 2)
        theta = num / den if den > 0 else np.nan
    else:
        Tm1 = T.astype(float) - 1.0
        num = 2.0 * (np.sum(Tm1)) ** 2
        den = N * np.sum(Tm1 * (T - 2.0))
        theta = num / den if den > 0 else np.nan
    out['theta_ferro_segers'] = float(np.clip(theta, 0.0, 1.0)) if np.isfinite(theta) else np.nan
    return out


# ------------------------------------------------------------------ EVT / GPD
def fit_gpd(losses, threshold_q=0.95):
    """POT GPD fit by MLE. losses: POSITIVE = bad, so pass -z for a left tail."""
    losses = np.asarray(losses, float); losses = losses[np.isfinite(losses)]
    u = np.quantile(losses, threshold_q); ex = losses[losses > u] - u
    if len(ex) < 20:
        raise ValueError(f"only {len(ex)} exceedances above u={u:.4f}")
    def nll(p):
        xi, b = p
        if b <= 0: return 1e10
        if abs(xi) < 1e-8: return len(ex)*np.log(b) + ex.sum()/b
        a = 1 + xi*ex/b
        if np.any(a <= 0): return 1e10
        return len(ex)*np.log(b) + (1 + 1/xi)*np.log(a).sum()
    m, v = ex.mean(), ex.var()
    xi0 = float(np.clip(0.5*(1 - m**2/v), -0.4, 0.4))
    r = minimize(nll, [xi0, max(m*(1-xi0), 1e-6)], method='Nelder-Mead',
                 options={'maxiter': 5000})
    return {'xi': float(r.x[0]), 'beta': float(r.x[1]), 'u': float(u),
            'n_exceed': int(len(ex)), 'n_total': int(len(losses)),
            'converged': bool(r.success)}

def gpd_var(f, q):
    """Quantile q of the loss distribution. Positive, loss scale."""
    tp = (f['n_total']/f['n_exceed'])*(1-q)
    if abs(f['xi']) < 1e-8: return f['u'] + f['beta']*(-np.log(tp))
    return f['u'] + (f['beta']/f['xi'])*(tp**(-f['xi']) - 1)

def gpd_es(f, q):
    """ES beyond gpd_var(f,q). McNeil-Frey closed form."""
    v = gpd_var(f, q)
    return v/(1-f['xi']) + (f['beta'] - f['xi']*f['u'])/(1-f['xi'])

def gpd_gof(losses, f):
    """GPD goodness of fit via the probability integral transform.
    Numerical companion to the QQ plot the Executive Summary section 3.2 asks for.
    If the GPD fits, transformed exceedances are Uniform(0,1)."""
    losses = np.asarray(losses, float); losses = losses[np.isfinite(losses)]
    ex = losses[losses > f['u']] - f['u']
    pit = stats.genpareto.cdf(ex, f['xi'], scale=f['beta'])
    ks = stats.kstest(pit, 'uniform')
    ad = stats.anderson(stats.norm.ppf(np.clip(pit, 1e-9, 1-1e-9)), 'norm')
    return {'n_exceed': int(len(ex)), 'ks_stat': float(ks.statistic),
            'ks_p': float(ks.pvalue), 'ad_stat': float(ad.statistic),
            'ad_crit_5pct': float(ad.critical_values[2]),
            'ad_reject_5pct': bool(ad.statistic > ad.critical_values[2])}

def gpd_qq(losses, f, n_points=None):
    """Empirical vs theoretical exceedance quantiles, for the QQ plot in script 41."""
    losses = np.asarray(losses, float)
    ex = np.sort(losses[losses > f['u']] - f['u']); n = len(ex)
    p = (np.arange(1, n+1) - 0.5)/n
    theo = stats.genpareto.ppf(p, f['xi'], scale=f['beta'])
    if n_points and n > n_points:
        idx = np.unique(np.linspace(0, n-1, n_points).astype(int))
        return theo[idx], ex[idx]
    return theo, ex

# ------------------------------------------------------------- VaR backtests
def kupiec_pof(breach, alpha):
    """Unconditional coverage. Kupiec (1995). LR ~ chi2(1)."""
    v = np.asarray(breach, int); n, x = len(v), int(v.sum())
    pi = x/n if n else np.nan
    if x == 0: lr = -2*n*np.log(1-alpha)
    elif x == n: lr = -2*n*np.log(alpha)
    else:
        ll0 = (n-x)*np.log(1-alpha) + x*np.log(alpha)
        ll1 = (n-x)*np.log(1-pi) + x*np.log(pi); lr = -2*(ll0-ll1)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))

def christoffersen_ind(breach):
    """Independence of breaches. Christoffersen (1998). LR ~ chi2(1)."""
    v = np.asarray(breach, int)
    n00 = int(((v[:-1]==0)&(v[1:]==0)).sum()); n01 = int(((v[:-1]==0)&(v[1:]==1)).sum())
    n10 = int(((v[:-1]==1)&(v[1:]==0)).sum()); n11 = int(((v[:-1]==1)&(v[1:]==1)).sum())
    p01 = n01/(n00+n01) if (n00+n01) else 0
    p11 = n11/(n10+n11) if (n10+n11) else 0
    p = (n01+n11)/max(n00+n01+n10+n11, 1)
    f = lambda pr, k: k*np.log(pr) if (pr > 0 and k > 0) else 0.0
    ll0 = f(1-p, n00+n10) + f(p, n01+n11)
    ll1 = f(1-p01, n00) + f(p01, n01) + f(1-p11, n10) + f(p11, n11)
    lr = max(-2*(ll0-ll1), 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1)), {'n00': n00, 'n01': n01,
                                                          'n10': n10, 'n11': n11}

def christoffersen_cc(breach, alpha):
    """JOINT conditional coverage: LR_cc = LR_uc + LR_ind ~ chi2(2).
    This is what Executive Summary section 4.2 names. Independence alone is a
    different question: a model can pass it while breaching at twice nominal."""
    lr_uc, p_uc = kupiec_pof(breach, alpha)
    lr_ind, p_ind, counts = christoffersen_ind(breach)
    lr_cc = lr_uc + lr_ind
    return {'lr_uc': lr_uc, 'p_uc': p_uc, 'lr_ind': lr_ind, 'p_ind': p_ind,
            'lr_cc': float(lr_cc), 'p_cc': float(1 - stats.chi2.cdf(lr_cc, 2)),
            **counts}

def dq_test(breach, var_f, alpha, lags=4):
    """Dynamic Quantile test. Engle & Manganelli (2004). Stat ~ chi2(2+lags).
    Named in Executive Summary section 4.2. Regresses the demeaned hit series on
    its own lags and the VaR level, so it catches clustering the Christoffersen
    Markov test misses - that test only looks one day back."""
    h = np.asarray(breach, float) - alpha; v = np.asarray(var_f, float); n = len(h)
    if n <= lags + 3: return {'stat': np.nan, 'p': np.nan, 'n': n}
    X = [np.ones(n-lags)]
    for L in range(1, lags+1): X.append(h[lags-L : n-L])
    X.append(v[lags:]); X = np.column_stack(X); y = h[lags:]
    XtX = X.T @ X
    try: b = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError: return {'stat': np.nan, 'p': np.nan, 'n': n}
    stat = float(b @ XtX @ b / (alpha*(1-alpha))); df = X.shape[1]
    return {'stat': stat, 'p': float(1 - stats.chi2.cdf(stat, df)),
            'n': int(n-lags), 'df': int(df)}

def backtest(actual, var, alpha, model_name, index_name=''):
    """One row of the VaR backtest table: coverage, independence, joint CC, DQ."""
    a = np.asarray(actual, float); v = np.asarray(var, float)
    m = np.isfinite(a) & np.isfinite(v); a, v = a[m], v[m]
    br = (a < v).astype(int)
    cc = christoffersen_cc(br, alpha); dq = dq_test(br, v, alpha)
    return {'index': index_name, 'model': model_name, 'confidence': round(1-alpha, 3),
            'n_obs': len(br), 'n_breach': int(br.sum()),
            'expected': round(alpha*len(br), 1),
            'rate_pct': round(100*br.mean(), 3), 'target_pct': round(100*alpha, 2),
            'kupiec_p': round(cc['p_uc'], 4), 'chris_ind_p': round(cc['p_ind'], 4),
            'chris_cc_p': round(cc['p_cc'], 4),
            'dq_p': round(dq['p'], 4) if np.isfinite(dq['p']) else np.nan,
            'n11': cc['n11'], 'pass_kupiec': cc['p_uc'] > 0.05,
            'pass_indep': cc['p_ind'] > 0.05, 'pass_cc': cc['p_cc'] > 0.05,
            'pass_dq': (dq['p'] > 0.05) if np.isfinite(dq['p']) else np.nan}

# ----------------------------------------------------------- loss functions
def qlike_series(actual_var, fc_var):
    a = np.asarray(actual_var, float); f = np.asarray(fc_var, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = a/f; out = r - np.log(r) - 1.0
    out[~np.isfinite(out)] = np.nan; return out

def vol_losses(actual_var, fc_var):
    a = np.asarray(actual_var, float); f = np.asarray(fc_var, float)
    m = np.isfinite(a) & np.isfinite(f) & (a > 0) & (f > 0); a, f = a[m], f[m]
    if len(a) == 0:
        return {k: np.nan for k in ['n','QLIKE','MSE','RMSE','MAE','MAPE']}
    return {'n': len(a), 'QLIKE': float(np.mean(a/f - np.log(a/f) - 1)),
            'MSE': float(np.mean((a-f)**2)), 'RMSE': float(np.sqrt(np.mean((a-f)**2))),
            'MAE': float(np.mean(np.abs(a-f))),
            'MAPE': float(np.mean(np.abs((a-f)/a))*100)}

def pinball_loss(actual, var_f, alpha):
    """Proper scoring rule for a quantile forecast. Minimised at the true quantile."""
    u = np.asarray(actual, float) - np.asarray(var_f, float)
    return u*(alpha - (u < 0).astype(float))

# ------------------------------------------------------- model comparison
def diebold_mariano(loss1, loss2, h=1, harvey=True, hac=True, max_lag=None):
    """NEGATIVE statistic => model 1 has the lower loss.

    2026-08-29 fix: the previous version's serial-correlation correction only ran for lag in
    range(1, h), which is a no-op whenever h=1 - the DM statistic silently used the plain
    single-period variance (as if the loss differential were white noise) for every 1-step-
    ahead comparison in this project, i.e. every DM call actually made. Loss differentials can
    be serially correlated even at h=1 (e.g. via persistence in the underlying loss series
    itself), so hac=True (default) now always applies a Newey-West/Bartlett-kernel long-run
    variance, with lag truncation L = max(h-1, floor(4*(n/100)^(2/9))) - the Newey-West (1994)
    plug-in bandwidth, floored so it still nests the classical h-step overlap correction for
    h>1. Set hac=False to recover the old (pre-fix) behaviour for comparison.
    """
    l1 = np.asarray(loss1, float); l2 = np.asarray(loss2, float)
    m = np.isfinite(l1) & np.isfinite(l2); l1, l2 = l1[m], l2[m]; n = len(l1)
    if n < 20: return {'stat': np.nan, 'p': np.nan, 'n': n, 'mean_diff': np.nan, 'lag': 0}
    d = l1 - l2; dbar = d.mean(); g0 = np.mean((d-dbar)**2); lrv = g0
    if hac:
        L = max(h - 1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
        L = min(L, n - 2)
        for lag in range(1, L + 1):
            w = 1.0 - lag / (L + 1.0)   # Bartlett kernel weight
            lrv += 2 * w * np.mean((d[lag:] - dbar) * (d[:-lag] - dbar))
    else:
        L = h - 1
        for lag in range(1, h): lrv += 2*np.mean((d[lag:]-dbar)*(d[:-lag]-dbar))
    if lrv <= 0: lrv = g0
    if lrv == 0: return {'stat': np.nan, 'p': np.nan, 'n': n, 'mean_diff': 0.0, 'lag': L}
    stat = dbar/np.sqrt(lrv/n)
    if harvey and n > h:
        stat *= np.sqrt((n + 1 - 2*h + h*(h-1)/n)/n)
        p = 2*(1 - stats.t.cdf(abs(stat), n-1))
    else:
        p = 2*(1 - stats.norm.cdf(abs(stat)))
    return {'stat': float(stat), 'p': float(p), 'n': n, 'mean_diff': float(dbar), 'lag': int(L)}

def es_ratio(returns, var_f, es_f):
    """Realised mean loss on breach days / predicted ES on those days.
    1.05 means realised tail losses were 5% deeper than predicted. Report this
    alongside Z2, which is only a p-value."""
    r = np.asarray(returns, float); v = np.asarray(var_f, float); e = np.asarray(es_f, float)
    m = np.isfinite(r) & np.isfinite(v) & np.isfinite(e); r, v, e = r[m], v[m], e[m]
    br = r < v
    if br.sum() == 0: return {'ratio': np.nan, 'n_breach': 0}
    return {'ratio': float(r[br].mean()/e[br].mean()), 'n_breach': int(br.sum()),
            'mean_realised': float(r[br].mean()), 'mean_predicted_es': float(e[br].mean())}

def acerbi_szekely_z2(returns, var_f, es_f, alpha, n_sim=5000, seed=SEED):
    """Acerbi-Szekely (2014) Z2 test for expected shortfall.
    The null is scaled off the VaR forecast, NOT off ES. Scaling off ES (an
    earlier version did) treats ES as if it were the alpha-quantile; since
    |ES| > |VaR| the simulated series then breaches far more than alpha, the null
    distribution is too wide, and the test is too slow to reject."""
    r = np.asarray(returns, float); v = np.asarray(var_f, float); e = np.asarray(es_f, float)
    m = np.isfinite(r) & np.isfinite(v) & np.isfinite(e)
    r, v, e = r[m], v[m], e[m]; n = len(r)
    if n == 0: return {'Z2': np.nan, 'p': np.nan, 'n_breach': 0}
    ind = (r < v).astype(float)
    z2 = float(np.sum(r*ind/(n*alpha*e)) + 1.0)
    rng = np.random.default_rng(seed)
    sig = np.abs(v)/max(abs(stats.norm.ppf(alpha)), 1e-6)
    null = np.empty(n_sim)
    for s in range(n_sim):
        sim = rng.standard_normal(n)*sig
        null[s] = np.sum(sim*(sim < v)/(n*alpha*e)) + 1.0
    return {'Z2': z2, 'p': float(np.mean(null <= z2)),
            'n_breach': int(ind.sum()), 'n': n}

def model_confidence_set(loss_dict, alpha=0.10, n_boot=1500, block=10, seed=SEED):
    """Hansen-Lunde-Nason Model Confidence Set, stationary block bootstrap."""
    rng = np.random.default_rng(seed); names = list(loss_dict)
    L = np.column_stack([np.asarray(loss_dict[k], float) for k in names])
    L = L[np.all(np.isfinite(L), axis=1)]; n, m = L.shape
    if m < 2 or n < 50: return {'mcs': names, 'eliminated': [], 'p': {}, 'n': n}
    idxs = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        ii, i = np.empty(n, dtype=int), 0
        while i < n:
            s = rng.integers(0, n); ln = min(rng.geometric(1/block), n-i)
            ii[i:i+ln] = (s + np.arange(ln)) % n; i += ln
        idxs[b] = ii
    surv, elim, pv = list(range(m)), [], {}
    while len(surv) > 1:
        sub = L[:, surv]; k = len(surv); means = sub.mean(0)
        T = np.zeros((k, k)); BT = np.zeros((n_boot, k, k))
        for i in range(k):
            for j in range(i+1, k):
                d = sub[:, i] - sub[:, j]; db = d.mean()
                bm = d[idxs].mean(1); se = np.sqrt(max(bm.var(), 1e-14))
                T[i, j] = db/se; T[j, i] = -T[i, j]
                BT[:, i, j] = (bm-db)/se; BT[:, j, i] = -BT[:, i, j]
        tmax = np.abs(T).max()
        p = float((np.abs(BT).max((1, 2)) >= tmax).mean())
        if p >= alpha: break
        w = int(np.argmax(means)); elim.append(names[surv[w]])
        pv[names[surv[w]]] = p; surv.pop(w)
    for i in surv: pv.setdefault(names[i], 1.0)
    return {'mcs': [names[i] for i in surv], 'eliminated': elim, 'p': pv, 'n': n}
