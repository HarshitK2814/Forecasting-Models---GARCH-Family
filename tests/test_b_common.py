# -*- coding: utf-8 -*-
"""
Unit tests for Datasets/10_SCRIPTS/40_b_common.py — the shared EVT/backtest/
loss-function library used by scripts 41-49.

Executive Summary, "Reproducibility & Environment": "Write unit tests for core
functions (e.g. GPD fit, quantile regression prediction)." This file (plus
test_forecast_io.py and test_basel_zone.py) closes that item.

Run with:  pytest tests/ -v
"""
import os
import sys

import numpy as np
import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "Datasets", "10_SCRIPTS")
sys.path.insert(0, SCRIPTS)

import importlib.util


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bc = _load("bc", "40_b_common.py")


# --------------------------------------------------------------------- GPD

def test_fit_gpd_recovers_known_parameters():
    """Simulate a pure GPD(xi=0.15, beta=1.0) sample above u=2.0 and confirm
    fit_gpd recovers parameters close to the generating ones (MLE consistency
    check, not a bit-exact check). threshold_q=0.01 keeps ~99% of the sample
    as exceedances, so the fit is effectively re-estimating the generating
    distribution directly."""
    rng = np.random.default_rng(0)
    from scipy.stats import genpareto
    true_xi, true_beta, u = 0.15, 1.0, 2.0
    exceed = genpareto.rvs(true_xi, scale=true_beta, size=3000, random_state=rng)
    losses = exceed + u
    f = bc.fit_gpd(losses, threshold_q=0.01)
    assert f["converged"]
    assert abs(f["xi"] - true_xi) < 0.05
    assert abs(f["beta"] - true_beta) < 0.15


def test_fit_gpd_raises_on_too_few_exceedances():
    losses = np.random.default_rng(1).normal(0, 1, 50)
    with pytest.raises(ValueError):
        bc.fit_gpd(losses, threshold_q=0.999)  # far too few points above u


def test_gpd_var_reduces_to_exponential_when_xi_zero():
    """xi=0 is the exponential-tail special case: VaR_q = u - beta*log(1-q_eff)."""
    f = {"xi": 0.0, "beta": 2.0, "u": 1.0, "n_exceed": 100, "n_total": 1000}
    q = 0.99
    tp = (f["n_total"] / f["n_exceed"]) * (1 - q)
    expected = f["u"] + f["beta"] * (-np.log(tp))
    assert bc.gpd_var(f, q) == pytest.approx(expected)


def test_gpd_var_monotonic_in_q():
    f = {"xi": 0.1, "beta": 1.5, "u": 1.0, "n_exceed": 150, "n_total": 3000}
    v95 = bc.gpd_var(f, 0.95)
    v99 = bc.gpd_var(f, 0.99)
    v999 = bc.gpd_var(f, 0.999)
    assert v95 < v99 < v999  # deeper quantile -> larger loss


def test_gpd_es_exceeds_gpd_var():
    """ES must lie beyond VaR on the loss scale for a valid (xi<1) tail fit."""
    f = {"xi": 0.15, "beta": 1.2, "u": 1.0, "n_exceed": 120, "n_total": 3000}
    v = bc.gpd_var(f, 0.99)
    e = bc.gpd_es(f, 0.99)
    assert e > v


def test_gpd_gof_uniform_pit_on_true_model():
    """Exceedances drawn from the fitted GPD itself should pass the KS test at
    a conventional level almost always — a sanity check that gpd_gof's PIT
    machinery is wired correctly (not a statistical guarantee, so seeded)."""
    rng = np.random.default_rng(7)
    from scipy.stats import genpareto
    xi, beta, u = 0.1, 1.0, 0.0
    ex = genpareto.rvs(xi, scale=beta, size=500, random_state=rng)
    losses = ex + u
    f = {"xi": xi, "beta": beta, "u": u, "n_exceed": len(ex), "n_total": len(ex)}
    g = bc.gpd_gof(losses, f)
    assert g["ks_p"] > 0.01


# --------------------------------------------------------------- VaR backtests

def test_kupiec_pof_matches_closed_form_all_breach_and_none():
    # all breaches: LR = -2n*log(alpha)
    n, alpha = 100, 0.01
    lr, p = bc.kupiec_pof(np.ones(n), alpha)
    assert lr == pytest.approx(-2 * n * np.log(alpha))
    # no breaches: LR = -2n*log(1-alpha)
    lr0, p0 = bc.kupiec_pof(np.zeros(n), alpha)
    assert lr0 == pytest.approx(-2 * n * np.log(1 - alpha))


def test_kupiec_pof_zero_at_nominal_rate():
    """If the empirical breach rate equals alpha exactly, LR should be ~0
    (pi_hat == alpha collapses ll0 == ll1)."""
    n, alpha = 1000, 0.05
    breach = np.zeros(n, dtype=int)
    breach[: int(n * alpha)] = 1
    lr, p = bc.kupiec_pof(breach, alpha)
    assert lr == pytest.approx(0.0, abs=1e-6)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_christoffersen_ind_no_clustering_gives_small_stat():
    """Alternating breach pattern (perfectly independent, no two breaches ever
    adjacent) should not be flagged for clustering the way an all-consecutive
    pattern would."""
    n = 200
    breach_scattered = np.zeros(n, dtype=int)
    breach_scattered[::10] = 1  # evenly spaced, never adjacent
    breach_clustered = np.zeros(n, dtype=int)
    breach_clustered[50:70] = 1  # 20 in a row
    lr_s, p_s, _ = bc.christoffersen_ind(breach_scattered)
    lr_c, p_c, _ = bc.christoffersen_ind(breach_clustered)
    assert lr_c > lr_s  # clustering must score higher than scattering at equal breach count


def test_christoffersen_cc_equals_sum_of_uc_and_ind():
    breach = np.random.default_rng(2).binomial(1, 0.05, 500)
    alpha = 0.05
    cc = bc.christoffersen_cc(breach, alpha)
    lr_uc, _ = bc.kupiec_pof(breach, alpha)
    lr_ind, _, _ = bc.christoffersen_ind(breach)
    assert cc["lr_cc"] == pytest.approx(lr_uc + lr_ind)


def test_dq_test_short_series_returns_nan():
    breach = np.zeros(5, dtype=int)
    var_f = np.zeros(5)
    out = bc.dq_test(breach, var_f, alpha=0.05, lags=4)
    assert np.isnan(out["stat"])


def test_backtest_row_internal_consistency():
    rng = np.random.default_rng(3)
    n = 2000
    actual = rng.standard_normal(n) * 0.01
    var_f = np.full(n, -0.02)  # 1% VaR level, roughly
    row = bc.backtest(actual, var_f, alpha=0.01, model_name="X", index_name="TEST")
    assert row["n_obs"] == n
    assert row["n_breach"] == int((actual < var_f).sum())
    assert row["rate_pct"] == pytest.approx(100 * row["n_breach"] / n, abs=1e-6)


# ------------------------------------------------------------------- losses

def test_qlike_series_zero_at_perfect_forecast():
    a = np.array([1.0, 2.0, 3.0])
    out = bc.qlike_series(a, a)
    assert np.allclose(out, 0.0, atol=1e-10)


def test_qlike_series_positive_away_from_truth():
    a = np.array([1.0])
    out_over = bc.qlike_series(a, np.array([2.0]))
    out_under = bc.qlike_series(a, np.array([0.5]))
    assert out_over[0] > 0
    assert out_under[0] > 0


def test_vol_losses_matches_manual_rmse():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    f = np.array([1.1, 1.9, 3.2, 3.8])
    out = bc.vol_losses(a, f)
    assert out["RMSE"] == pytest.approx(np.sqrt(np.mean((a - f) ** 2)))
    assert out["MAE"] == pytest.approx(np.mean(np.abs(a - f)))


def test_pinball_loss_minimised_at_true_quantile():
    """Proper scoring rule property: expected pinball loss is minimised when
    var_f equals the true alpha-quantile of a known distribution."""
    rng = np.random.default_rng(4)
    alpha = 0.05
    sample = rng.standard_normal(200000)
    true_q = np.quantile(sample, alpha)
    losses_at_true = bc.pinball_loss(sample, np.full_like(sample, true_q), alpha).mean()
    for off in (-0.3, 0.3, 1.0):
        losses_off = bc.pinball_loss(sample, np.full_like(sample, true_q + off), alpha).mean()
        assert losses_off >= losses_at_true - 1e-6


def test_pinball_loss_sign_convention():
    # actual below var_f (a "bad" quantile miss on the loss side) contributes (alpha-1)*u < 0 handling
    actual = np.array([-0.05])
    var_f = np.array([-0.02])  # actual breached (actual < var_f)
    alpha = 0.05
    u = actual - var_f  # negative
    expected = u * (alpha - 1.0)
    assert bc.pinball_loss(actual, var_f, alpha)[0] == pytest.approx(expected[0])


# ----------------------------------------------------------- model comparison

def test_diebold_mariano_identical_series_gives_zero_stat():
    rng = np.random.default_rng(5)
    loss = rng.standard_normal(500) ** 2
    out = bc.diebold_mariano(loss, loss)
    assert out["mean_diff"] == pytest.approx(0.0)
    assert np.isnan(out["stat"]) or out["stat"] == pytest.approx(0.0, abs=1e-8)


def test_diebold_mariano_sign_reflects_which_model_is_better():
    rng = np.random.default_rng(6)
    n = 2000
    loss1 = rng.standard_normal(n) * 0.1 + 1.0   # mean loss 1.0
    loss2 = rng.standard_normal(n) * 0.1 + 2.0   # mean loss 2.0, strictly worse
    out = bc.diebold_mariano(loss1, loss2)
    assert out["stat"] < 0  # model 1 (loss1) is better -> negative statistic
    assert out["p"] < 0.01  # difference of 1.0 vs noise 0.1 should be decisive


def test_diebold_mariano_too_few_obs_returns_nan():
    out = bc.diebold_mariano(np.arange(5), np.arange(5) + 1)
    assert np.isnan(out["stat"])


def test_es_ratio_no_breaches_is_nan():
    r = np.array([0.01, 0.02, -0.001])
    v = np.array([-0.05, -0.05, -0.05])  # nothing breaches this deep VaR
    e = np.array([-0.08, -0.08, -0.08])
    out = bc.es_ratio(r, v, e)
    assert np.isnan(out["ratio"])
    assert out["n_breach"] == 0


def test_model_confidence_set_trivial_case_returns_all_when_too_few_obs():
    out = bc.model_confidence_set({"A": np.ones(10), "B": np.ones(10)}, n_boot=10)
    assert out["mcs"] == ["A", "B"]
    assert out["eliminated"] == []


def test_model_confidence_set_eliminates_clearly_worse_model():
    rng = np.random.default_rng(8)
    n = 500
    loss_good = rng.standard_normal(n) * 0.05 + 1.0
    loss_bad = rng.standard_normal(n) * 0.05 + 3.0  # much worse, low noise
    out = bc.model_confidence_set({"good": loss_good, "bad": loss_bad}, n_boot=200, seed=1)
    assert "bad" in out["eliminated"]
    assert "good" in out["mcs"]
