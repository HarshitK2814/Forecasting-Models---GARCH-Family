# -*- coding: utf-8 -*-
"""
Integration tests for Datasets/10_SCRIPTS/34_causal_evt_residuals.py.

Unlike the other test files, these run against the real committed data
(01_ANALYSIS_READY, 08_VALIDATION/rolling_engine_refit_log.csv) rather than
synthetic fixtures, because the whole point of this module is "reconstruct
exactly what 29_rolling_forecast_engine.py already computed" - a synthetic
GARCH series would not exercise that property. Still fast: only a handful of
`.fix()` calls (no optimisation) against one index.
"""
import os
import sys

import numpy as np
import pandas as pd
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


HAS_DATA = os.path.exists(os.path.join(SCRIPTS, "..", "01_ANALYSIS_READY", "SPX_analysis.csv")) and \
    os.path.exists(os.path.join(SCRIPTS, "..", "08_VALIDATION", "rolling_engine_refit_log.csv"))

pytestmark = pytest.mark.skipif(not HAS_DATA, reason="requires committed analysis-ready data + refit log")

causal = _load("causal", "34_causal_evt_residuals.py") if HAS_DATA else None


def test_no_refit_before_first_block_returns_none():
    src = causal.CausalResidualSource("SPX")
    before_first = src.log["RefitDate"].iloc[0] - pd.Timedelta(days=365 * 20)
    assert src.residual_history_as_of(before_first) is None
    assert np.isnan(src.mu_forecast(before_first, 0.01, -0.03))


def test_residual_history_never_extends_past_origin():
    """The core causality property: every residual date returned must be on or
    before the refit that's active at origin_date, which is itself <= origin_date."""
    src = causal.CausalResidualSource("SPX")
    for origin in src.log["RefitDate"].iloc[[0, len(src.log) // 2, -1]]:
        hist = src.residual_history_as_of(origin)
        assert hist is not None
        assert hist.index.max() <= origin


def test_residual_history_matches_reported_n_for_its_block():
    """fit consistency: the reconstructed training window length must equal
    what 29_rolling_forecast_engine.py itself recorded for that refit (this is
    also asserted inside the module and would raise if it drifted)."""
    src = causal.CausalResidualSource("SPX")
    block = src.log.iloc[10]
    hist = src._hist_for_block(block)
    assert len(hist) == int(block["N"])


def test_later_blocks_use_more_history_than_earlier_blocks():
    src = causal.CausalResidualSource("SPX")
    n_first = int(src.log.iloc[0]["N"])
    n_last = int(src.log.iloc[-1]["N"])
    assert n_last > n_first  # expanding window


def test_mu_forecast_matches_direct_arch_forecast():
    """Regression pin for the algebraic Mu recovery: cross-checked once against
    a direct `.fix(theta).forecast(horizon=1)` call to float precision (see
    34_causal_evt_residuals.py's docstring); this test re-derives that check at
    a different origin so it isn't just re-asserting a hard-coded number."""
    from arch import arch_model

    src = causal.CausalResidualSource("SPX")
    fio = _load("fio", "26_forecast_io.py")
    base = fio.read_forecasts(os.path.join(SCRIPTS, "..", "20_FORECASTS", "GJR-skewt__SPX_forecasts.csv"))
    row = base[base["Valid"]].iloc[300]
    origin, sigma, var01 = row["OriginDate"], float(row["SigmaHat"]), float(row["VaR_01"])

    mu_algebra = src.mu_forecast(origin, sigma, var01)

    block = src._block_for(origin)
    theta = src._theta(block)
    train = src.r.loc[:origin]
    am = arch_model(train, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="skewt")
    f = am.fix(theta).forecast(horizon=1, reindex=False)
    mu_true = float(f.mean.values[-1, 0]) / 100.0

    assert mu_algebra == pytest.approx(mu_true, abs=1e-8)


def test_caching_returns_identical_object_for_same_refit():
    src = causal.CausalResidualSource("SPX")
    origin_a = src.log["RefitDate"].iloc[5]
    origin_b = origin_a + pd.Timedelta(days=3)  # still within the same block
    h1 = src.residual_history_as_of(origin_a)
    h2 = src.residual_history_as_of(origin_b)
    pd.testing.assert_series_equal(h1, h2)
