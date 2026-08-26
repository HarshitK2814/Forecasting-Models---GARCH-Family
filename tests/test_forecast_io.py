# -*- coding: utf-8 -*-
"""
Unit tests for the forecast file contract, Datasets/10_SCRIPTS/26_forecast_io.py.

This is the single interface between Researcher A (writes) and Researcher B
(reads) — the file every downstream table/figure ultimately depends on. These
tests build a minimal valid frame and then poke each contract rule to confirm
validate() actually catches the violation it claims to.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "Datasets", "10_SCRIPTS")
sys.path.insert(0, SCRIPTS)


def _load():
    spec = importlib.util.spec_from_file_location("fio", os.path.join(SCRIPTS, "26_forecast_io.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fio = _load()


def _valid_frame(n=30):
    """A minimal frame that should satisfy every rule in validate()."""
    dates = pd.bdate_range("2020-01-02", periods=n + 1)
    df = pd.DataFrame({
        "Date": dates[1:],
        "OriginDate": dates[:-1],
        "Code": "SPX",
        "Model": "GJR-skewt",
        "Spec": "GJR-skewt-expanding",
        "Horizon": 1,
        "SigmaHat": np.full(n, 0.01),
        "VarHat": np.full(n, 0.01 ** 2),
        "VaR_01": np.full(n, -0.03),
        "VaR_025": np.full(n, -0.025),
        "VaR_05": np.full(n, -0.02),
        "ES_01": np.full(n, -0.04),
        "ES_025": np.full(n, -0.035),
        "Realized": np.full(n, 0.001),
        "RVProxy": np.full(n, 0.0001),
        "Valid": True,
        "Reason": "",
    })
    return df[fio.COLS]


def test_valid_frame_passes():
    df = _valid_frame()
    assert fio.validate(df, strict=True) == []


def test_missing_column_raises():
    df = _valid_frame().drop(columns=["SigmaHat"])
    with pytest.raises(fio.ForecastContractError):
        fio.validate(df, strict=True)


def test_lookahead_violation_caught():
    """The contract's core rule: OriginDate must strictly precede Date."""
    df = _valid_frame()
    df.loc[0, "OriginDate"] = df.loc[0, "Date"]  # OriginDate == Date, a leak
    with pytest.raises(fio.ForecastContractError, match="LOOK-AHEAD"):
        fio.validate(df, strict=True)


def test_lookahead_future_origin_caught():
    df = _valid_frame()
    df.loc[0, "OriginDate"] = df.loc[0, "Date"] + pd.Timedelta(days=1)
    with pytest.raises(fio.ForecastContractError, match="LOOK-AHEAD"):
        fio.validate(df, strict=True)


def test_var_sign_convention_caught():
    """VaR must be negative. A positive VaR_05 is the classic sign-flip bug."""
    df = _valid_frame()
    df.loc[0, "VaR_05"] = 0.02  # wrong sign
    with pytest.raises(fio.ForecastContractError, match="SIGNED, NEGATIVE"):
        fio.validate(df, strict=True)


def test_var_ordering_violation_caught():
    """VaR_01 <= VaR_025 <= VaR_05 must hold (deeper quantile = more negative)."""
    df = _valid_frame()
    df.loc[0, "VaR_01"] = -0.001  # now less negative than VaR_025, violates ordering
    with pytest.raises(fio.ForecastContractError, match="out of order"):
        fio.validate(df, strict=True)


def test_es_beyond_var_violation_caught():
    """ES must be at least as extreme (<=) as VaR at the same level."""
    df = _valid_frame()
    df.loc[0, "ES_01"] = -0.01  # less negative than VaR_01 (-0.03) -- wrong direction
    with pytest.raises(fio.ForecastContractError, match="ES_01 > VaR_01"):
        fio.validate(df, strict=True)


def test_varhat_must_equal_sigmahat_squared():
    df = _valid_frame()
    df.loc[0, "VarHat"] = 999.0
    with pytest.raises(fio.ForecastContractError, match="VarHat != SigmaHat"):
        fio.validate(df, strict=True)


def test_nonpositive_sigmahat_caught():
    df = _valid_frame()
    df.loc[0, "SigmaHat"] = -0.01
    with pytest.raises(fio.ForecastContractError, match="SigmaHat <= 0"):
        fio.validate(df, strict=True)


def test_duplicate_date_caught():
    df = _valid_frame()
    df.loc[1, "Date"] = df.loc[0, "Date"]
    with pytest.raises(fio.ForecastContractError, match="duplicate Date"):
        fio.validate(df, strict=True)


def test_unknown_code_caught():
    df = _valid_frame()
    df["Code"] = "ZZZ"
    with pytest.raises(fio.ForecastContractError, match="unknown Code"):
        fio.validate(df, strict=True)


def test_non_strict_returns_problem_list_without_raising():
    df = _valid_frame()
    df.loc[0, "VaR_05"] = 0.02
    problems = fio.validate(df, strict=False)
    assert isinstance(problems, list) and len(problems) > 0


def test_invalid_rows_exempt_from_required_numeric_check():
    """Rule 5: a Valid=False row may carry NaN numerics without failing the
    contract -- that's the whole point of 'no-forecast days are rows, not
    absences'."""
    df = _valid_frame()
    df.loc[0, ["SigmaHat", "VarHat", "VaR_01", "VaR_025", "VaR_05"]] = np.nan
    df.loc[0, "Valid"] = False
    df.loc[0, "Reason"] = "no data"
    assert fio.validate(df, strict=True) == []
