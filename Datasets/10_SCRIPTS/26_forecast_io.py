# -*- coding: utf-8 -*-
"""
FORECAST FILE CONTRACT - the single interface between Researcher A and Researcher B.

WHY THIS EXISTS
  A produces forecasts (baseline GARCH, GJR/EGARCH, Realized GARCH, and the rolling engine).
  B consumes them (evaluation metrics, VaR backtests, Diebold-Mariano, Model Confidence Set).
  If the two sides invent their own formats, one of them rewrites. This module IS the format:
  both sides import it rather than agreeing in prose and drifting.

  A writes with   write_forecasts(df, model, code)
  B reads with    read_forecasts(path)   ->  already validated, correct dtypes

  Anything read_forecasts() accepts is guaranteed to satisfy every rule in CONVENTIONS below,
  because it re-runs validate() on load. A malformed file raises rather than silently
  producing a wrong backtest.

CONVENTIONS - these are the decisions, not suggestions
  1. UNITS. Every return-space number is a DECIMAL LOG RETURN, never a percent.
     A 1% move is 0.01. SigmaHat is a standard deviation on that scale.

  2. SCALE. SigmaHat is the conditional standard deviation of the CLOSE-TO-CLOSE daily log
     return - the same object as `Return` in the analysis files. It is NOT session-only
     volatility. A Realized GARCH fitted on raw RV produces a session-scale quantity and must
     be converted before it lands here; see ScaleFactor_HL and precaution 3 below.

  3. VaR SIGN. VaR is the SIGNED QUANTILE of the return distribution, so it is NEGATIVE.
     VaR_01 is the 1st percentile. A breach is `Realized < VaR_01`. There is no sign flip
     anywhere in the pipeline and no "loss is positive" convention. Ordering must hold:
         VaR_01 <= VaR_025 <= VaR_05 < 0

  4. DATE = TARGET. `Date` is the day being forecast. `OriginDate` is the last day whose data
     entered the information set. For a 1-step-ahead forecast OriginDate is the previous
     trading day of THAT index. This makes look-ahead auditable: any row where
     OriginDate >= Date is a leak, and validate() rejects it.

  5. NO-FORECAST DAYS ARE ROWS, NOT ABSENCES. When a model cannot produce a forecast - the
     Nikkei has no valid RV in 2016-17, so Realized GARCH cannot - the row is still present
     with Valid=False, NaN numerics, and a Reason string. Deleting the row instead would make
     two models' loss series different lengths and silently corrupt Diebold-Mariano.
     B must filter on Valid, never assume the file is dense.

  6. EVALUATION WINDOW. Rows span each index's InSample_B days. Estimation may use history
     from 1990; only the forecast window is common. Pooled tests (MCS) additionally restrict
     to CommonDate_B. See RESEARCHER_A_DECISIONS.md section 3.

  7. ONE FILE PER (MODEL, INDEX). Filename `<MODEL>__<CODE>_forecasts.csv`, double underscore
     as the separator, because model names contain single hyphens (GARCH-EVT).

Output layout:
    20_FORECASTS/<MODEL>__<CODE>_forecasts.csv
    20_FORECASTS/FORECAST_SCHEMA.csv          machine-readable column spec
    20_FORECASTS/_SYNTHETIC/                  placeholder files in this exact format, so B can
                                              develop and unit-test before A's real ones exist
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
FCDIR = os.path.join(ROOT, '20_FORECASTS')

CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

# tau levels carried in every file. Fixed - B's backtest code indexes on these names.
LEVELS = [("01", 0.01), ("025", 0.025), ("05", 0.05)]

# (name, dtype, required, description)
SCHEMA = [
    ("Date",        "date",  True,  "Target date. The day being forecast. Sorted, unique."),
    ("OriginDate",  "date",  True,  "Last date in the information set. Must be < Date."),
    ("Code",        "str",   True,  "Index code: SPX NDX UKX DAX NKY HSI."),
    ("Model",       "str",   True,  "Model family: GARCH GJR EGARCH RealGARCH GARCH-EVT QR."),
    ("Spec",        "str",   True,  "Variant label, e.g. 'GJR-skewt-expanding'. Free text, stable within a file."),
    ("Horizon",     "int",   True,  "Forecast horizon in trading days. 1 for the main results."),
    ("SigmaHat",    "float", True,  "Conditional SD of the close-to-close daily log return. Decimal, > 0."),
    ("VarHat",      "float", True,  "SigmaHat ** 2. Redundant but carried so QLIKE needs no recompute."),
    ("VaR_01",      "float", True,  "1% quantile of the return distribution. Signed, negative."),
    ("VaR_025",     "float", True,  "2.5% quantile. Signed, negative."),
    ("VaR_05",      "float", True,  "5% quantile. Signed, negative."),
    ("ES_01",       "float", False, "Expected shortfall at 1%. Signed, negative. NaN if the model does not produce it."),
    ("ES_025",      "float", False, "Expected shortfall at 2.5%. Signed, negative. NaN allowed."),
    ("Realized",    "float", True,  "Actual Return on Date, copied from the analysis file. For B's convenience and as a join check."),
    ("RVProxy",     "float", True,  "RV_Scaled on Date - realized variance on the close-to-close scale. NaN where RV_Valid is False."),
    ("Valid",       "bool",  True,  "True if this row is an evaluable forecast. B must filter on this."),
    ("Reason",      "str",   False, "Why Valid is False. Empty string when Valid is True."),
]

COLS = [s[0] for s in SCHEMA]
REQUIRED_NUMERIC = ["SigmaHat", "VarHat", "VaR_01", "VaR_025", "VaR_05"]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
class ForecastContractError(ValueError):
    pass


def validate(df, strict=True):
    """Check a forecast frame against the contract. Returns the list of problems found.

    strict=True raises on the first non-empty problem list. B should keep strict=True;
    it is the whole point of having a contract.
    """
    p = []
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        p.append(f"missing columns: {missing}")
        if strict:
            raise ForecastContractError("; ".join(p))
        return p

    extra = [c for c in df.columns if c not in COLS]
    if extra:
        p.append(f"unexpected columns (drop or fold into Spec): {extra}")

    if len(df) == 0:
        p.append("empty file")

    d = pd.to_datetime(df["Date"])
    o = pd.to_datetime(df["OriginDate"])
    if not d.is_monotonic_increasing:
        p.append("Date is not sorted ascending")
    if d.duplicated().any():
        p.append(f"duplicate Date values: {int(d.duplicated().sum())}")

    # ---- the look-ahead rule. OriginDate must strictly precede the target date. ----
    bad = (o >= d) & o.notna()
    if bad.any():
        p.append(f"LOOK-AHEAD: OriginDate >= Date on {int(bad.sum())} rows, "
                 f"first {d[bad].iloc[0].date()}")

    for c in ("Code", "Model"):
        if df[c].nunique(dropna=False) != 1:
            p.append(f"{c} is not constant within the file: {sorted(df[c].unique())[:5]}")
    if df["Code"].iloc[0] not in CODES:
        p.append(f"unknown Code {df['Code'].iloc[0]!r}")

    v = df["Valid"].astype(bool)
    if not v.any():
        p.append("no valid rows at all")

    sub = df.loc[v]
    for c in REQUIRED_NUMERIC:
        n = sub[c].isna().sum()
        if n:
            p.append(f"{c} is NaN on {int(n)} rows where Valid is True")

    s = pd.to_numeric(sub["SigmaHat"], errors="coerce")
    if (s <= 0).any():
        p.append(f"SigmaHat <= 0 on {int((s <= 0).sum())} valid rows")

    # VarHat must actually be the square, or QLIKE and MSE disagree with each other
    vh = pd.to_numeric(sub["VarHat"], errors="coerce")
    rel = (vh - s ** 2).abs() / (s ** 2).replace(0, np.nan)
    if (rel > 1e-8).any():
        p.append(f"VarHat != SigmaHat^2 on {int((rel > 1e-8).sum())} rows "
                 f"(max rel dev {float(rel.max()):.2e})")

    # ---- sign and ordering. This is the convention people get wrong. ----
    q01, q025, q05 = (pd.to_numeric(sub[f"VaR_{k}"], errors="coerce") for k, _ in LEVELS)
    if (q05 >= 0).any():
        p.append(f"VaR_05 >= 0 on {int((q05 >= 0).sum())} rows - VaR must be a SIGNED, "
                 f"NEGATIVE quantile, not a positive loss magnitude")
    if (q01 > q025).any() or (q025 > q05).any():
        p.append(f"VaR levels out of order on "
                 f"{int(((q01 > q025) | (q025 > q05)).sum())} rows; "
                 f"required VaR_01 <= VaR_025 <= VaR_05")

    for k in ("01", "025"):
        e, q = sub[f"ES_{k}"], sub[f"VaR_{k}"]
        m = e.notna()
        if m.any() and (e[m] > q[m]).any():
            p.append(f"ES_{k} > VaR_{k} on {int((e[m] > q[m]).sum())} rows - "
                     f"expected shortfall is further into the tail, so it must be <= VaR")

    # a plausibility band, not a hard law. Daily equity-index sigma outside this is a bug.
    if v.any() and (s.median() < 0.001 or s.median() > 0.15):
        p.append(f"median SigmaHat is {float(s.median()):.5f} - outside the plausible daily "
                 f"range. Is it in percent instead of decimal, or on the session scale?")

    if strict and p:
        raise ForecastContractError("FORECAST CONTRACT VIOLATED:\n  - " + "\n  - ".join(p))
    return p


# ---------------------------------------------------------------------------
# read / write
# ---------------------------------------------------------------------------
def read_forecasts(path, strict=True):
    """Read one forecast file, validate it, and return a typed frame.

    This is the ONLY function B should use to load A's output. It fails loudly on a
    malformed file rather than letting a sign error propagate into a backtest.
    """
    df = pd.read_csv(path, parse_dates=["Date", "OriginDate"])
    df["Valid"] = df["Valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["Reason"] = df.get("Reason", "").fillna("")
    validate(df, strict=strict)
    return df


def read_all(model, codes=CODES, base=None, strict=True):
    """Read every index for one model into a dict keyed by code."""
    base = base or FCDIR
    return {c: read_forecasts(os.path.join(base, f"{model}__{c}_forecasts.csv"), strict)
            for c in codes}


def write_forecasts(df, model, code, base=None, spec=None, validate_first=True):
    """Write one forecast file in the contract format. A calls this; nothing else writes."""
    base = base or FCDIR
    os.makedirs(base, exist_ok=True)
    out = df.copy()
    out["Code"] = code
    out["Model"] = model
    if spec is not None:
        out["Spec"] = spec
    if "Horizon" not in out:
        out["Horizon"] = 1
    for c, t, req, _ in SCHEMA:
        if c not in out:
            if req:
                raise ForecastContractError(f"required column {c!r} not supplied")
            out[c] = "" if t == "str" else np.nan
    out["Reason"] = out["Reason"].fillna("").astype(str)
    out = out[COLS].sort_values("Date").reset_index(drop=True)
    if validate_first:
        validate(out, strict=True)
    path = os.path.join(base, f"{model}__{code}_forecasts.csv")
    out.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.10g")
    return path


# ---------------------------------------------------------------------------
# the evaluation skeleton A guarantees B can build against
# ---------------------------------------------------------------------------
def load_actuals(code):
    """The truth series B evaluates against, pulled straight from the analysis file.

    Returns Date, Realized (close-to-close log return) and RVProxy (RV_Scaled_Causal, i.e.
    realized variance converted to the close-to-close scale using the CAUSAL, expanding
    Hansen-Lunde factor - see 15_build_analysis_dataset.py DECISION 5. RV_Scaled (the older
    full-sample-constant factor) is look-ahead and must never feed an evaluation number;
    2026-08-29 fix, since every model's QLIKE/MSE ultimately compares against this series.
    RVProxy is NaN wherever RV_Valid is False or the causal factor's warm-up hasn't kicked in
    yet - B must drop those rows for QLIKE/MSE but KEEP them for VaR backtests, which need
    only Realized.
    """
    a = pd.read_csv(os.path.join(ANA, f"{code}_analysis.csv"),
                    parse_dates=["Date"], low_memory=False)
    rv_valid_causal = a["RV_Valid"].astype(bool) & a["ScaleFactor_HL_Causal"].notna()
    a["RVProxy"] = np.where(rv_valid_causal, a["RV_Scaled_Causal"], np.nan)
    return a[["Date", "Return", "RVProxy", "InSample_B", "CommonDate_B",
              "BalancedRV_B", "RV_Valid", "CrisisLabel", "IsCrisis",
              "VolRegime", "VolRegime_ExAnte"]].rename(columns={"Return": "Realized"})


def eval_frame(fc, code=None):
    """Join a forecast file to the actuals and return only the rows that can be evaluated.

    Adds the two loss functions the plan names, so both researchers compute them identically:
      QLIKE = RVProxy/VarHat - log(RVProxy/VarHat) - 1     (0 at a perfect forecast)
      SE    = (RVProxy - VarHat) ** 2
    and the breach indicators the VaR backtests consume.
    """
    code = code or fc["Code"].iloc[0]
    act = load_actuals(code)
    m = fc.merge(act.drop(columns=["Realized", "RVProxy"]), on="Date", how="left")
    m = m[m["Valid"].astype(bool)].copy()

    r = m["RVProxy"] / m["VarHat"]
    m["QLIKE"] = np.where(r > 0, r - np.log(r.where(r > 0)) - 1.0, np.nan)
    m["SE"] = (m["RVProxy"] - m["VarHat"]) ** 2
    for k, tau in LEVELS:
        m[f"Breach_{k}"] = m["Realized"] < m[f"VaR_{k}"]
    return m


def schema_frame():
    return pd.DataFrame(SCHEMA, columns=["Column", "Type", "Required", "Description"])


if __name__ == "__main__":
    print(schema_frame().to_string(index=False))
