# -*- coding: utf-8 -*-
"""
EDA STAGE 3 / PREPROCESSING - build the cleaned, model-ready analysis dataset.

This is where the audit findings become rules. Every decision below is deliberate and each
one is stated with its reason, because in a tail-risk study the cleaning choices drive the
results at least as much as the models do.

-------------------------------------------------------------------------------------
DECISION 1  Bad realized measures are NULLED, the row is NOT deleted.
  A DEFECT session means the intraday feed failed. It does NOT mean the day did not happen:
  the exchange closing price is still correct, so the daily return is still a valid
  observation. Deleting the row would silently shorten the return series that GARCH-EVT and
  Quantile Regression are estimated on, and - far worse - it would delete days
  non-randomly with respect to volatility. So RV and everything derived from it is set to
  NaN on DEFECT days while the return, range and volatility-index columns survive intact.
  Consequence: the three models legitimately see different numbers of observations. That is
  correct, not a flaw, and RV_Valid records exactly which days each can use.

DECISION 2  Half-days are kept but flagged, and excluded from RV_Valid.
  A half-day is a real session, so its RV is a correct measurement - of a three-hour day.
  Feeding a three-hour variance into a measurement equation calibrated on six-and-a-half
  hour days biases the Realized GARCH intercept. The realized-volatility literature
  routinely removes shortened sessions for exactly this reason. They are therefore excluded
  from RV_Valid but retained in the file with IsHalfDay=True so the choice can be reversed.

DECISION 3  Returns are NOT winsorized, trimmed or de-jumped. At all.
  This has to be said explicitly because winsorizing is a near-reflex in applied work. The
  object of study here IS the tail. Clipping the largest moves would remove the very
  observations that identify the GPD shape parameter, mechanically shrink the estimated tail
  index, and produce VaR forecasts that look well-calibrated in-sample precisely because the
  exceedances were deleted. The extreme returns were individually verified in stage 1
  (NDX 2001-01-03 +17.2%, HSI 1997-10-29 +17.3% - both real market events) and are kept.

DECISION 4  Macro predictors are forward-filled up to 5 business days, and only forward.
  The macro series are US-market series joined onto six different exchange calendars, so
  they are missing whenever a local market trades and the US does not. The correct value for
  a predictor on such a day is the last published one, because that is genuinely what was
  known at the time. Forward fill is therefore not an interpolation, it is the information
  set. Backward fill would be look-ahead and is never used. The 5-day cap stops a long
  holiday from propagating a stale level indefinitely, and MacroFilled_t records the days
  where filling was applied.

DECISION 5  The RV / daily-variance scale gap is measured, not assumed away.
  RV covers the cash session only; the daily return spans close to close and so also
  contains the overnight gap. RV is therefore a biased-low estimator of daily variance by
  construction. The Hansen-Lunde (2005) scaling factor c = sum(r^2)/sum(RV) is estimated per
  index on the clean sample and stored, so the measurement equation and the QLIKE evaluation
  can both use a like-for-like variance.

DECISION 6  Nothing is standardised, differenced or de-meaned here.
  Those are modelling choices that depend on the estimator, and baking them into the stored
  data would make the file unusable for anything else. Levels are stored; transformations
  belong in the modelling code.
-------------------------------------------------------------------------------------

Output: analysis/<CODE>_analysis.csv
        analysis/SCALE_FACTORS.csv
        _logs/phase13_analysis_build.csv
"""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAN = os.path.join(ROOT, '07_PANEL_INTERMEDIATE')
RVD = os.path.join(ROOT, '06_REALIZED_MEASURES')
VAL = os.path.join(ROOT, '08_VALIDATION')
OUT = os.path.join(ROOT, '01_ANALYSIS_READY')
LOG = os.path.join(ROOT, '11_LOGS')
os.makedirs(OUT, exist_ok=True)

CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]
FFILL_LIMIT = 5
MACRO_LEVELS = ['US10Y_pct', 'US13W_pct', 'DXY', 'WTI_usd', 'GOLD_usd', 'HYG_px', 'IEF_px']

# realized columns that must be nulled when the session is not usable
RV_DERIVED = [
    'RV', 'RVol', 'LogRV', 'RV_1min', 'RV_ss', 'BPV', 'MedRV', 'RQ', 'TQ',
    'RS_pos', 'RS_neg', 'SignedJump', 'RSkew', 'RKurt', 'Jump', 'ContVar',
    'JumpShare', 'RSV_Ratio', 'RV_RelSE', 'NoiseRatio_1v5', 'NBars',
    'CFD_SessionReturn', 'Overnight_LogRet']


def har(s, window, min_frac=0.6):
    """Trailing average INCLUDING today - the standard HAR predictor dated t for t+1.

    min_frac guards the NKY 2016-17 hole: a 22-day average computed from three surviving
    observations is not a monthly volatility measure, so it is returned as NaN rather than
    silently standing in for one.
    """
    return s.rolling(window, min_periods=max(2, int(np.ceil(min_frac * window)))).mean()


def build(code):
    p = pd.read_csv(os.path.join(PAN, f'{code}_panel_daily.csv'), parse_dates=['Date'])
    ex = pd.read_csv(os.path.join(RVD, f'{code}_RV_extended.csv'), parse_dates=['Date'])
    cl = pd.read_csv(os.path.join(VAL, 'eda2_session_class.csv'), parse_dates=['Date'])
    cl = cl[cl['Symbol'] == code][['Date', 'Class', 'Coverage', 'HasOpen',
                                   'MaxInteriorGap_blocks']]

    a = p[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Return',
           'VolUsed', 'VolUsed_Symbol', 'VolUsed_IsProxy',
           'InSample_A', 'InSample_B', 'InSample_C'] + MACRO_LEVELS].copy()
    a = a.rename(columns={'VolUsed': 'VolIdx', 'VolUsed_Symbol': 'VolIdx_Symbol',
                          'VolUsed_IsProxy': 'VolIdx_IsProxy'})

    # Recompute the return from the price at FULL precision instead of inheriting it.
    # The download script rounded LogReturn to 6 decimal places, which on a typical 1%
    # daily move is only about 5e-5 relative precision and shows up as a 5e-7 absolute
    # mismatch against log(C_t/C_t-1). Harmless for estimation, but there is no reason to
    # carry a quantised target when the price it derives from is stored to 6 significant
    # decimals, so it is rederived here and the rounded column is discarded.
    a['Return'] = np.log(a['Close'] / a['Close'].shift(1))

    # ---- realized measures, 5-min grid ------------------------------------
    take = {'RV_5min': 'RV', 'RV_1min': 'RV_1min', 'RV_ss5min': 'RV_ss',
            'BPV_c_5min': 'BPV', 'MedRV_5min': 'MedRV', 'RQ_5min': 'RQ', 'TQ_5min': 'TQ',
            'RS_pos_5min': 'RS_pos', 'RS_neg_5min': 'RS_neg', 'SignedJump_5min': 'SignedJump',
            'RSkew_5min': 'RSkew', 'RKurt_5min': 'RKurt', 'NBars_5min': 'NBars',
            'Jump_MedRV_5min': 'Jump', 'ContVar_MedRV_5min': 'ContVar',
            'JumpShare_5min': 'JumpShare', 'RSV_Ratio_5min': 'RSV_Ratio',
            'RV_RelSE_5min': 'RV_RelSE', 'NoiseRatio_1v5': 'NoiseRatio_1v5'}
    e = ex[['Date'] + [k for k in take if k in ex.columns]].rename(columns=take)
    a = a.merge(e, on='Date', how='left')
    a = a.merge(p[['Date', 'CFD_SessionReturn', 'Overnight_LogRet']], on='Date', how='left')
    a = a.merge(cl, on='Date', how='left')

    # ---- DECISION 1 + 2: quality flags, then null the unusable measures ----
    a['SessionClass'] = a['Class'].fillna('MISSING')
    a = a.drop(columns=['Class'])
    a['RV_Coverage'] = a['Coverage']
    a = a.drop(columns=['Coverage'])
    a['IsHalfDay'] = a['SessionClass'].eq('HALFDAY')
    a['IsDefect'] = a['SessionClass'].eq('DEFECT')
    a['RV_Valid'] = a['SessionClass'].eq('FULL') & a['RV'].notna() & (a['RV'] > 0)

    n_nulled = int((~a['RV_Valid'] & a['RV'].notna()).sum())
    for c in RV_DERIVED:
        if c in a.columns:
            a.loc[~a['RV_Valid'], c] = np.nan

    # ---- derived volatility quantities ------------------------------------
    a['RVol'] = np.sqrt(a['RV'])
    a['LogRV'] = np.log(a['RV'].where(a['RV'] > 0))
    a['RVol_Ann_Pct'] = 100.0 * np.sqrt(252.0 * a['RV'])
    a['LogRS_neg'] = np.log(a['RS_neg'].where(a['RS_neg'] > 0))

    # range-based estimators from the exchange daily bar - available on EVERY day, including
    # the ones where the intraday feed failed, which makes them the natural robustness proxy
    hl = np.log(a['High'] / a['Low'])
    co = np.log(a['Close'] / a['Open'])
    a['ParkinsonVar'] = hl ** 2 / (4 * np.log(2))
    a['GarmanKlassVar'] = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    a['RangePct'] = 100 * (a['High'] - a['Low']) / a['Close']

    # ---- HAR components (dated t, no look-ahead) ---------------------------
    for src, tag in (('RV', 'RV'), ('LogRV', 'LogRV'), ('RS_neg', 'RSneg'),
                     ('ContVar', 'CV'), ('Jump', 'J')):
        if src not in a.columns:
            continue
        a[f'{tag}_d'] = a[src]
        a[f'{tag}_w'] = har(a[src], 5)
        a[f'{tag}_m'] = har(a[src], 22)

    # ---- implied volatility in comparable units ----------------------------
    # the vol indices are quoted as ANNUALISED percentage volatility; convert to a daily
    # variance so it sits on the same scale as RV and can be differenced against it
    a['IV_DailyVar'] = (a['VolIdx'] / 100.0) ** 2 / 252.0
    a['VRP'] = a['IV_DailyVar'] - a['RV']          # variance risk premium
    a['LogIV'] = np.log(a['IV_DailyVar'].where(a['IV_DailyVar'] > 0))

    # ---- DECISION 4: macro forward fill, forward only, capped --------------
    before = a[MACRO_LEVELS].isna()
    a[MACRO_LEVELS] = a[MACRO_LEVELS].ffill(limit=FFILL_LIMIT)
    after = a[MACRO_LEVELS].isna()
    a['MacroFilled_t'] = (before & ~after).any(axis=1)
    n_filled = int(a['MacroFilled_t'].sum())
    a['TermSpread_pct'] = a['US10Y_pct'] - a['US13W_pct']
    # 2026-08-29 fix: 44_qr.py's QR_MACRO predictor block names 'TermSpread_diff' (the
    # day-over-day change - stationary, unlike the level TermSpread_pct) but this canonical
    # builder never created that column, so 44_qr.py has been reading a KeyError, not a value,
    # every time it was run against a freshly-rebuilt analysis file. Discovered when rerunning
    # QR after this session's other fixes; not related to the causal-scaling / NKY-session work.
    a['TermSpread_diff'] = a['TermSpread_pct'].diff()
    for c, tag in (('DXY', 'DXY'), ('WTI_usd', 'WTI'), ('GOLD_usd', 'GOLD'),
                   ('HYG_px', 'HYG'), ('IEF_px', 'IEF')):
        a[f'{tag}_ret'] = np.log(a[c]).diff()
    a['CreditStress'] = -(a['HYG_ret'] - a['IEF_ret'])

    # ---- return-side predictors -------------------------------------------
    a['AbsReturn'] = a['Return'].abs()
    a['NegReturn'] = a['Return'].clip(upper=0).abs()
    a['Return_Sq'] = a['Return'] ** 2
    a['Return_Pct'] = 100 * a['Return']

    # ---- DECISION 5: Hansen-Lunde scaling ----------------------------------
    # RV_Scaled / ScaleFactor_HL below are the ORIGINAL full-sample-constant version, kept for
    # descriptive EDA use only (EDA_REPORT.md, frequency_sensitivity.png) - NEVER feed these
    # into a walk-forward forecast, because c_hl is estimated using the entire evaluation
    # window (including its own future), which is look-ahead in the realized-measure scale.
    m = a['RV_Valid'] & a['Return'].notna() & a['InSample_B']
    c_hl = float(a.loc[m, 'Return_Sq'].sum() / a.loc[m, 'RV'].sum()) if m.sum() > 100 else np.nan
    a['RV_Scaled'] = c_hl * a['RV']
    a['ScaleFactor_HL'] = c_hl

    # ---- CAUSAL Hansen-Lunde scaling (expanding, strictly pre-date) --------
    # c_t = sum(r^2)/sum(RV) using every valid prior observation up to and including t-1 -
    # not restricted to InSample_B, so an index with pre-2013 RV history (NDX/UKX/HSI/NKY/SPX)
    # gets a genuine pre-OOS warm-up instead of leaning on its own evaluation window. Requires
    # MIN_PRIOR valid observations before trusting the ratio; before that, ScaleFactor_HL_Causal
    # (and hence RV_Scaled_Causal) is NaN, which 28_realized_garch.py's existing missing-x_t
    # handling (substitute h_{t-1} in the recursion, skip the likelihood term) already absorbs
    # correctly - no new imputation logic needed for the bootstrap window.
    MIN_PRIOR = 60
    mv = (a['RV_Valid'] & a['Return'].notna())
    cum_retsq = a['Return_Sq'].where(mv).cumsum()
    cum_rv = a['RV'].where(mv).cumsum()
    prior_retsq = cum_retsq.shift(1)
    prior_rv = cum_rv.shift(1)
    prior_n = mv.cumsum().shift(1).fillna(0)
    c_causal = np.where((prior_n >= MIN_PRIOR) & (prior_rv > 0), prior_retsq / prior_rv, np.nan)
    a['ScaleFactor_HL_Causal'] = c_causal
    a['RV_Scaled_Causal'] = a['ScaleFactor_HL_Causal'] * a['RV']

    a = a.sort_values('Date').reset_index(drop=True)
    path = os.path.join(OUT, f'{code}_analysis.csv')
    a.to_csv(path, index=False, date_format='%Y-%m-%d', float_format='%.10g')

    b = a[a['InSample_B']]
    return dict(Code=code, Rows=len(a), Cols=len(a.columns),
                First=str(a['Date'].min().date()), Last=str(a['Date'].max().date()),
                SampleB_Rows=len(b),
                RV_Valid_All=int(a['RV_Valid'].sum()),
                RV_Valid_SampleB=int(b['RV_Valid'].sum()),
                RV_Nulled=n_nulled,
                HalfDays=int(a['IsHalfDay'].sum()), Defects=int(a['IsDefect'].sum()),
                Macro_Filled_Rows=n_filled,
                ScaleFactor_HL=round(c_hl, 4),
                Median_RVol_Ann=round(float(b['RVol_Ann_Pct'].median()), 2),
                Median_VolIdx=round(float(b['VolIdx'].median()), 2))


if __name__ == "__main__":
    rows = []
    for c in CODES:
        r = build(c)
        rows.append(r)
        print(f"  [{c}] rows={r['Rows']} cols={r['Cols']} RV_valid(B)={r['RV_Valid_SampleB']} "
              f"nulled={r['RV_Nulled']} scale={r['ScaleFactor_HL']}")
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(LOG, 'phase13_analysis_build.csv'), index=False)
    s[['Code', 'ScaleFactor_HL', 'Median_RVol_Ann', 'Median_VolIdx']].to_csv(
        os.path.join(OUT, 'SCALE_FACTORS.csv'), index=False)
    pd.set_option('display.width', 220)
    print()
    print(s.to_string(index=False))
