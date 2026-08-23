# -*- coding: utf-8 -*-
"""
EDA STAGE 8 - act on the screening results, then document the dataset.

TWO FIXES CARRIED OVER FROM THE SCREENING

  1. Stage 5 found US10Y_pct and TermSpread_pct to be non-stationary in levels on all six
     indices (ADF p = 0.87 and 0.33) while their first differences are stationary at p < 0.001.
     Feeding a unit-root regressor into a quantile regression against a stationary target
     invites a spurious fit, so the differenced forms are added here and the data dictionary
     marks the levels as "do not use as a QR regressor". The levels are retained because
     they are legitimate as a regime descriptor and for plotting.

  2. The balanced-panel flag was defined on data availability alone, before the session
     classification existed. It is recomputed here as BalancedRV_B: dates inside sample B
     where EVERY index has a VALID realized measure. This is the set to use for any POOLED
     cross-index statistic. Per-index work should NOT use it - it needlessly discards days
     that are perfectly good for the index in question.

Then the dataset is documented: a per-column data dictionary and an explicit statement of
which fields feed which of the three models.

Outputs: analysis/<CODE>_analysis.csv  (rewritten with the added columns)
         01_ANALYSIS_READY/DATA_DICTIONARY.csv
         01_ANALYSIS_READY/FEATURE_SETS.csv
         _logs/phase14_finalise.csv
"""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
LOG = os.path.join(ROOT, '11_LOGS')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

# ------------------------------------------------------------------ dictionary
# col, group, units, availability, look-ahead status, note
DICT = [
 ("Date", "key", "date", "always", "-", "exchange trading date, local"),
 ("Symbol", "key", "code", "always", "-", "index code"),
 ("Open,High,Low,Close", "price", "index points", "always", "known at close t",
  "exchange daily bar from Yahoo Finance"),
 ("Return", "target", "log return", "always", "known at close t",
  "log(C_t/C_t-1). PRIMARY TARGET for GARCH-EVT and Quantile Regression"),
 ("Return_Pct,Return_Sq,AbsReturn,NegReturn", "target", "various", "always",
  "known at close t", "transformations of Return; NegReturn = |min(r,0)| leverage term"),
 ("RV", "realized", "daily variance", "RV_Valid only", "known at close t",
  "5-min realized variance, cash session only. PRIMARY realized measure"),
 ("RVol,LogRV,RVol_Ann_Pct", "realized", "sd / log / annualised %", "RV_Valid only",
  "known at close t", "transformations of RV"),
 ("RV_1min,RV_ss", "realized", "daily variance", "RV_Valid only", "known at close t",
  "1-min RV and the 5-grid subsampled RV (Zhang-Mykland-Ait-Sahalia)"),
 ("BPV,MedRV", "realized", "daily variance", "RV_Valid only", "known at close t",
  "jump-robust integrated variance. MedRV is robust to consecutive jumps"),
 ("Jump,ContVar,JumpShare", "realized", "daily variance / ratio", "RV_Valid only",
  "known at close t", "J = max(RV - MedRV, 0); ContVar = RV - J; JumpShare = J/RV"),
 ("RS_pos,RS_neg,SignedJump,RSV_Ratio", "realized", "daily variance / ratio",
  "RV_Valid only", "known at close t",
  "realized semivariance. RS_pos + RS_neg = RV EXACTLY. RSV_Ratio = RS_neg/RV"),
 ("LogRS_neg", "realized", "log variance", "RV_Valid only", "known at close t",
  "DO NOT use together with LogRV - VIF ~95, they are the same variable plus a constant"),
 ("RQ,TQ,RV_RelSE", "realized", "quarticity / ratio", "RV_Valid only", "known at close t",
  "realized quarticity for HAR-Q; RV_RelSE = sqrt(2*RQ/n)/RV is the relative measurement error"),
 ("RSkew,RKurt", "realized", "dimensionless", "RV_Valid only", "known at close t",
  "intraday realized skewness and kurtosis (Amaya et al. 2015)"),
 ("NoiseRatio_1v5", "diagnostic", "ratio", "RV_Valid only", "known at close t",
  "RV_1min / RV_5min; a microstructure-noise indicator, ~1.06-1.09 here"),
 ("RV_d,RV_w,RV_m", "HAR", "daily variance", "RV_Valid only", "known at close t",
  "HAR cascade: today, trailing 5-day mean, trailing 22-day mean, all INCLUDING t"),
 ("LogRV_d,LogRV_w,LogRV_m,RSneg_*,CV_*,J_*", "HAR", "various", "RV_Valid only",
  "known at close t", "same cascade on the log, semivariance, continuous and jump parts"),
 ("RV_Scaled,ScaleFactor_HL", "realized", "daily variance", "RV_Valid only",
  "IN-SAMPLE constant",
  "Hansen-Lunde scaling of session RV onto close-to-close variance. WARNING: the factor is "
  "estimated once over the whole of sample B, so it is in-sample information. Fine for the "
  "measurement equation and for description; for a strict recursive out-of-sample exercise "
  "re-estimate it on each rolling window."),
 ("VolIdx,VolIdx_Symbol,VolIdx_IsProxy", "implied", "annualised vol %", "always",
  "known at close t", "regional implied volatility index actually used; IsProxy flags NKY->VXEFA"),
 ("IV_DailyVar,LogIV", "implied", "daily variance", "always", "known at close t",
  "(VolIdx/100)^2/252 - the vol index converted onto the RV scale"),
 ("VRP", "implied", "daily variance", "RV_Valid only", "known at close t",
  "variance risk premium = IV_DailyVar - RV"),
 ("ParkinsonVar,GarmanKlassVar,RangePct", "range", "daily variance / %", "always",
  "known at close t",
  "range estimators from the exchange daily bar. AVAILABLE ON EVERY DAY, including "
  "intraday-defect days - the natural fallback predictor"),
 ("Overnight_LogRet", "return", "log return", "RV_Valid only", "known at OPEN of t",
  "close(t-1) -> open(t). NOT a day-t close variable; the only column that is not close-dated"),
 ("CFD_SessionReturn", "diagnostic", "log return", "RV_Valid only", "known at close t",
  "close-to-close return of the CFD used to build RV; for reconciliation against Return"),
 ("US10Y_pct,US13W_pct,TermSpread_pct", "05_RAW_MACRO", "percent", "always", "known at close t",
  "LEVELS ARE NON-STATIONARY (ADF p=0.87/0.33). Do not use as QR regressors - use the _diff forms"),
 ("US10Y_diff,TermSpread_diff", "05_RAW_MACRO", "percentage points", "always", "known at close t",
  "first differences; stationary at p<0.001. USE THESE in the quantile regression"),
 ("DXY,WTI_usd,GOLD_usd,HYG_px,IEF_px", "05_RAW_MACRO", "level", "always", "known at close t",
  "levels, forward-filled up to 5 business days; use the _ret forms as regressors"),
 ("DXY_ret,WTI_ret,GOLD_ret,HYG_ret,IEF_ret", "05_RAW_MACRO", "log return", "always",
  "known at close t", "log differences of the above"),
 ("CreditStress", "05_RAW_MACRO", "log return", "always", "known at close t",
  "-(HYG_ret - IEF_ret); rises with credit stress, same sign as a widening high-yield spread"),
 ("MacroFilled_t", "quality", "boolean", "always", "-",
  "TRUE where at least one macro level was forward-filled on this row"),
 ("SessionClass", "quality", "FULL/HALFDAY/DEFECT/MISSING", "always", "-",
  "intraday session classification driving RV_Valid"),
 ("RV_Coverage,HasOpen,MaxInteriorGap_blocks", "quality", "ratio / bool / blocks",
  "where intraday exists", "-", "inputs to the classification, kept for auditability"),
 ("RV_Valid", "quality", "boolean", "always", "-",
  "THE GATE for every realized measure. TRUE only for FULL sessions with RV>0"),
 ("IsHalfDay,IsDefect", "quality", "boolean", "always", "-",
  "half-days are real short sessions; defects are feed failures"),
 ("InSample_A,InSample_B,InSample_C", "sample", "boolean", "always", "-",
  "sample windows; B is PRIMARY (all six indices, 2013-09-30 onward)"),
 ("CommonDate_B", "sample", "boolean", "always", "-",
  "date present for all six indices in sample B, ignoring RV validity"),
 ("CrisisLabel,IsCrisis", "regime", "label / boolean", "always", "known at close t",
  "10 named crisis windows with fixed calendar dates, identical across all six indices so "
  "cross-index comparison is meaningful. See CRISIS_PERIODS.csv for the dates and the basis "
  "for each. Sample B contains 6 of the 10; it MISSES the GFC, dot-com, Asian crisis and Euro "
  "sovereign crisis"),
 ("TrailVol63_Ann", "regime", "annualised %", "always", "known at close t",
  "trailing 63-day return volatility, annualised. Defined on every day including "
  "intraday-defect days, because it uses returns rather than RV"),
 ("VolRegime", "regime", "Calm/Normal/Stressed/Crisis", "always", "EX-POST",
  "quartiles of TrailVol63_Ann using FULL-SAMPLE cut-points. In-sample information: correct "
  "for ex-post subsample REPORTING, NEVER as a predictor"),
 ("VolRegime_ExAnte", "regime", "Calm/Normal/Stressed/Crisis", "from ~750 obs",
  "known at close t",
  "same quartiles from EXPANDING-WINDOW quantiles using only data strictly before each date. "
  "No look-ahead - safe as a regressor or conditioning variable"),
 ("BalancedRV_B", "sample", "boolean", "always", "-",
  "date where ALL SIX indices have a VALID realized measure. Use for POOLED cross-index "
  "statistics only; per-index work should use RV_Valid and keep its own days"),
]

# ------------------------------------------------------------------ feature sets
FEATURES = [
 ("GARCH-EVT", "target", "Return",
  "log return, full history from 1990 - this model needs no intraday data"),
 ("GARCH-EVT", "estimation note", "-",
  "stage 1 AR(1)-GJR-GARCH on Return; stage 2 GPD on the standardised residuals"),
 ("GARCH-EVT", "POT threshold", "-",
  "95th-97.5th percentile of the standardised residuals - the region where the stage-6 "
  "stability plot shows xi flat with 230-460 exceedances"),
 ("GARCH-EVT", "sample", "InSample_B for evaluation",
  "estimate on all history; only the forecast window must match the other models"),

 ("Realized GARCH", "target", "Return", "log return"),
 ("Realized GARCH", "realized measure", "RV",
  "5-min RV, gated on RV_Valid. RV_ss is the low-noise alternative"),
 ("Realized GARCH", "measurement eq.", "ScaleFactor_HL",
  "RV covers the session only, so it is 1.71-3.04x below close-to-close variance depending "
  "on session length; the intercept must absorb this or RV_Scaled must be used"),
 ("Realized GARCH", "optional", "MedRV, ContVar, Jump, RS_neg",
  "jump-robust and signed variants for the robustness table"),
 ("Realized GARCH", "sample", "RV_Valid & InSample_B",
  "NKY has no valid RV in 2016-17; that index is estimated on 2,265 days with a gap"),

 ("Quantile Regression", "target", "Return(t+1)", "next-day log return, tau = 0.01 and 0.05"),
 ("Quantile Regression", "core regressors",
  "LogRV, LogRV_w, LogRV_m, LogIV, VRP, NegReturn, RangePct",
  "the orthogonal parameterisation - max VIF 8.1 versus 21.9 for the naive set"),
 ("Quantile Regression", "asymmetry", "RSV_Ratio, JumpShare, RSkew",
  "downside share, jump share and intraday skew; these replace LogRS_neg, which is "
  "collinear with LogRV by construction"),
 ("Quantile Regression", "05_RAW_MACRO", "TermSpread_diff, CreditStress, DXY_ret",
  "differenced or already-stationary forms only"),
 ("Quantile Regression", "fallback", "RangePct, ParkinsonVar, GarmanKlassVar",
  "available on every day including intraday-defect days; RangePct alone reaches R2=0.42 "
  "for next-day log RV, close to the implied-vol index"),
 ("Quantile Regression", "excluded", "LogRS_neg, US10Y_pct, TermSpread_pct level",
  "collinear (VIF 95) or non-stationary in levels"),

 ("All models", "evaluation", "RV or RV_Scaled as the volatility proxy",
  "QLIKE and MSE against the realized measure; VaR backtests against Return"),
 ("All models", "pooled tests", "BalancedRV_B",
  "Model Confidence Set across indices requires aligned loss series"),
]


def main():
    # ---- recompute the balanced flag on RV validity ----
    A = {c: pd.read_csv(os.path.join(ANA, f'{c}_analysis.csv'), parse_dates=['Date'])
         for c in CODES}
    inter = None
    for c in CODES:
        d = set(A[c].loc[A[c]['InSample_B'] & A[c]['RV_Valid'], 'Date'])
        inter = d if inter is None else (inter & d)
    inter = pd.DatetimeIndex(sorted(inter))

    common = None
    for c in CODES:
        d = set(A[c].loc[A[c]['InSample_B'], 'Date'])
        common = d if common is None else (common & d)

    rows = []
    for c in CODES:
        a = A[c]
        a['US10Y_diff'] = a['US10Y_pct'].diff()
        a['TermSpread_diff'] = a['TermSpread_pct'].diff()
        a['CommonDate_B'] = a['Date'].isin(common)
        a['BalancedRV_B'] = a['Date'].isin(inter)
        a.to_csv(os.path.join(ANA, f'{c}_analysis.csv'), index=False,
                 date_format='%Y-%m-%d', float_format='%.10g')
        rows.append(dict(Code=c, Rows=len(a), Cols=len(a.columns),
                         InSample_B=int(a['InSample_B'].sum()),
                         RV_Valid_in_B=int((a['InSample_B'] & a['RV_Valid']).sum()),
                         CommonDate_B=int(a['CommonDate_B'].sum()),
                         BalancedRV_B=int(a['BalancedRV_B'].sum())))

    d = pd.DataFrame(DICT, columns=['Columns', 'Group', 'Units', 'Availability',
                                    'Timing', 'Note'])
    d.to_csv(os.path.join(ANA, 'DATA_DICTIONARY.csv'), index=False)
    f = pd.DataFrame(FEATURES, columns=['Model', 'Role', 'Fields', 'Note'])
    f.to_csv(os.path.join(ANA, 'FEATURE_SETS.csv'), index=False)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(LOG, 'phase14_finalise.csv'), index=False)

    pd.set_option('display.width', 200)
    print(s.to_string(index=False))
    print()
    print(f"balanced-RV sample B : {len(inter)} days  "
          f"{inter.min().date()} -> {inter.max().date()}")
    print(f"common-date sample B : {len(common)} days")
    print(f"data dictionary      : {len(d)} entries -> 01_ANALYSIS_READY/DATA_DICTIONARY.csv")
    print(f"feature sets         : {len(f)} entries -> 01_ANALYSIS_READY/FEATURE_SETS.csv")


if __name__ == "__main__":
    main()
