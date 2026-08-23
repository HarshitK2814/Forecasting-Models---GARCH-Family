# -*- coding: utf-8 -*-
"""
Phase 9: build the ANALYSIS-READY master panel - one CSV per index.

Everything upstream is raw-ish: separate folders for daily prices, realized measures,
volatility indices and macro factors, each on its own calendar. This script joins them
into the single table the three models are actually estimated on, so that nobody has to
re-derive the merge (and get it subtly wrong) later.

JOIN RULE - read this before changing anything
  The exchange DAILY file is the spine. Every other series is LEFT-joined onto it on Date.
  That is deliberate: the dependent variable is the index return, so the sample must be
  exactly the days the index traded. Realized measures are only available where the CFD
  feed had a clean session, so RV_5min is NaN on some spine days; that is honest missingness
  and is reported per index, not silently filled.

NO LOOK-AHEAD
  Every predictor column is dated at the close of day t and is used to forecast day t+1.
  We do NOT lag anything inside this file - the file is a clean contemporaneous panel.
  The modelling code must apply the lag. The column HasRV_t tells you whether a realized
  measure exists for day t at all.

  The one exception is Overnight_LogRet, which spans close(t-1) -> open(t) and is therefore
  known at the OPEN of day t. It is still dated t. Do not treat it as day-t information when
  forecasting day t; treat it as the first observable piece of day t.

Output : Datasets/07_PANEL_INTERMEDIATE/<CODE>_panel_daily.csv
Summary: Datasets/11_LOGS/phase9_panel_summary.csv
"""
import os, sys, glob, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, '07_PANEL_INTERMEDIATE')
LOG  = os.path.join(ROOT, '11_LOGS')
os.makedirs(OUT, exist_ok=True); os.makedirs(LOG, exist_ok=True)

# index -> (its own regional volatility index, fallback/global proxy)
VOLMAP = {
    "SPX": ("VIX",   None),
    "NDX": ("VXN",   "VIX"),
    "UKX": ("VXEFA", "VIX"),    # no free FTSE-100 vol index exists; VXEFA = developed ex-US
    "DAX": ("V1X",   "V2TX"),   # VDAX-NEW, fallback VSTOXX
    "NKY": ("NKVI",  "VXEFA"),  # Nikkei VI only from 2018 - fallback matters here
    "HSI": ("VXEEM", "VIX"),    # no free HSI vol index; VXEEM = emerging-market proxy
}

# macro columns lifted into every panel (Close only - see the OHLC note in script 09)
MACRO = {
    "US10Y": "US10Y_pct", "US13W": "US13W_pct", "DXY": "DXY",
    "WTI": "WTI_usd", "GOLD": "GOLD_usd",
    "HYG": "HYG_px", "IEF": "IEF_px",
}


def load_one(pattern, cols, rename):
    fs = glob.glob(pattern)
    if not fs:
        return None
    d = pd.read_csv(fs[0], parse_dates=['Date'])
    d = d[[c for c in cols if c in d.columns]].rename(columns=rename)
    return d.drop_duplicates(subset='Date').sort_values('Date')


def build(code):
    dly = load_one(os.path.join(ROOT, '02_RAW_DAILY', code, f'{code}_daily_*.csv'),
                   ['Date', 'Open', 'High', 'Low', 'Close', 'LogReturn'],
                   {'Open': 'Open', 'High': 'High', 'Low': 'Low',
                    'Close': 'Close', 'LogReturn': 'Return'})
    if dly is None:
        print(f"  [{code}] SKIP - no daily file"); return None
    p = dly.copy()
    p.insert(1, 'Symbol', code)

    # ---- realized measures -------------------------------------------------
    rvp = os.path.join(ROOT, '06_REALIZED_MEASURES', f'{code}_RV_daily.csv')
    rv_cols = []
    if os.path.exists(rvp):
        rv = pd.read_csv(rvp, parse_dates=['Date'])
        keep = ['Date', 'RV_1min', 'RV_5min', 'RV_10min', 'RV_15min', 'RV_30min',
                'BPV_5min', 'NBars_5min', 'RVol_5min', 'LogRV_5min',
                'CloseToClose_LogRet', 'Overnight_LogRet']
        rv = rv[[c for c in keep if c in rv.columns]].drop_duplicates(subset='Date')
        rv = rv.rename(columns={'CloseToClose_LogRet': 'CFD_SessionReturn'})
        p = p.merge(rv, on='Date', how='left')
        rv_cols = [c for c in rv.columns if c != 'Date']
        p['HasRV_t'] = p['RV_5min'].notna()
        # jump component, Barndorff-Nielsen & Shephard: J = max(RV - BPV, 0)
        if {'RV_5min', 'BPV_5min'}.issubset(p.columns):
            p['Jump_5min'] = (p['RV_5min'] - p['BPV_5min']).clip(lower=0)
            p['ContVar_5min'] = p['RV_5min'] - p['Jump_5min']
    else:
        p['HasRV_t'] = False

    # ---- volatility index (own, then fallback) -----------------------------
    own, fb = VOLMAP[code]
    for tag, sym in (('VolIdx', own), ('VolIdx_Fallback', fb)):
        if not sym:
            continue
        v = load_one(os.path.join(ROOT, '04_RAW_VOLATILITY', sym, f'{sym}_daily_*.csv'),
                     ['Date', 'Close'], {'Close': tag})
        if v is not None:
            p = p.merge(v, on='Date', how='left')
            p[tag + '_Symbol'] = sym

    # ---- macro -------------------------------------------------------------
    for mcode, col in MACRO.items():
        m = load_one(os.path.join(ROOT, '05_RAW_MACRO', mcode, f'{mcode}_daily_*.csv'),
                     ['Date', 'Close'], {'Close': col})
        if m is not None:
            p = p.merge(m, on='Date', how='left')
    # UNITS: Yahoo HISTORICALLY quoted the CBOE yield indices (^TNX etc.) as percent x 10,
    # and a lot of old code still divides by 10. It no longer does. Verified 2026-08-23
    # against known history: raw ^TNX averages 8.55 in 1990, 0.88 in 2020 and prints 4.74
    # today - i.e. already percent. Dividing by 10 here produced a 10-year yield with a
    # range of 0.05-0.91%, which is what caught it. Do NOT reintroduce the /10.
    if {'US10Y_pct', 'US13W_pct'}.issubset(p.columns):
        p['TermSpread_pct'] = p['US10Y_pct'] - p['US13W_pct']
    if {'HYG_px', 'IEF_px'}.issubset(p.columns):
        # credit-risk proxy: HY underperformance vs duration-matched Treasuries.
        # Rises when credit stress rises, i.e. same sign as a widening HY OAS.
        p['CreditStress'] = -(np.log(p['HYG_px']).diff() - np.log(p['IEF_px']).diff())

    # ---- simple derived predictors used by all three models ----------------
    p['AbsReturn'] = p['Return'].abs()
    p['NegReturn'] = p['Return'].clip(upper=0).abs()      # leverage / asymmetry term
    p['ParkinsonVar'] = (np.log(p['High'] / p['Low']) ** 2) / (4 * np.log(2))
    p['RangePct'] = 100 * (p['High'] - p['Low']) / p['Close']

    p = p.sort_values('Date').reset_index(drop=True)
    path = os.path.join(OUT, f'{code}_panel_daily.csv')
    p.to_csv(path, index=False, date_format='%Y-%m-%d', float_format='%.10g')

    rvfirst = p.loc[p['HasRV_t'], 'Date'].min() if p['HasRV_t'].any() else pd.NaT
    n_rv = int(p['HasRV_t'].sum())
    # coverage measured only from the day RV actually starts - before that it is not
    # "missing", it simply does not exist yet
    tail = p[p['Date'] >= rvfirst] if pd.notna(rvfirst) else p.iloc[0:0]
    cov = 100.0 * n_rv / max(len(tail), 1)
    print(f"  [{code}] {len(p):5d} rows {p.Date.min().date()} -> {p.Date.max().date()} | "
          f"RV on {n_rv} days ({cov:.1f}% of {rvfirst.date() if pd.notna(rvfirst) else '-'}+) | "
          f"{len(p.columns)} cols")
    return dict(Code=code, Rows=len(p), Columns=len(p.columns),
                First=str(p.Date.min().date()), Last=str(p.Date.max().date()),
                RV_Days=n_rv, RV_First=str(rvfirst.date()) if pd.notna(rvfirst) else "",
                RV_Coverage_Pct_Since_RV_Start=round(cov, 2),
                VolIdx=own, VolIdx_Fallback=fb or "",
                File=os.path.relpath(path, ROOT).replace("\\", "/"))


if __name__ == "__main__":
    codes = [a.upper() for a in sys.argv[1:]] or list(VOLMAP)
    rows = [r for r in (build(c) for c in codes) if r]
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(LOG, 'phase9_panel_summary.csv'), index=False)
    print("\n" + s.to_string(index=False))
    print("\nwrote Datasets/07_PANEL_INTERMEDIATE/ and _logs/phase9_panel_summary.csv")
