"""
Phase 6: FRED macro predictors for the Quantile Regression feature set.

STATUS: NOT PRE-DOWNLOADED. Requires a free FRED API key (30 seconds to obtain).
Verified 2026-08-22: fred.stlouisfed.org (website / keyless fredgraph.csv) is UNREACHABLE
from this network - 4/4 attempts timed out. api.stlouisfed.org IS reachable and returned
HTTP 400 "missing api_key", i.e. the API path works and only needs a key.

HOW TO RUN
  1. Get a free key: https://fredaccount.stlouisfed.org/apikeys
  2. set FRED_API_KEY=your_key_here      (Windows cmd)
     $env:FRED_API_KEY="your_key_here"   (PowerShell)
  3. python Datasets/10_SCRIPTS/04_download_macro_FRED.py
Output: Datasets/05_RAW_MACRO/<SERIES>/<SERIES>_<freq>_<first>_<last>.csv
"""
import os, sys, time, requests, pandas as pd

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=os.path.join(ROOT,'05_RAW_MACRO'); LOG=os.path.join(ROOT,'11_LOGS')
KEY=os.environ.get("FRED_API_KEY")

SERIES={
 "DGS10":       ("Daily","10-Year Treasury constant maturity yield"),
 "DGS2":        ("Daily","2-Year Treasury constant maturity yield"),
 "T10Y2Y":      ("Daily","10Y-2Y Treasury spread (term spread)"),
 "T10Y3M":      ("Daily","10Y-3M Treasury spread (recession signal)"),
 "DFF":         ("Daily","Effective Federal Funds Rate"),
 "BAMLH0A0HYM2":("Daily","ICE BofA US High Yield option-adjusted credit spread"),
 "BAMLC0A0CM":  ("Daily","ICE BofA US Corporate option-adjusted spread"),
 "DTWEXBGS":    ("Daily","Trade-weighted US Dollar index (broad goods+services)"),
 "DEXUSEU":     ("Daily","USD/EUR spot exchange rate"),
 "DEXJPUS":     ("Daily","JPY/USD spot exchange rate"),
 "TEDRATE":     ("Daily","TED spread (discontinued 2022; funding stress pre-2022)"),
 "NFCI":        ("Weekly","Chicago Fed National Financial Conditions Index"),
 "STLFSI4":     ("Weekly","St. Louis Fed Financial Stress Index"),
 "CPIAUCSL":    ("Monthly","CPI all urban consumers (lag 1 month when merging)"),
 "UNRATE":      ("Monthly","Civilian unemployment rate (lag 1 month)"),
 "INDPRO":      ("Monthly","Industrial production index (lag 1 month)"),
 "USREC":       ("Monthly","NBER recession indicator (regime label)"),
}

def fetch(sid, key, attempts=4):
    """Fetch with retries; accept only when two consecutive calls agree on row count."""
    url="https://api.stlouisfed.org/fred/series/observations"
    last=None
    for a in range(attempts):
        try:
            r=requests.get(url, params=dict(series_id=sid, api_key=key, file_type="json"), timeout=40)
            r.raise_for_status()
            obs=r.json().get("observations",[])
            if last is not None and len(obs)==last: return obs, a+1
            last=len(obs); keep=obs
        except Exception:
            time.sleep(1.5*(a+1))
    if last is None: raise RuntimeError(f"{sid}: all attempts failed")
    return keep, attempts

if __name__=="__main__":
    if not KEY:
        print(__doc__); print("ERROR: FRED_API_KEY not set. Nothing downloaded."); sys.exit(1)
    rows=[]
    for sid,(freq,desc) in SERIES.items():
        try:
            obs,att=fetch(sid,KEY)
            df=pd.DataFrame(obs)[["date","value"]]
            df["date"]=pd.to_datetime(df["date"])
            df["value"]=pd.to_numeric(df["value"].replace(".",None),errors="coerce")
            df=df.dropna(subset=["value"]).rename(columns={"date":"Date","value":sid})
            d=os.path.join(OUT,sid); os.makedirs(d,exist_ok=True)
            f0,f1=df["Date"].iloc[0].date(),df["Date"].iloc[-1].date()
            p=os.path.join(d,f"{sid}_{freq.lower()}_{f0}_{f1}.csv")
            df.to_csv(p,index=False,date_format="%Y-%m-%d")
            rows.append(dict(series=sid,freq=freq,desc=desc,rows=len(df),first=str(f0),last=str(f1),
                             attempts=att,status="OK",file=os.path.relpath(p,ROOT)))
            print(f"OK   {sid:14s} {len(df):>6} rows  {f0} -> {f1}")
        except Exception as e:
            rows.append(dict(series=sid,freq=freq,desc=desc,status=f"FAIL {e}"))
            print(f"FAIL {sid:14s} {e}")
    pd.DataFrame(rows).to_csv(os.path.join(LOG,'phase6_macro_manifest.csv'),index=False)
