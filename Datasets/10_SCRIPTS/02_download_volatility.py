"""
Phase 3: Volatility / implied-vol indices.
Sources: CBOE (cdn.cboe.com CSV), STOXX (h_*.txt), Yahoo (yfinance).
Output: Datasets/04_RAW_VOLATILITY/<CODE>/<CODE>_daily_<first>_<last>.csv
Unified schema: Date,Symbol,Open,High,Low,Close  (Open/High/Low blank if source is close-only)
"""
import os, io, ssl, certifi, warnings, urllib.request
os.environ['SSL_CERT_FILE']=certifi.where(); os.environ['REQUESTS_CA_BUNDLE']=certifi.where()
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, yfinance as yf

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=os.path.join(ROOT,'04_RAW_VOLATILITY'); LOG=os.path.join(ROOT,'11_LOGS')
CTX=ssl.create_default_context(cafile=certifi.where())
NOVERIFY=ssl.create_default_context(); NOVERIFY.check_hostname=False; NOVERIFY.verify_mode=ssl.CERT_NONE

def http(u, ctx=CTX, timeout=60):
    req=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'})
    return urllib.request.urlopen(req,timeout=timeout,context=ctx).read().decode('utf-8','replace')

# code -> (kind, locator, tracks_index, description)
CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
SPECS = {
 "VIX":   ("cboe","VIX",    "SPX", "CBOE Volatility Index (S&P 500, 30d IV)"),
 "VXN":   ("cboe","VXN",    "NDX", "CBOE Nasdaq-100 Volatility Index"),
 "VXEFA": ("cboe","VXEFA",  "UKX/NKY","CBOE EFA ETF Vol Index - developed mkts ex-US proxy"),
 "VXEEM": ("cboe","VXEEM",  "HSI", "CBOE Emerging Markets ETF Vol Index - HSI proxy"),
 "VVIX":  ("cboe","VVIX",   "global","VIX of VIX (vol-of-vol)"),
 "SKEW":  ("cboe","SKEW",   "SPX", "CBOE SKEW Index (tail risk)"),
 "VIX9D": ("cboe","VIX9D",  "SPX", "9-day VIX (term structure short end)"),
 "VIX3M": ("cboe","VIX3M",  "SPX", "3-month VIX"),
 "VIX6M": ("cboe","VIX6M",  "SPX", "6-month VIX"),
 "RVX":   ("cboe","RVX",    "US small cap","Russell 2000 Volatility Index"),
 "VXD":   ("cboe","VXD",    "US","DJIA Volatility Index"),
 "OVX":   ("cboe","OVX",    "oil","Crude Oil ETF Volatility Index"),
 "GVZ":   ("cboe","GVZ",    "gold","Gold ETF Volatility Index"),
 "V2TX":  ("stoxx","h_v2tx.txt","EuroStoxx/UKX","VSTOXX - EURO STOXX 50 Volatility"),
 "V1X":   ("stoxx","h_v1x.txt", "DAX","VDAX-NEW - DAX Volatility Index"),
 "V6I1":  ("stoxx","h_v6i1.txt","EuroStoxx","VSTOXX 1-month sub-index"),
 "NKVI":  ("yahoo","^NKVI.OS","NKY","Nikkei 225 Volatility Index"),
}

def load_cboe(sym, attempts=4):
    last=None
    for i in range(attempts):
        try:
            txt=http(CBOE.format(sym))
            df=pd.read_csv(io.StringIO(txt))
            if last is not None and len(df)==last: return df, i+1
            last=len(df)
        except Exception as e:
            last=None
    if last is None: raise RuntimeError("cboe fetch failed")
    return df, attempts

def load_stoxx(fname, attempts=4):
    last=None
    for i in range(attempts):
        try:
            txt=http("https://www.stoxx.com/document/Indices/Current/HistoricalData/"+fname, ctx=NOVERIFY)
            df=pd.read_csv(io.StringIO(txt), sep=';')
            if last is not None and len(df)==last: return df, i+1
            last=len(df)
        except Exception:
            last=None
    if last is None: raise RuntimeError("stoxx fetch failed")
    return df, attempts

def load_yahoo(tkr, attempts=4):
    last=None; df=None
    for i in range(attempts):
        d=yf.Ticker(tkr).history(period="max", auto_adjust=False)
        if d is None or d.empty: continue
        if last is not None and len(d)==last: return d, i+1
        last=len(d); df=d
    if df is None: raise RuntimeError("yahoo fetch failed")
    return df, attempts

rows=[]
for code,(kind,loc,tracks,desc) in SPECS.items():
    try:
        if kind=="cboe":
            raw,att = load_cboe(loc)
            raw.columns=[c.strip().upper() for c in raw.columns]
            dt=pd.to_datetime(raw["DATE"], format="%m/%d/%Y", errors="coerce")
            if dt.isna().mean()>0.5: dt=pd.to_datetime(raw["DATE"], errors="coerce")
            out=pd.DataFrame({"Date":dt})
            if {"OPEN","HIGH","LOW","CLOSE"}.issubset(raw.columns):
                out["Open"]=pd.to_numeric(raw["OPEN"],errors="coerce")
                out["High"]=pd.to_numeric(raw["HIGH"],errors="coerce")
                out["Low"] =pd.to_numeric(raw["LOW"], errors="coerce")
                out["Close"]=pd.to_numeric(raw["CLOSE"],errors="coerce")
            else:
                vcol=[c for c in raw.columns if c!="DATE"][0]
                out["Open"]=np.nan; out["High"]=np.nan; out["Low"]=np.nan
                out["Close"]=pd.to_numeric(raw[vcol],errors="coerce")
            src=CBOE.format(loc)
        elif kind=="stoxx":
            raw,att = load_stoxx(loc)
            raw.columns=[c.strip() for c in raw.columns]
            dt=pd.to_datetime(raw["Date"], format="%d.%m.%Y", errors="coerce")
            out=pd.DataFrame({"Date":dt,"Open":np.nan,"High":np.nan,"Low":np.nan,
                              "Close":pd.to_numeric(raw["Indexvalue"],errors="coerce")})
            src="https://www.stoxx.com/document/Indices/Current/HistoricalData/"+loc
        else:
            raw,att = load_yahoo(loc)
            idx=pd.to_datetime(raw.index).tz_localize(None).normalize()
            out=pd.DataFrame({"Date":idx,"Open":raw["Open"].values,"High":raw["High"].values,
                              "Low":raw["Low"].values,"Close":raw["Close"].values})
            src=f"yfinance ticker {loc}"

        out=out.dropna(subset=["Date","Close"]).drop_duplicates("Date",keep="last").sort_values("Date")
        out.insert(1,"Symbol",code)
        d=os.path.join(OUT,code); os.makedirs(d,exist_ok=True)
        f0,f1=out["Date"].iloc[0].date(), out["Date"].iloc[-1].date()
        path=os.path.join(d,f"{code}_daily_{f0}_{f1}.csv")
        out.to_csv(path,index=False,date_format="%Y-%m-%d",float_format="%.4f")
        rows.append(dict(code=code,kind=kind,tracks=tracks,desc=desc,rows=len(out),
                         first=str(f0),last=str(f1),has_ohlc=bool(out["Open"].notna().any()),
                         attempts_to_stable=att,source_url=src,file=os.path.relpath(path,ROOT),status="OK"))
        print(f"OK   {code:6s} rows={len(out):>6} {f0} -> {f1} ohlc={out['Open'].notna().any()}")
    except Exception as e:
        rows.append(dict(code=code,kind=kind,tracks=tracks,desc=desc,status=f"FAIL {type(e).__name__}"))
        print(f"FAIL {code:6s} {type(e).__name__}: {str(e)[:60]}")

pd.DataFrame(rows).to_csv(os.path.join(LOG,'phase3_volatility_manifest.csv'),index=False)
print("\nManifest -> _logs/phase3_volatility_manifest.csv")
