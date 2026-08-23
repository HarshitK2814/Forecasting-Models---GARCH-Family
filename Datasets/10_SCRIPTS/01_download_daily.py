"""
Phase 2: Daily index OHLCV for the 6 Tier-1 indices.
Source: Yahoo Finance via yfinance (free, no API key).
Output: Datasets/02_RAW_DAILY/<CODE>/<CODE>_daily_<first>_<last>.csv
"""
import os, sys, certifi, warnings, json
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
warnings.filterwarnings('ignore')
import pandas as pd, yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, '02_RAW_DAILY')
LOG  = os.path.join(ROOT, '11_LOGS')

INDICES = {
    "SPX":  ("^GSPC",  "S&P 500",      "United States", "USD", "America/New_York"),
    "NDX":  ("^NDX",   "Nasdaq 100",   "United States", "USD", "America/New_York"),
    "UKX":  ("^FTSE",  "FTSE 100",     "United Kingdom","GBP", "Europe/London"),
    "DAX":  ("^GDAXI", "DAX 40",       "Germany",       "EUR", "Europe/Berlin"),
    "NKY":  ("^N225",  "Nikkei 225",   "Japan",         "JPY", "Asia/Tokyo"),
    "HSI":  ("^HSI",   "Hang Seng",    "Hong Kong",     "HKD", "Asia/Hong_Kong"),
}
START = "1990-01-01"

def fetch(ticker, attempts=4):
    """Fetch with retries; require 2 consecutive identical row counts to accept."""
    results = []
    for i in range(attempts):
        try:
            df = yf.Ticker(ticker).history(start=START, auto_adjust=False, actions=False)
            if df is None or df.empty:
                results.append((0, None)); continue
            results.append((len(df), df))
            if len(results) >= 2 and results[-1][0] == results[-2][0] and results[-1][0] > 0:
                return df, [r[0] for r in results]
        except Exception as e:
            results.append((-1, None))
    good = [r for r in results if r[0] > 0]
    if not good: return None, [r[0] for r in results]
    return max(good, key=lambda r: r[0])[1], [r[0] for r in results]

summary = []
for code, (tkr, name, country, ccy, tz) in INDICES.items():
    df, counts = fetch(tkr)
    if df is None:
        print(f"FAIL {code} ({tkr}) counts={counts}"); 
        summary.append(dict(code=code, ticker=tkr, status="FAIL", attempts=str(counts)))
        continue

    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    df = df[keep]
    df = df[~df.index.duplicated(keep='last')].sort_index()
    # drop rows with no close
    df = df[df["Close"].notna()]
    # log return
    df["LogReturn"] = (df["Close"] / df["Close"].shift(1)).apply(lambda x: pd.NA if pd.isna(x) or x<=0 else x)
    import numpy as np
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df.insert(0, "Symbol", code)

    d = os.path.join(OUT, code); os.makedirs(d, exist_ok=True)
    f0, f1 = df.index[0].date(), df.index[-1].date()
    path = os.path.join(d, f"{code}_daily_{f0}_{f1}.csv")
    df.to_csv(path, date_format="%Y-%m-%d", float_format="%.6f")

    summary.append(dict(code=code, ticker=tkr, name=name, country=country, currency=ccy,
                        tz=tz, rows=len(df), first=str(f0), last=str(f1),
                        cols=";".join(df.columns), status="OK",
                        attempts=str(counts), file=os.path.relpath(path, ROOT)))
    print(f"OK   {code:4s} {tkr:8s} rows={len(df):>6} {f0} -> {f1}  attempts={counts}")

pd.DataFrame(summary).to_csv(os.path.join(LOG,'phase2_daily_manifest.csv'), index=False)
print("\nManifest -> _logs/phase2_daily_manifest.csv")
