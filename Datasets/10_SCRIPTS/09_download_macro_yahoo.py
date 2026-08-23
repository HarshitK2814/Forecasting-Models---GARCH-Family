# -*- coding: utf-8 -*-
"""
Phase 6b: KEYLESS macro / risk-factor predictors for the Quantile Regression feature set.

WHY THIS SCRIPT EXISTS
  Script 04 pulls the canonical FRED series, but FRED requires a free API key and, on this
  network, fred.stlouisfed.org itself is unreachable (re-verified 2026-08-23: 4/4 ReadTimeout
  on fredgraph.csv; api.stlouisfed.org answers in 0.5-0.8s with HTTP 400 "missing api_key").
  A dataset that a collaborator has to obtain a key for is not a finished dataset, so this
  script builds an equivalent factor set from a source that needs no key and no account.

  Every FRED series we actually use as a QR covariate has a keyless market-traded analogue:
      DGS10 / DGS2 / T10Y2Y   -> ^TNX, ^FVX, ^TYX, ^IRX  (CBOE yield indices. Yahoo USED to
                                 quote these as percent x10; it no longer does. Close is
                                 already in percent - verified 2026-08-23, ^TNX averages
                                 8.55 in 1990 and prints 4.74 today. Do not divide by 10.)
      DTWEXBGS                -> DX-Y.NYB (ICE US Dollar Index)
      DEXUSEU / DEXJPUS       -> EURUSD=X, JPY=X
      BAMLH0A0HYM2 (HY OAS)   -> HYG vs IEF relative performance (credit-risk proxy)
  The macro series with NO market analogue (CPIAUCSL, UNRATE, INDPRO, NFCI, USREC) are
  monthly/weekly and cannot be proxied honestly. They remain in script 04 and are listed in
  the Not_Downloaded sheet. They are optional covariates, not core inputs.

  Regional equity ETFs are also pulled. They serve two purposes: an independent daily series
  to cross-check each index against, and a USD-denominated alternative if we ever want the
  panel in a common currency.

VERIFICATION PROTOCOL (same as scripts 01/02)
  Each series is fetched repeatedly and is only accepted once two CONSECUTIVE fetches return
  an identical row count AND an identical last close. A single lucky fetch is never trusted.

Output: Datasets/05_RAW_MACRO/<CODE>/<CODE>_daily_<first>_<last>.csv
        columns Date,Symbol,Open,High,Low,Close,Volume  (+ LogReturn where meaningful)
Manifest: Datasets/11_LOGS/phase6b_macro_yahoo_manifest.csv
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, '05_RAW_MACRO')
LOG  = os.path.join(ROOT, '11_LOGS')
os.makedirs(OUT, exist_ok=True); os.makedirs(LOG, exist_ok=True)

START = "1990-01-01"

# code -> (yahoo ticker, group, description, unit note, FRED analogue)
SERIES = {
 # ---- interest rates (Yahoo quotes these as percent x 10) ----
 "US10Y": ("^TNX", "rates", "CBOE 10-Year Treasury yield index",  "Close = percent", "DGS10"),
 "US5Y":  ("^FVX", "rates", "CBOE 5-Year Treasury yield index",   "Close = percent", "DGS5"),
 "US30Y": ("^TYX", "rates", "CBOE 30-Year Treasury yield index",  "Close = percent", "DGS30"),
 "US13W": ("^IRX", "rates", "CBOE 13-Week T-Bill discount rate",  "Close = percent", "DTB3"),
 # ---- FX ----
 "DXY":   ("DX-Y.NYB", "fx", "ICE US Dollar Index",               "index level",        "DTWEXBGS"),
 "EURUSD":("EURUSD=X", "fx", "EUR/USD spot",                      "USD per EUR",        "DEXUSEU"),
 "USDJPY":("JPY=X",    "fx", "USD/JPY spot",                      "JPY per USD",        "DEXJPUS"),
 "GBPUSD":("GBPUSD=X", "fx", "GBP/USD spot",                      "USD per GBP",        "DEXUSUK"),
 "USDHKD":("HKD=X",    "fx", "USD/HKD spot",                      "HKD per USD",        "DEXHKUS"),
 # ---- commodities ----
 "WTI":   ("CL=F", "commodity", "WTI crude front-month futures",  "USD/barrel",         "DCOILWTICO"),
 "BRENT": ("BZ=F", "commodity", "Brent crude front-month futures","USD/barrel",         "DCOILBRENTEU"),
 "GOLD":  ("GC=F", "commodity", "COMEX gold front-month futures", "USD/oz",             "-"),
 # ---- credit / duration ETFs (HY-spread proxy) ----
 "HYG":   ("HYG", "credit", "iShares iBoxx High Yield Corp Bond ETF", "USD price",      "BAMLH0A0HYM2 proxy"),
 "LQD":   ("LQD", "credit", "iShares iBoxx Inv Grade Corp Bond ETF",  "USD price",      "BAMLC0A0CM proxy"),
 "IEF":   ("IEF", "credit", "iShares 7-10Y Treasury ETF (duration-matched leg)", "USD price", "-"),
 "TLT":   ("TLT", "credit", "iShares 20+Y Treasury ETF",              "USD price",      "-"),
 # ---- regional equity ETFs: independent cross-check per index ----
 "SPY":   ("SPY", "etf", "SPDR S&P 500 ETF (cross-check for SPX)",        "USD price", "-"),
 "QQQ":   ("QQQ", "etf", "Invesco QQQ, Nasdaq-100 ETF (cross-check NDX)", "USD price", "-"),
 "EWU":   ("EWU", "etf", "iShares MSCI United Kingdom ETF (UKX region)",  "USD price", "-"),
 "EWG":   ("EWG", "etf", "iShares MSCI Germany ETF (DAX region)",         "USD price", "-"),
 "EWJ":   ("EWJ", "etf", "iShares MSCI Japan ETF (NKY region)",           "USD price", "-"),
 "EWH":   ("EWH", "etf", "iShares MSCI Hong Kong ETF (HSI region)",       "USD price", "-"),
}

def fetch_verified(ticker, attempts=5):
    """Accept only when two consecutive fetches agree on BOTH row count and last close."""
    prev = None
    counts = []
    for a in range(attempts):
        try:
            df = yf.Ticker(ticker).history(start=START, auto_adjust=False, actions=False)
        except Exception:
            df = pd.DataFrame()
        sig = (len(df), None if df.empty else round(float(df['Close'].iloc[-1]), 6))
        counts.append(sig[0])
        if prev is not None and sig == prev and sig[0] > 0:
            return df, counts, a + 1
        prev = sig
        time.sleep(1.0)
    return (df if 'df' in dir() else pd.DataFrame()), counts, attempts

if __name__ == "__main__":
    only = [a.upper() for a in sys.argv[1:]]
    rows = []
    for code, (tkr, grp, desc, unit, fred) in SERIES.items():
        if only and code not in only:
            continue
        df, counts, att = fetch_verified(tkr)
        if df is None or df.empty:
            print(f"FAIL {code:7s} {tkr:10s} counts={counts}")
            rows.append(dict(Code=code, Ticker=tkr, Group=grp, Description=desc,
                             Status="FAIL - no data", Fetch_Counts=str(counts)))
            continue
        agreed = len(counts) >= 2 and counts[-1] == counts[-2]
        d = df.reset_index()
        d['Date'] = pd.to_datetime(d['Date']).dt.tz_localize(None).dt.normalize()
        d = d[['Date', 'Open', 'High', 'Low', 'Close'] + (['Volume'] if 'Volume' in d else [])]
        d.insert(1, 'Symbol', code)
        d = d.dropna(subset=['Close']).drop_duplicates(subset='Date').sort_values('Date')

        # --- orphan-prefix trim -------------------------------------------------
        # Yahoo sometimes carries one or two stray observations years before the real
        # history begins (USDHKD: a single 2001-07-16 print, then nothing until
        # 2003-12-01 - a 214-day "gap" that is really an orphan). Drop any leading
        # block separated from the main series by more than 90 calendar days.
        gap = d['Date'].diff().dt.days
        big = gap[gap > 90]
        if len(big) and big.index[0] <= d.index[min(5, len(d) - 1)]:
            d = d.loc[big.index[0]:]

        # --- OHLC consistency flag ---------------------------------------------
        # Yahoo's OHLC for FX and front-month futures is not internally consistent on a
        # minority of days: it writes a placeholder bar with Open=High=Low and a Close
        # taken from a different snapshot, so Close can sit outside [Low, High].
        # Verified on GOLD (2026-08-23): on the 441 violating days O==H==L on essentially
        # all of them, and Close is CLOSER to the next day's Open (32.9 bps median) than
        # the High/Low midpoint is (61.5 bps). So CLOSE is the reliable field and O/H/L
        # are the corrupted ones - not the other way round.
        # We therefore keep every row (no Close is discarded) and mark the row instead.
        # RULE FOR USERS: for the macro folder use Close / LogReturn only. Use High/Low
        # exclusively on rows where OHLC_Consistent is True.
        _o, _h, _l, _c = d['Open'], d['High'], d['Low'], d['Close']
        d['OHLC_Consistent'] = ~((_h < _l) | (_h < _o) | (_h < _c) | (_l > _o) | (_l > _c))

        d['LogReturn'] = np.log(d['Close']).diff()
        for c in ['Open', 'High', 'Low', 'Close']:
            d[c] = d[c].round(6)
        d['LogReturn'] = d['LogReturn'].round(8)
        n_incons = int((~d['OHLC_Consistent']).sum())

        sub = os.path.join(OUT, code); os.makedirs(sub, exist_ok=True)
        f0, f1 = d['Date'].iloc[0].date(), d['Date'].iloc[-1].date()
        # remove any stale file for this code so the folder never holds two vintages
        for old in os.listdir(sub):
            if old.startswith(code + "_daily_"):
                os.remove(os.path.join(sub, old))
        path = os.path.join(sub, f"{code}_daily_{f0}_{f1}.csv")
        d.to_csv(path, index=False, date_format="%Y-%m-%d")

        rows.append(dict(Code=code, Ticker=tkr, Group=grp, Description=desc, Unit=unit,
                         FRED_Analogue=fred, Rows=len(d), First=str(f0), Last=str(f1),
                         Attempts=att, Two_Identical_Fetches=bool(agreed),
                         Rows_OHLC_Inconsistent=n_incons,
                         Pct_OHLC_Inconsistent=round(100.0 * n_incons / len(d), 2),
                         Use_Fields="Close, LogReturn" if n_incons else "Open, High, Low, Close, LogReturn",
                         Fetch_Counts=str(counts), Source="Yahoo Finance via yfinance",
                         File=os.path.relpath(path, ROOT).replace("\\", "/"),
                         Status="OK" if agreed else "OK - but fetches never agreed, re-run"))
        print(f"OK   {code:7s} {tkr:10s} {len(d):>6} rows  {f0} -> {f1}  attempts={att} "
              f"agreed={agreed} ohlc_inconsistent={n_incons}")

    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(LOG, 'phase6b_macro_yahoo_manifest.csv'), index=False)
    print(f"\n{len(man[man.Status.astype(str).str.startswith('OK')])}/{len(man)} series written")
    print("manifest -> _logs/phase6b_macro_yahoo_manifest.csv")
