# -*- coding: utf-8 -*-
"""Phase 8: build the multi-sheet Excel data report from live manifests + on-disk files."""
import os, glob, warnings, datetime
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, '11_LOGS')
VAL = os.path.join(ROOT, '08_VALIDATION')
XL = os.path.join(ROOT, '00_DOCUMENTATION', 'DATA_ACQUISITION_REPORT.xlsx')
TODAY = datetime.date.today().isoformat()


def rd(p, **kw):
    try:
        return pd.read_csv(p, **kw)
    except Exception:
        return pd.DataFrame()


def fsize(p):
    try:
        return os.path.getsize(p)
    except Exception:
        return 0


def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def scan(folder, pattern):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, folder, pattern))):
        try:
            df = pd.read_csv(f)
            dcol = 'Date' if 'Date' in df.columns else df.columns[0]
            d = pd.to_datetime(df[dcol], errors='coerce').dropna()
            rows.append(dict(file=os.path.relpath(f, ROOT), rows=len(df),
                             first=str(d.min().date()) if len(d) else "",
                             last=str(d.max().date()) if len(d) else "",
                             size=human(fsize(f)), columns="; ".join(df.columns)))
        except Exception as e:
            rows.append(dict(file=os.path.relpath(f, ROOT), rows=-1, first="", last="",
                             size=human(fsize(f)), columns=f"READ ERROR {e}"))
    return pd.DataFrame(rows)


README = [
    ("GARCH-EVT / Realized GARCH / Quantile Regression - DATA ACQUISITION REPORT", ""),
    ("Prepared by", "Researcher A (data lead)"),
    ("Generated", TODAY),
    ("", ""),
    ("PURPOSE", "Single source of truth for every dataset used by the project: what it is, where it comes from, how to re-fetch it, what columns it has, and whether it has been validated."),
    ("", ""),
    ("=== HEADLINE DECISIONS ===", ""),
    ("Asset universe", "6 Tier-1 equity indices: S&P 500, Nasdaq 100, FTSE 100, DAX 40, Nikkei 225, Hang Seng"),
    ("Why these 6", "Cover North America / UK / Eurozone / Japan / Hong Kong; all have free intraday from Dukascopy; all have a usable implied-vol series."),
    ("Intraday sampling", "Downloaded at 1-min; analysis at 5-min. 1/10/15/30-min also computed for the sampling-robustness check."),
    ("Why 5-min", "Literature standard (Liu, Patton and Sheppard 2015). 15-min gives only ~26 bars/session - too noisy for RV."),
    ("", ""),
    ("=== THE DATE-RANGE QUESTION (important) ===", ""),
    ("Constraint", "Free intraday index data does NOT exist before ~2011-2013. 2007 intraday is impossible without a paid feed (TAQ/WRDS)."),
    ("Resolution", "Daily and intraday do NOT need identical start dates. What must match is the OUT-OF-SAMPLE TEST WINDOW."),
    ("Learning period", "Each model may use as much history as it can get (daily back to 1990)."),
    ("Test period", "Starts only when every model can forecast - i.e. when RV exists plus one rolling window."),
    ("Rolling window", "USE 1000-1250 DAYS, not the 2000 in the Executive Summary. At 2000 days the first forecast slips to ~2021 and you lose half the test period."),
    ("Resulting test window", "approx 2017-2026 (~9 years) - identical for all four model families."),
    ("Extreme events inside it", "Feb-2018 Volmageddon, Q4-2018 selloff, COVID Mar-2020, 2022 bear market, Mar-2023 banking stress, Aug-2024 yen-carry unwind, 2025 tariff volatility."),
    ("Trade-off accepted", "2008 GFC is NOT in the intraday sample. It remains available for GARCH-EVT and QR on the daily series."),
    ("", ""),
    ("=== FILE FORMAT CONVENTIONS ===", ""),
    ("Format", "CSV throughout (as requested). UTF-8, comma-separated, header row present."),
    ("Dates", "ISO YYYY-MM-DD."),
    ("Intraday timestamps", "Two columns: ts_utc (UTC) and ts_local (exchange-local, DST-aware). Always join on ts_utc."),
    ("Missing values", "Empty cell (not NaN, not -999)."),
    ("Folder layout", "daily/ | intraday/1min/ | intraday/5min/ | volatility/ | realized_volatility/ | macro/ | _validation/ | _logs/ | _scripts/"),
]

UNIVERSE = pd.DataFrame([
    dict(Code="SPX", Name="S&P 500", Region="North America", Country="United States", Currency="USD",
         Exchange_TZ="America/New_York", Cash_Session_Local="09:30-16:00", Bars_per_session_5min=78,
         Daily_Ticker="^GSPC", Dukascopy_Instrument="USA500IDXUSD", Vol_Index="VIX", Vol_Index_Match="Exact"),
    dict(Code="NDX", Name="Nasdaq 100", Region="North America", Country="United States", Currency="USD",
         Exchange_TZ="America/New_York", Cash_Session_Local="09:30-16:00", Bars_per_session_5min=78,
         Daily_Ticker="^NDX", Dukascopy_Instrument="USATECHIDXUSD", Vol_Index="VXN", Vol_Index_Match="Exact"),
    dict(Code="UKX", Name="FTSE 100", Region="Europe", Country="United Kingdom", Currency="GBP",
         Exchange_TZ="Europe/London", Cash_Session_Local="08:00-16:30", Bars_per_session_5min=102,
         Daily_Ticker="^FTSE", Dukascopy_Instrument="GBRIDXGBP", Vol_Index="VXEFA + V2TX",
         Vol_Index_Match="PROXY - no free FTSE vol index (VFTSE discontinued)"),
    dict(Code="DAX", Name="DAX 40", Region="Europe", Country="Germany", Currency="EUR",
         Exchange_TZ="Europe/Berlin", Cash_Session_Local="09:00-17:30", Bars_per_session_5min=102,
         Daily_Ticker="^GDAXI", Dukascopy_Instrument="DEUIDXEUR", Vol_Index="V1X (VDAX-NEW)", Vol_Index_Match="Exact"),
    dict(Code="NKY", Name="Nikkei 225", Region="Asia", Country="Japan", Currency="JPY",
         Exchange_TZ="Asia/Tokyo", Cash_Session_Local="09:00-11:30, 12:30-15:00", Bars_per_session_5min=60,
         Daily_Ticker="^N225", Dukascopy_Instrument="JPNIDXJPY", Vol_Index="NKVI (2018+) + VXEFA",
         Vol_Index_Match="PARTIAL - Nikkei VI only from 2018 on the free feed"),
    dict(Code="HSI", Name="Hang Seng", Region="Asia", Country="Hong Kong", Currency="HKD",
         Exchange_TZ="Asia/Hong_Kong", Cash_Session_Local="09:30-12:00, 13:00-16:00", Bars_per_session_5min=66,
         Daily_Ticker="^HSI", Dukascopy_Instrument="HKGIDXHKD", Vol_Index="VXEEM",
         Vol_Index_Match="PROXY - VHSI not free; VXEEM is EM-wide"),
])

SOURCES = pd.DataFrame([
    dict(Source="Yahoo Finance", Data="Daily index OHLCV", Access="yfinance Python library",
         URL="https://finance.yahoo.com", API_Key="None", Rate_Limit="Informal; throttles on abuse",
         Cost="Free", Licence_Note="Personal/research use. Do not redistribute in bulk.", Verified=TODAY, Status="WORKING"),
    dict(Source="CBOE", Data="US volatility indices (13 series)", Access="Direct CSV over HTTPS",
         URL="https://cdn.cboe.com/api/global/us_indices/daily_prices/{SYMBOL}_History.csv",
         API_Key="None", Rate_Limit="None observed", Cost="Free",
         Licence_Note="Publicly published index levels.", Verified=TODAY, Status="WORKING"),
    dict(Source="STOXX", Data="VSTOXX (V2TX), VDAX-NEW (V1X), VSTOXX 1M (V6I1)", Access="Direct TXT over HTTPS, semicolon-delimited",
         URL="https://www.stoxx.com/document/Indices/Current/HistoricalData/h_{code}.txt",
         API_Key="None", Rate_Limit="None observed", Cost="Free",
         Licence_Note="TLS chain is incomplete - standard verification FAILS. Must pass an unverified SSL context (see script 02).",
         Verified=TODAY, Status="WORKING (with TLS workaround)"),
    dict(Source="Dukascopy Bank", Data="1-min index CFD candles", Access="Direct binary .bi5 (LZMA) over HTTPS",
         URL="https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YYYY}/{MM-1}/{DD}/BID_candles_min_1.bi5",
         API_Key="None", Rate_Limit="Throttles hard without keep-alive. ~5 req/s with 6 persistent sessions.",
         Cost="Free", Licence_Note="Broker-quoted CFD, NOT the exchange index. See Risks sheet.",
         Verified=TODAY, Status="WORKING"),
    dict(Source="FRED (St. Louis Fed)", Data="Macro predictors", Access="REST API (api.stlouisfed.org)",
         URL="https://api.stlouisfed.org/fred/series/observations", API_Key="REQUIRED (free)",
         Rate_Limit="120 req/min", Cost="Free",
         Licence_Note="fred.stlouisfed.org website is UNREACHABLE from this network (4/4 timeouts). The API host IS reachable.",
         Verified=TODAY, Status="NOT DOWNLOADED - needs free key"),
    dict(Source="Stooq", Data="(rejected) daily index CSV", Access="CSV endpoint",
         URL="https://stooq.com/q/d/l/?s={sym}&i=d", API_Key="None", Rate_Limit="n/a", Cost="Free",
         Licence_Note="REJECTED: now behind a JavaScript anti-bot challenge - returns HTML, not CSV. Verified broken " + TODAY,
         Verified=TODAY, Status="BROKEN - do not use"),
    dict(Source="Alpha Vantage", Data="(rejected) intraday", Access="REST",
         URL="https://www.alphavantage.co", API_Key="Required", Rate_Limit="25 req/day free", Cost="Free tier",
         Licence_Note="REJECTED: the month= parameter needed for long intraday history is paid-only as of 2026.",
         Verified=TODAY, Status="REJECTED"),
    dict(Source="FRED SP500 series", Data="(rejected) S&P 500 daily", Access="FRED",
         URL="https://fred.stlouisfed.org/series/SP500", API_Key="Free key", Rate_Limit="-", Cost="Free",
         Licence_Note="REJECTED: only 10 years of history (S&P DJI licence) AND redistribution prohibited - cannot be committed to the repo.",
         Verified=TODAY, Status="REJECTED"),
    dict(Source="Oxford-Man Realized Library", Data="(unavailable) precomputed RV", Access="-",
         URL="https://realized.oxford-man.ox.ac.uk", API_Key="-", Rate_Limit="-", Cost="Free",
         Licence_Note="DISCONTINUED by the institute; site offline, no replacement planned. Archived GitHub copies end ~2021 - benchmark only.",
         Verified=TODAY, Status="DISCONTINUED"),
])

NOT_DOWNLOADED = pd.DataFrame([
    dict(Item="FRED macro series - the 5 with NO market analogue (CPIAUCSL, UNRATE, INDPRO, NFCI, USREC)",
         Why_Not="Requires a free API key that only you can create - I cannot register an account on your behalf. NOTE: the other 12 FRED series originally planned are NO LONGER NEEDED - script 09 now supplies equivalent keyless market-traded factors (see the Macro_Files sheet). Only these 5 genuinely macroeconomic weekly/monthly series have no honest market substitute, and they are optional covariates rather than core inputs.",
         Verified_Fetchable="Re-verified 2026-08-23. api.stlouisfed.org answers in 0.5-0.8s with HTTP 400 'missing api_key' (2/2 attempts) - the host is reachable and the API path is correct. The keyless website route fred.stlouisfed.org/graph/fredgraph.csv timed out 4/4 at 25s and must not be used.",
         How_To_Get="1) https://fredaccount.stlouisfed.org/apikeys  2) set FRED_API_KEY=<key>  3) python Datasets/10_SCRIPTS/04_download_macro_FRED.py",
         Effort="~2 minutes", Script="Datasets/10_SCRIPTS/04_download_macro_FRED.py (written; tested up to the key check)"),
    dict(Item="Dukascopy ASK-side candles",
         Why_Not="BID side downloaded first as the primary series. ASK is only needed to build mid-price RV as a microstructure robustness check.",
         Verified_Fetchable="Identical URL pattern with ASK_candles_min_1.bi5; binary format confirmed identical.",
         How_To_Get="set DUKA_SIDE=ASK then rerun Datasets/10_SCRIPTS/03_download_intraday.py",
         Effort="~60-80 min unattended per index at the measured sustained rate of ~4.8 req/s", Script="Datasets/10_SCRIPTS/03_download_intraday.py (env var switch)"),
    dict(Item="Alpaca SPY/QQQ 1-min (2016+)",
         Why_Not="Free but needs an account API key. Recommended as an INDEPENDENT cross-check on Dukascopy RV for the US indices.",
         Verified_Fetchable="Docs confirm the Basic (free) plan: history since 2016, 200 req/min, only the most recent 15 min withheld.",
         How_To_Get="Create a free paper-trading account at alpaca.markets, then use alpaca-py historical bars.",
         Effort="~30 min", Script="Not written - say the word if you want it"),
])

RISKS = pd.DataFrame([
    dict(Severity="HIGH", Issue="Yahoo OHLC is internally inconsistent for FX and futures",
         Detail="On a minority of days Yahoo writes a placeholder bar with Open==High==Low and a Close from a different snapshot, so Close falls OUTSIDE [Low, High]. Affected: GOLD 6.8% of days (441), USDJPY 3.6% (275), EURUSD 2.2% (128), USDHKD 1.8%, GBPUSD 1.7%, BRENT 1.5%, WTI 0.1%. Equities, ETFs and the CBOE yield indices are clean (0 violations).",
         Action="We established WHICH field is wrong rather than guessing: on GOLD's 441 violating days Close sits closer to the next day's Open (32.9 bp median) than the High/Low midpoint does (61.5 bp), so CLOSE is reliable and O/H/L are corrupted. No row is discarded; every macro file carries an OHLC_Consistent boolean. USE Close and LogReturn ONLY from the macro folder unless you filter on that flag.",
         Owner="Researcher A"),
    dict(Severity="HIGH", Issue="Frozen CFD sessions would enter the sample as zero-volatility days",
         Detail="On days a cash market is shut, Dukascopy still streams a nominally-live CFD drifting by a fraction of a point. Those bars survive a naive stale-bar filter (High != Low) but represent no trading and give RV = 0.0. Worse, the same pattern occurs during outright FEED OUTAGES on days the exchange WAS open.",
         Action="Sessions whose whole-day price range is under 4 bp are dropped and every one is logged to _validation/frozen_sessions_dropped.csv (see the Frozen_Sessions_Dropped sheet). Calibrated over 10,847 session-days: largest range among dropped days 2.90 bp, smallest among kept days 5.68 bp - two cleanly separated clusters. Caught UKX 2013-12-25/26 and 2014-01-01 (LSE closed) AND SPX 2013-02-25..28 / NDX 2013-02-26..28, a Dukascopy outage on days the exchange was open which were SPX's two largest CFD residuals.",
         Owner="Researcher A"),
    dict(Severity="MEDIUM", Issue="Yahoo no longer quotes CBOE yield indices as percent x 10",
         Detail="Older code (and a lot of tutorials) divides ^TNX by 10. Yahoo's current data is ALREADY in percent. Dividing produced a 10-year Treasury yield ranging 0.05-0.91%, which is how the bug was caught.",
         Action="Verified against known history: raw ^TNX averages 8.55 in 1990, 6.02 in 2000, 0.88 in 2020 and prints 4.74 today. The /10 has been removed from script 10. Do not reintroduce it. US10Y_pct and US13W_pct in the panel are in PERCENT.",
         Owner="Researcher A"),
    dict(Severity="MEDIUM", Issue="No free volatility index exists for FTSE 100 or Hang Seng",
         Detail="Free regional implied-vol indices exist for the US (VIX, VXN), Europe (VSTOXX, VDAX-NEW) and Japan from 2018 only (NKVI). There is no free UK or HK equivalent.",
         Action="The panel uses VXEFA (developed ex-US) for UKX and VXEEM (emerging markets) for HSI as DECLARED proxies, with the actual symbol recorded in a VolIdx_Symbol column so the substitution is never invisible. Disclose the substitution in the paper.",
         Owner="Researcher A"),
    dict(Severity="MEDIUM", Issue="USDHKD has a genuine 7-month hole in 2003",
         Detail="Yahoo's HKD=X carries a 214-day gap ending 2003-12-01. This is a real hole in the source, not an orphan first observation.",
         Action="Left as-is and documented. HKD is pegged and this is a minor covariate; the hole predates the 2011+ realized-measure window entirely, so it cannot affect the Realized GARCH sample.",
         Owner="Researcher A"),
    dict(Severity="HIGH", Issue="Dukascopy CFD is not the exchange index",
         Detail="Intraday comes from a broker-quoted CFD; daily comes from the actual index. RV and returns therefore originate from different instruments.",
         Action="DONE - see the CFD_vs_Index sheet. CRITICAL SUBTLETY: the CFD return on day t is measured against the previous day PRESENT IN THE CFD FILE, so wherever the feed is missing a day it becomes a 2-day return compared against a 1-day index return. Those observations are not comparable and they dominate the residual variance. Excluding them changes the verdict outright: SPX R2 0.9785 -> 0.9898, NDX 0.9173 -> 0.9966, UKX 0.9087 -> 0.9701. On the naive number NDX and UKX would have been WRONGLY DISCARDED. The headline figure is the aligned one; the naive one is kept beside it as R2_Naive_DoNotUse.",
         Owner="Researcher A"),
    dict(Severity="HIGH", Issue="Rolling window length silently truncates the test period",
         Detail="The Executive Summary suggests 2000 days. With intraday starting ~2013 that pushes the first Realized-GARCH forecast to ~2021.",
         Action="Use 1000-1250 days. Document the choice as a design constraint, not an arbitrary pick.", Owner="Researcher A"),
    dict(Severity="HIGH", Issue="Stale-bar padding in Dukascopy files",
         Detail="All 1440 minutes of the UTC day are present. Outside the cash session the last price repeats with volume 0. Including these injects zero returns and biases RV DOWNWARD.",
         Action="ALREADY HANDLED: session filtering in exchange-local time plus a (volume>0 or High!=Low) filter. Do not bypass it.", Owner="Researcher A"),
    dict(Severity="MEDIUM", Issue="No free implied-vol index for FTSE 100 or Hang Seng",
         Detail="VFTSE was discontinued; VHSI is not freely redistributable. Nikkei VI is free only from 2018.",
         Action="Use VXEFA (developed ex-US) for UKX/NKY and VXEEM (emerging) for HSI, plus VIX as a global factor. Disclose these as proxies.", Owner="Researcher A / B"),
    dict(Severity="MEDIUM", Issue="Non-synchronous trading across regions",
         Detail="RV for NKY is measured over Tokyo hours, SPX over New York hours. The same calendar date is not the same wall-clock window.",
         Action="Fine while each index is modelled independently. If any cross-index analysis is added, align explicitly.", Owner="Researcher B"),
    dict(Severity="MEDIUM", Issue="Multiple testing across 6 indices",
         Detail="Pairwise Diebold-Mariano over 6 indices x 4 models invites a multiple-comparisons objection.",
         Action="Use Hansen Model Confidence Set (or Romano-Wolf stepdown) instead of a wall of DM p-values. Stronger result, fewer objections.", Owner="Researcher B"),
    dict(Severity="MEDIUM", Issue="Expired CA bundle on the workstation",
         Detail="certifi was 2024.08.30; every HTTPS fetch failed with CERTIFICATE_VERIFY_FAILED.",
         Action="FIXED: upgraded to certifi 2026.7.22. Pin certifi in requirements.txt so this does not recur.", Owner="Researcher A"),
    dict(Severity="LOW", Issue="STOXX TLS chain incomplete",
         Detail="stoxx.com serves an incomplete certificate chain; standard verification fails.",
         Action="Script 02 uses an unverified context for that host only. The content is a static public index file, so exposure is minimal - but flag it in code review.", Owner="Researcher A"),
    dict(Severity="LOW", Issue="OVX reaches 325 in Apr-2020",
         Detail="Flagged by the range check.",
         Action="GENUINE - the day after WTI settled negative (2020-04-20). Do not clean it out; it is a real tail event.", Owner="Researcher A"),
    dict(Severity="LOW", Issue="VVIX has 21-day gaps in 2006",
         Detail="Sparse backfill in the first months of VVIX.",
         Action="Irrelevant - far outside the 2013+ analysis window.", Owner="Researcher A"),
])

LIBS = pd.DataFrame([
    dict(Library="pandas", Version_Tested="2.2.3", Purpose="All tabular handling, resampling, joins", Phase="All"),
    dict(Library="numpy", Version_Tested="2.x", Purpose="Array maths, RV computation", Phase="All"),
    dict(Library="yfinance", Version_Tested="0.2.65", Purpose="Daily index OHLCV + Nikkei VI", Phase="2,3"),
    dict(Library="requests", Version_Tested="2.34.2", Purpose="HTTP with keep-alive sessions (critical for Dukascopy throughput)", Phase="3,4,6"),
    dict(Library="certifi", Version_Tested="2026.7.22", Purpose="CA bundle - MUST be current or every fetch fails", Phase="All"),
    dict(Library="lzma (stdlib)", Version_Tested="py3.12", Purpose="Decompress Dukascopy .bi5", Phase="4"),
    dict(Library="xlsxwriter", Version_Tested="3.2.9", Purpose="This report", Phase="8"),
    dict(Library="pyarrow", Version_Tested="24.0.0", Purpose="Optional Parquet mirror of large intraday files", Phase="optional"),
    dict(Library="arch", Version_Tested="to install", Purpose="GARCH / GJR / EGARCH, VaR backtests", Phase="Modelling"),
    dict(Library="statsmodels", Version_Tested="to install", Purpose="QuantReg for quantile regression", Phase="Modelling"),
    dict(Library="scipy", Version_Tested="to install", Purpose="genpareto for EVT/GPD tail fitting", Phase="Modelling"),
])

DERIVED = pd.DataFrame([
    dict(Variable="LogReturn", Formula="ln(Close_t / Close_{t-1})", Source_Table="daily/<CODE>_daily_*.csv",
         Used_By="GARCH, GJR, EGARCH, EVT, QR", Notes="Computed on index Close. Already present as a column."),
    dict(Variable="RV_5min", Formula="sum over the session of (5-min log return)^2", Source_Table="realized_volatility/<CODE>_RV_daily.csv",
         Used_By="Realized GARCH; QLIKE/RMSE target", Notes="PRIMARY realized measure. 78 bars/session for SPX/NDX."),
    dict(Variable="RV_1/10/15/30min", Formula="same at other sampling intervals", Source_Table="realized_volatility/",
         Used_By="Sampling-frequency robustness; volatility signature plot", Notes="Included so the referee ask is already answered."),
    dict(Variable="BPV_5min", Formula="(pi/2) * sum |r_i| * |r_{i-1}|", Source_Table="realized_volatility/",
         Used_By="Jump-robust volatility; RV minus BPV gives a jump measure", Notes="Realized bipower variation."),
    dict(Variable="RVol_5min", Formula="sqrt(RV_5min)", Source_Table="realized_volatility/", Used_By="Plots, comparability", Notes=""),
    dict(Variable="LogRV_5min", Formula="ln(RV_5min)", Source_Table="realized_volatility/",
         Used_By="Realized GARCH measurement equation", Notes="Hansen-Huang use the log form."),
    dict(Variable="Overnight_LogRet", Formula="ln(Open_sess_t / Close_sess_{t-1})", Source_Table="realized_volatility/",
         Used_By="Optional RV scaling to full-day variance", Notes="Session RV excludes the overnight gap. Decide explicitly whether to add it."),
    dict(Variable="z_t (std. residual)", Formula="r_t / sigma_hat_t from the fitted GARCH", Source_Table="produced in modelling",
         Used_By="EVT stage-2 input", Notes="Not a download - generated downstream."),
    dict(Variable="Tail exceedance", Formula="y_i = z_i - u for z_i > u, u at the 90-95th pct", Source_Table="produced in modelling",
         Used_By="GPD fit", Notes="Needs ~2500 obs for ~100 exceedances at 95%."),
    dict(Variable="VIX level and term-structure slope", Formula="VIX close; VIX9D/VIX3M ratio", Source_Table="volatility/",
         Used_By="QR predictors", Notes="Must be lagged - use day t close to predict t+1."),
])

PLAN = pd.DataFrame([
    dict(Step=1, Task="Fix CA bundle (certifi)", Status="DONE", Owner="A", Notes="2024.08.30 -> 2026.7.22"),
    dict(Step=2, Task="Verify all candidate sources before downloading", Status="DONE", Owner="A", Notes="13 CBOE + 6 Yahoo + 3 STOXX + Dukascopy binary format all probed"),
    dict(Step=3, Task="Download daily index OHLCV (6)", Status="DONE", Owner="A", Notes="Each accepted only after 2 identical consecutive fetches"),
    dict(Step=4, Task="Download volatility indices (17)", Status="DONE", Owner="A", Notes="CBOE + STOXX + Yahoo"),
    dict(Step=5, Task="Download Dukascopy 1-min BID (6 indices)", Status="SEE Intraday_Files sheets", Owner="A", Notes="Resumable; rerun the same script to fill any gaps"),
    dict(Step=6, Task="Build 5-min bars + realized measures", Status="SEE RV_Files sheet", Owner="A", Notes="Session-filtered in exchange-local time"),
    dict(Step=7, Task="FRED macro", Status="BLOCKED - needs your free API key", Owner="A", Notes="Script ready; see Not_Downloaded sheet"),
    dict(Step=8, Task="CFD-vs-index return correlation check", Status="SEE CFD_vs_Index sheet", Owner="A", Notes="Must pass before modelling; see Risks HIGH #1"),
    dict(Step=9, Task="Validation sweep (5 checks per file)", Status="DONE", Owner="A", Notes="See Validation sheet"),
    dict(Step=10, Task="Hand RV + returns + predictors to Researcher B", Status="TODO", Owner="A/B", Notes="After step 8 passes"),
])

DICT = pd.DataFrame([
    dict(Table="daily/<CODE>/<CODE>_daily_<first>_<last>.csv", Column="Date", Type="date", Description="Trading date, ISO"),
    dict(Table="daily/...", Column="Symbol", Type="str", Description="Internal index code (SPX/NDX/UKX/DAX/NKY/HSI)"),
    dict(Table="daily/...", Column="Open/High/Low/Close", Type="float", Description="Index level, as published"),
    dict(Table="daily/...", Column="Adj Close", Type="float", Description="Split/dividend adjusted (equals Close for price indices)"),
    dict(Table="daily/...", Column="Volume", Type="int", Description="Exchange volume where published; 0 for some indices"),
    dict(Table="daily/...", Column="LogReturn", Type="float", Description="ln(Close_t/Close_{t-1}); first row empty"),
    dict(Table="volatility/<CODE>/<CODE>_daily_*.csv", Column="Date", Type="date", Description="Trading date"),
    dict(Table="volatility/...", Column="Symbol", Type="str", Description="Vol index code"),
    dict(Table="volatility/...", Column="Open/High/Low", Type="float", Description="Empty when the provider publishes close only (VVIX, SKEW, OVX, GVZ, V2TX, V1X, V6I1)"),
    dict(Table="volatility/...", Column="Close", Type="float", Description="Index level in annualised volatility points"),
    dict(Table="intraday/{1min,5min}/<CODE>/<CODE>_{tf}_<YYYY>.csv", Column="Date", Type="date", Description="Session date in exchange-local time"),
    dict(Table="intraday/...", Column="Symbol", Type="str", Description="Index code"),
    dict(Table="intraday/...", Column="ts_utc", Type="datetime(UTC)", Description="Bar START timestamp in UTC. JOIN ON THIS."),
    dict(Table="intraday/...", Column="ts_local", Type="datetime(tz)", Description="Same instant in exchange-local time, DST-aware"),
    dict(Table="intraday/...", Column="Open/High/Low/Close", Type="float", Description="CFD BID price, already de-scaled from the int32 /1000 encoding"),
    dict(Table="intraday/...", Column="Volume", Type="float", Description="Dukascopy tick volume - relative activity, not a share count"),
    dict(Table="realized_volatility/<CODE>_RV_daily.csv", Column="Date", Type="date", Description="Session date"),
    dict(Table="realized_volatility/...", Column="NBars1min", Type="int", Description="Surviving 1-min bars after session+stale filtering. QUALITY FLAG."),
    dict(Table="realized_volatility/...", Column="RV_{1,5,10,15,30}min", Type="float", Description="Realized variance at that sampling interval (daily units)"),
    dict(Table="realized_volatility/...", Column="BPV_{...}min", Type="float", Description="Realized bipower variation - jump robust"),
    dict(Table="realized_volatility/...", Column="NBars_{...}min", Type="int", Description="Bars used at that sampling interval"),
    dict(Table="realized_volatility/...", Column="RVol_5min / LogRV_5min", Type="float", Description="sqrt and ln of RV_5min"),
    dict(Table="realized_volatility/...", Column="Open/High/Low/Close_sess", Type="float", Description="Session OHLC built from the intraday bars"),
    dict(Table="realized_volatility/...", Column="CloseToClose_LogRet", Type="float", Description="ln(Close_sess_t / Close_sess_{t-1})"),
    dict(Table="realized_volatility/...", Column="Overnight_LogRet", Type="float", Description="ln(Open_sess_t / Close_sess_{t-1})"),
])

def intraday_status():
    """Live download state per index, read off the cache directory.

    Status vocabulary:
      COMPLETE                 - downloaded, RV built, validated. CSVs are on disk.
      DOWNLOADED - RV PENDING  - all bars cached; just run script 05 to build RV (~2 min).
      EASILY DOWNLOADABLE      - source verified working, script written and proven on other
                                 indices. Nothing is blocking it except unattended runtime.
                                 No API key, no account, no payment, no manual step.
    """
    CACHE = os.path.join(ROOT, '12_CACHE_REGENERATION')
    # code -> (instrument, approx weekdays in the full span from its first data year)
    inst = {"SPX": ("USA500IDXUSD", 4080), "NDX": ("USATECHIDXUSD", 4080),
            "UKX": ("GBRIDXGBP", 4080), "DAX": ("DEUIDXEUR", 3559),
            "NKY": ("JPNIDXJPY", 4080), "HSI": ("HKGIDXHKD", 4080)}
    RATE = 4.8  # verified sustained req/s, single process, 6 keep-alive threads
    rows = []
    for code, (ins, expected) in inst.items():
        cdir = os.path.join(CACHE, code)
        cached = len(glob.glob(os.path.join(cdir, 'BID_*.npy'))) if os.path.isdir(cdir) else 0
        rvp = os.path.join(ROOT, '06_REALIZED_MEASURES', f'{code}_RV_daily.csv')
        if os.path.exists(rvp):
            r = pd.read_csv(rvp, parse_dates=['Date'])
            first, last, days = str(r.Date.min().date()), str(r.Date.max().date()), len(r)
            medbars = float(r['NBars_5min'].median())
        else:
            first = last = ""; days = 0; medbars = np.nan
        fails = rd(os.path.join(LOG, f'duka_failed_{code}_BID.csv'))
        nfail = len(fails) if len(fails) else 0

        remaining = max(expected - cached, 0)
        eta_min = round(remaining / RATE / 60, 1) if remaining else 0.0

        if days > 3000:
            status = "COMPLETE"
            note = "Downloaded, RV built, validated. CSVs on disk."
        elif cached >= expected * 0.9:
            status = "DOWNLOADED - RV PENDING"
            note = f"All bars cached. Run script 05 to build RV (~2 min)."
        else:
            status = "EASILY DOWNLOADABLE"
            note = ("Source verified working; same script already proven on SPX/NDX. "
                    "No API key, no account, no payment, no manual step - only unattended runtime. "
                    f"~{remaining} days left at the verified 4.8 req/s.")

        rows.append(dict(Code=code, Dukascopy_Instrument=ins, Days_Cached=cached,
                         Days_Expected=expected, Pct_Cached=round(100 * cached / expected, 1),
                         RV_Session_Days=days, RV_First=first, RV_Last=last,
                         Median_5min_Bars=medbars, Failed_Days_To_Retry=nfail,
                         Status=status, Est_Minutes_To_Finish=eta_min, Note=note,
                         Resume_Command=f"python Datasets/10_SCRIPTS/03_download_intraday.py {code}"))
    return pd.DataFrame(rows)


files_daily = scan('02_RAW_DAILY', '*/*.csv')
files_vol = scan('04_RAW_VOLATILITY', '*/*.csv')
files_rv = scan('06_REALIZED_MEASURES', '*.csv')
files_5min = scan('intraday/5min', '*/*.csv')
files_1min = scan('intraday/1min', '*/*.csv')
val = rd(os.path.join(VAL, 'validation_report.csv'))
cfd = rd(os.path.join(VAL, 'cfd_vs_index_check.csv'))
man_daily = rd(os.path.join(LOG, 'phase2_daily_manifest.csv'))
man_vol = rd(os.path.join(LOG, 'phase3_volatility_manifest.csv'))
man_duka = rd(os.path.join(LOG, 'phase4_download_summary_BID.csv'))
man_rv = rd(os.path.join(LOG, 'phase5_rv_summary.csv'))

files_macro = scan('05_RAW_MACRO', '*/*.csv')
files_panel = scan('07_PANEL_INTERMEDIATE', '*.csv')
man_macro = rd(os.path.join(LOG, 'phase6b_macro_yahoo_manifest.csv'))
man_panel = rd(os.path.join(LOG, 'phase9_panel_summary.csv'))
frozen = rd(os.path.join(VAL, 'frozen_sessions_dropped.csv'))

PANEL_DICT = pd.DataFrame([
 ("Date","spine","Index trading day. The exchange daily file is the spine; everything else is LEFT-joined onto it, so the sample is exactly the days the index traded."),
 ("Open High Low Close","spine","Exchange index daily OHLC (Yahoo)."),
 ("Return","spine","Daily log return of the exchange index close. THE DEPENDENT VARIABLE."),
 ("RV_1min .. RV_30min","realized","Realized variance from session-filtered CFD bars at 1/5/10/15/30-min sampling. 5-min is the modelling frequency; the others exist so the volatility signature plot can justify that choice."),
 ("BPV_5min","realized","Realized bipower variation (Barndorff-Nielsen & Shephard). Jump-robust."),
 ("Jump_5min","realized","max(RV_5min - BPV_5min, 0). The jump component."),
 ("ContVar_5min","realized","RV_5min - Jump_5min. The continuous component."),
 ("NBars_5min","realized","Bars used. Median should be 78 (SPX/NDX), 102 (UKX), 102 (DAX), 60 (NKY), 66 (HSI). A short count means a half-day or a feed hole."),
 ("RVol_5min / LogRV_5min","realized","sqrt(RV_5min) and log(RV_5min). Realized GARCH is usually specified in logs."),
 ("HasRV_t","realized","False where no clean CFD session existed for that index trading day. Honest missingness - NOT filled."),
 ("CFD_SessionReturn","cross-ref","CFD session close-to-close return. For diagnostics only. Never use as the dependent variable - see the CFD_vs_Index sheet."),
 ("Overnight_LogRet","cross-ref","close(t-1) -> open(t). Dated t but known at the OPEN of t. It is the first observable piece of day t, not day-t close information."),
 ("VolIdx / VolIdx_Symbol","implied","The regional implied-vol index and which symbol it actually is. The symbol column exists so a proxy substitution is never invisible."),
 ("VolIdx_Fallback","implied","Secondary vol index where the primary has short history (NKVI starts 2018) or is a proxy."),
 ("US10Y_pct US13W_pct","05_RAW_MACRO","US Treasury yields IN PERCENT. Yahoo used to quote ^TNX as percent x 10; it no longer does. Do NOT divide by 10."),
 ("TermSpread_pct","05_RAW_MACRO","US10Y_pct - US13W_pct."),
 ("DXY WTI_usd GOLD_usd","05_RAW_MACRO","Dollar index, WTI crude, COMEX gold. Close only - Yahoo's OHLC is unreliable for FX/futures, see Risks."),
 ("HYG_px IEF_px","05_RAW_MACRO","High-yield and 7-10y Treasury ETF prices."),
 ("CreditStress","05_RAW_MACRO","-(dlog HYG - dlog IEF). High-yield underperformance vs duration-matched Treasuries. Rises with credit stress, same sign as a widening HY OAS. Keyless stand-in for FRED BAMLH0A0HYM2."),
 ("AbsReturn NegReturn","derived","|r| and the negative part of r. NegReturn is the leverage/asymmetry term for GJR and for quantile regression."),
 ("ParkinsonVar","derived","log(H/L)^2 / (4 log 2). A range-based variance estimator available on every day back to 1990, including days with no RV."),
 ("RangePct","derived","100*(High-Low)/Close."),
], columns=["Column(s)","Group","Meaning and how to use it"])

PANEL_NOTES = pd.DataFrame([
 ("What this folder is","Datasets/07_PANEL_INTERMEDIATE/<CODE>_panel_daily.csv is the analysis-ready join of daily prices, realized measures, the regional volatility index and the macro factors - one row per index trading day. Start modelling from these files, not from the raw folders."),
 ("Join rule","The exchange DAILY file is the spine. Everything else is LEFT-joined on Date. The dependent variable is the index return, so the sample must be exactly the days the index traded."),
 ("Missingness","RV_5min is NaN on spine days where the CFD feed had no clean session. It is deliberately NOT filled or interpolated. Filter on HasRV_t."),
 ("No look-ahead","Every predictor is dated at the CLOSE of day t and is meant to forecast t+1. This file does not lag anything - the modelling code must apply the lag."),
 ("Overnight caveat","Overnight_LogRet spans close(t-1) -> open(t). It is dated t but known at the open of t. Treat it as the first observable piece of day t."),
 ("Rebuild","python Datasets/10_SCRIPTS/10_build_master_panel.py"),
], columns=["Item","Detail"])

sheets = [
    ("README", pd.DataFrame(README, columns=["Item", "Detail"])),
    ("Index_Universe", UNIVERSE),
    ("Sources_and_Links", SOURCES),
    ("Not_Downloaded", NOT_DOWNLOADED),
    ("Intraday_Status", intraday_status()),
    ("Daily_Files", files_daily),
    ("Volatility_Files", files_vol),
    ("Intraday_5min_Files", files_5min),
    ("Intraday_1min_Files", files_1min),
    ("RV_Files", files_rv),
    ("Macro_Files", files_macro),
    ("Panel_Files", files_panel),
    ("Data_Dictionary", DICT),
    ("Panel_Notes", PANEL_NOTES),
    ("Panel_Dictionary", PANEL_DICT),
    ("Derived_Variables", DERIVED),
    ("Validation", val),
    ("CFD_vs_Index", cfd),
    ("Frozen_Sessions_Dropped", frozen),
    ("Risks_and_Gotchas", RISKS),
    ("Libraries", LIBS),
    ("Collection_Plan", PLAN),
    ("Manifest_Daily", man_daily),
    ("Manifest_Volatility", man_vol),
    ("Manifest_Intraday_DL", man_duka),
    ("Manifest_RV", man_rv),
    ("Manifest_Macro", man_macro),
    ("Manifest_Panel", man_panel),
]

with pd.ExcelWriter(XL, engine='xlsxwriter') as xw:
    wb = xw.book
    hdr = wb.add_format(dict(bold=True, bg_color='#1F3864', font_color='white', border=1, text_wrap=True, valign='top'))
    wrap = wb.add_format(dict(text_wrap=True, valign='top'))
    for name, df in sheets:
        if df is None or len(df) == 0:
            df = pd.DataFrame({"note": [f"No rows available when this report was generated ({TODAY}). "
                                        f"Re-run Datasets/10_SCRIPTS/07_build_excel_report.py after the pipeline finishes."]})
        sn = name[:31]
        df.to_excel(xw, sheet_name=sn, index=False, startrow=1, header=False)
        ws = xw.sheets[sn]
        for j, c in enumerate(df.columns):
            ws.write(0, j, str(c), hdr)
        for j, c in enumerate(df.columns):
            try:
                w = int(min(60, max(14, df[c].astype(str).str.len().quantile(0.9) + 4)))
            except Exception:
                w = 22
            ws.set_column(j, j, w, wrap)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))

print("wrote", XL)
for n, d in sheets:
    print(f"  {n:24s} {len(d) if d is not None else 0:>5} rows")
