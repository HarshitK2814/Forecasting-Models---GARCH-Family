"""
Phase 4b/5: cached Dukascopy 1-min .npy  ->  session-filtered 1-min & 5-min CSVs  ->  daily realized measures.

Session filtering is done in EXCHANGE-LOCAL time (handles DST automatically), because
Dukascopy pads every minute of the 24h UTC day with stale zero-volume bars outside the
cash session. Including those bars injects zero returns and biases RV downward.

Outputs
  intraday/1min/<CODE>/<CODE>_1min_<YYYY>.csv
  intraday/5min/<CODE>/<CODE>_5min_<YYYY>.csv
  realized_volatility/<CODE>_RV_daily.csv
"""
import os, sys, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from datetime import datetime, timezone

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE=os.path.join(ROOT,'12_CACHE_REGENERATION'); LOG=os.path.join(ROOT,'11_LOGS')
OUT1=os.path.join(ROOT,'03_RAW_INTRADAY','1min'); OUT5=os.path.join(ROOT,'03_RAW_INTRADAY','5min')
OUTRV=os.path.join(ROOT,'06_REALIZED_MEASURES')
for p in (OUT1,OUT5,OUTRV,LOG): os.makedirs(p,exist_ok=True)

SESSIONS={  # code -> (tz, [(start,end) local]) ; cash session only
 "SPX":("America/New_York",[("09:30","16:00")]),
 "NDX":("America/New_York",[("09:30","16:00")]),
 "UKX":("Europe/London",   [("08:00","16:30")]),
 "DAX":("Europe/Berlin",   [("09:00","17:30")]),
 "NKY":("Asia/Tokyo",      [("09:00","11:30"),("12:30","15:00")]),
 "HSI":("Asia/Hong_Kong",  [("09:30","12:00"),("13:00","16:00")]),
}
SCALE=1000.0
SAMPLES=[1,5,10,15,30]

def load_day(code, path):
    rec=np.load(path, allow_pickle=False)
    if rec.size==0: return None
    day=os.path.basename(path).split('_')[1].replace('.npy','')
    base=pd.Timestamp(day, tz='UTC')
    df=pd.DataFrame({
        'ts_utc': base + pd.to_timedelta(rec['t'].astype('int64'), unit='s'),
        'Open':  rec['o'].astype('float64')/SCALE,
        'Close': rec['c'].astype('float64')/SCALE,
        'Low':   rec['l'].astype('float64')/SCALE,
        'High':  rec['h'].astype('float64')/SCALE,
        'Volume':rec['v'].astype('float64'),
    })
    return df[df['Close']>0]

def session_mask(local_idx, windows):
    t=local_idx.time
    m=np.zeros(len(local_idx),dtype=bool)
    for a,b in windows:
        ha,ma=map(int,a.split(':')); hb,mb=map(int,b.split(':'))
        import datetime as _dt
        m |= (t>=_dt.time(ha,ma)) & (t< _dt.time(hb,mb))
    return m

def realized_measures(px_series, m):
    """px_series: 1-min close indexed by local ts within one session-day."""
    s=px_series.resample(f'{m}min').last().dropna()
    if len(s)<3: return np.nan, np.nan, len(s)
    r=np.diff(np.log(s.values))
    rv=float(np.sum(r**2))
    bpv=float((np.pi/2)*np.sum(np.abs(r[:-1])*np.abs(r[1:]))) if len(r)>1 else np.nan
    return rv, bpv, len(s)

FROZEN_RANGE_THRESHOLD = 4e-4   # 4 basis points of session range - see process() for calibration
frozen = []                     # audit log of every session this rule removes


def process(code):
    tz,windows=SESSIONS[code]
    files=sorted(glob.glob(os.path.join(CACHE,code,'BID_*.npy')))
    if not files: print(f"  [{code}] no cache"); return None
    rv_rows=[]; by_year_1={}; by_year_5={}
    kept=0; skipped=0
    for fp in files:
        d=load_day(code,fp)
        if d is None or d.empty: skipped+=1; continue
        d['ts_local']=d['ts_utc'].dt.tz_convert(tz)
        d=d[session_mask(d['ts_local'].dt, windows)] if False else d[session_mask(pd.DatetimeIndex(d['ts_local']), windows)]
        if d.empty: skipped+=1; continue
        # drop stale padding: no volume AND no intrabar range
        d=d[(d['Volume']>0) | (d['High']!=d['Low'])]
        if len(d)<10: skipped+=1; continue
        # Exchange-holiday guard. On days the cash market is shut, Dukascopy still streams a
        # nominally-live CFD that drifts by a fraction of a point. Those bars survive the
        # stale-padding filter above (High != Low by ~0.015) but they are not trading - the
        # resulting RV is 0.0 and would enter the sample as a spurious zero-volatility day.
        # It also catches outright FEED OUTAGES on days the exchange really was open.
        #
        # Everything this rule removes, verified individually 2026-08-23:
        #   UKX 2013-12-25, 2013-12-26, 2014-01-01  range 0.02 bp  LSE closed (not in the
        #       exchange daily file either, so they were never valid observations)
        #   SPX 2013-02-25..2013-02-28              range 0.09-2.11 bp  } Dukascopy feed
        #   NDX 2013-02-26..2013-02-28              range 0.41-2.90 bp  } outage, exchange OPEN
        # That Feb-2013 outage matters: those days ARE in the exchange daily file, so without
        # this rule they enter the sample as ~zero-RV days and they were SPX's two largest
        # CFD-vs-index residuals (-1.82% CFD vs +0.61% index on 2013-02-26).
        #
        # Threshold calibration (empirical, all three completed indices, 10,847 session-days):
        #   largest range among days we drop   = 2.90 bp
        #   smallest range among days we keep  = 5.68 bp  (SPX)
        # 4 bp sits between the two clusters with ~38%% margin on each side. Every dropped day
        # is written to _validation/frozen_sessions_dropped.csv so the rule stays auditable.
        rng=float((d['High'].max()-d['Low'].min())/d['Close'].iloc[-1])
        if not np.isfinite(rng) or rng < FROZEN_RANGE_THRESHOLD:
            frozen.append(dict(Symbol=code,
                               Date=str(pd.DatetimeIndex(d['ts_local']).date[0]),
                               Range_bps=round(rng*1e4,3),
                               NBars=len(d), Close=float(d['Close'].iloc[-1])))
            skipped+=1; continue
        d=d.sort_values('ts_utc').reset_index(drop=True)
        sess_date=pd.DatetimeIndex(d['ts_local']).date[0]
        d.insert(0,'Symbol',code); d.insert(0,'Date',sess_date)
        kept+=1

        # ---- 1-min output
        o1=d[['Date','Symbol','ts_utc','ts_local','Open','High','Low','Close','Volume']].copy()
        by_year_1.setdefault(sess_date.year,[]).append(o1)

        # ---- 5-min output
        idx=pd.DatetimeIndex(d['ts_local']); ser=d.set_index(idx)
        agg=ser.resample('5min').agg(Open=('Open','first'),High=('High','max'),
                                     Low=('Low','min'),Close=('Close','last'),Volume=('Volume','sum')).dropna(subset=['Close'])
        if not agg.empty:
            a=agg.reset_index().rename(columns={'index':'ts_local'})
            a.columns=['ts_local','Open','High','Low','Close','Volume']
            a.insert(0,'Symbol',code); a.insert(0,'Date',sess_date)
            a['ts_utc']=pd.DatetimeIndex(a['ts_local']).tz_convert('UTC')
            by_year_5.setdefault(sess_date.year,[]).append(
                a[['Date','Symbol','ts_utc','ts_local','Open','High','Low','Close','Volume']])

        # ---- realized measures
        px=ser['Close']
        row=dict(Date=sess_date, Symbol=code, NBars1min=len(d))
        for m in SAMPLES:
            rv,bpv,n=realized_measures(px,m)
            row[f'RV_{m}min']=rv; row[f'BPV_{m}min']=bpv; row[f'NBars_{m}min']=n
        row['RVol_5min']=np.sqrt(row['RV_5min']) if pd.notna(row['RV_5min']) else np.nan
        row['LogRV_5min']=np.log(row['RV_5min']) if pd.notna(row['RV_5min']) and row['RV_5min']>0 else np.nan
        row['Open_sess']=float(d['Open'].iloc[0]); row['Close_sess']=float(d['Close'].iloc[-1])
        row['High_sess']=float(d['High'].max());   row['Low_sess']=float(d['Low'].min())
        rv_rows.append(row)

    # write yearly CSVs
    for yr,chunks in by_year_1.items():
        dd=os.path.join(OUT1,code); os.makedirs(dd,exist_ok=True)
        pd.concat(chunks).to_csv(os.path.join(dd,f"{code}_1min_{yr}.csv"),index=False,float_format="%.4f")
    for yr,chunks in by_year_5.items():
        dd=os.path.join(OUT5,code); os.makedirs(dd,exist_ok=True)
        pd.concat(chunks).to_csv(os.path.join(dd,f"{code}_5min_{yr}.csv"),index=False,float_format="%.4f")

    rv=pd.DataFrame(rv_rows).sort_values('Date')
    # overnight + close-to-close return from session closes
    rv['CloseToClose_LogRet']=np.log(rv['Close_sess']/rv['Close_sess'].shift(1))
    rv['Overnight_LogRet']=np.log(rv['Open_sess']/rv['Close_sess'].shift(1))
    rv.to_csv(os.path.join(OUTRV,f"{code}_RV_daily.csv"),index=False,float_format="%.10f",date_format="%Y-%m-%d")
    print(f"  [{code}] session-days kept={kept} skipped={skipped} years={min(by_year_1)}-{max(by_year_1)}")
    return dict(code=code,days=kept,skipped=skipped,first=str(rv['Date'].iloc[0]),last=str(rv['Date'].iloc[-1]),
                median_bars_5min=float(rv['NBars_5min'].median()))

if __name__=="__main__":
    codes=sys.argv[1:] if len(sys.argv)>1 else list(SESSIONS)
    out=[]
    for c in codes:
        print(f"=== {c} ===",flush=True)
        r=process(c)
        if r: out.append(r)
    if out:
        pd.DataFrame(out).to_csv(os.path.join(LOG,'phase5_rv_summary.csv'),index=False)
        print(pd.DataFrame(out).to_string(index=False))
    VAL=os.path.join(ROOT,'08_VALIDATION'); os.makedirs(VAL,exist_ok=True)
    fz=pd.DataFrame(frozen)
    fz.to_csv(os.path.join(VAL,'frozen_sessions_dropped.csv'),index=False)
    print(f"\nfrozen/outage sessions dropped: {len(fz)}  -> _validation/frozen_sessions_dropped.csv")
    if len(fz): print(fz.to_string(index=False))
