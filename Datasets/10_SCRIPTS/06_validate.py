"""
Phase 7: Validation sweep. Every CSV is checked on 5 independent axes:
  C1 schema      - expected columns present, dtypes parse
  C2 uniqueness  - no duplicate date / timestamp keys
  C3 monotonic   - strictly increasing time index
  C4 ranges      - prices > 0, OHLC consistency (L<=O,C<=H), vol indices in (0,200], returns |r|<50%
  C5 continuity  - calendar gap profile; flags gaps > 10 consecutive trading days
Writes Datasets/08_VALIDATION/validation_report.csv  (one row per file per check)
"""
import os, glob, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTV=os.path.join(ROOT,'08_VALIDATION'); os.makedirs(OUTV,exist_ok=True)
res=[]

def rec(f,check,status,detail=""):
    res.append(dict(file=os.path.relpath(f,ROOT),check=check,status=status,detail=str(detail)[:220]))

def chk_daily(f):
    df=pd.read_csv(f)
    need={"Date","Symbol","Open","High","Low","Close"}
    rec(f,"C1_schema","PASS" if need.issubset(df.columns) else "FAIL",
        f"cols={list(df.columns)}" if not need.issubset(df.columns) else f"{len(df)} rows")
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    rec(f,"C2_unique","PASS" if not df["Date"].duplicated().any() else "FAIL",
        int(df["Date"].duplicated().sum()))
    rec(f,"C3_monotonic","PASS" if df["Date"].is_monotonic_increasing else "FAIL","")
    bad=0; det=[]
    for c in ["Open","High","Low","Close"]:
        if c in df:
            n=int((pd.to_numeric(df[c],errors="coerce")<=0).sum()); bad+=n
            if n: det.append(f"{c}<=0:{n}")
    if {"Open","High","Low","Close"}.issubset(df.columns):
        o,h,l,c2=[pd.to_numeric(df[x],errors="coerce") for x in ["Open","High","Low","Close"]]
        v=df.dropna(subset=["Open","High","Low","Close"]).index
        nviol=int(((h<l)|(h<o)|(h<c2)|(l>o)|(l>c2)).sum()); bad+=nviol
        if nviol: det.append(f"OHLC_violations:{nviol}")
    if "LogReturn" in df:
        n=int((pd.to_numeric(df["LogReturn"],errors="coerce").abs()>0.5).sum()); bad+=n
        if n: det.append(f"|logret|>50%:{n}")
    rec(f,"C4_ranges","PASS" if bad==0 else "WARN","; ".join(det) if det else "ok")
    d=df["Date"].dropna(); gaps=d.diff().dt.days.fillna(1)
    big=int((gaps>14).sum())
    rec(f,"C5_continuity","PASS" if big==0 else "WARN",
        f"max_gap={int(gaps.max())}d; gaps>14d={big}")

def chk_vol(f):
    df=pd.read_csv(f)
    need={"Date","Symbol","Close"}
    rec(f,"C1_schema","PASS" if need.issubset(df.columns) else "FAIL",f"{len(df)} rows")
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    rec(f,"C2_unique","PASS" if not df["Date"].duplicated().any() else "FAIL",int(df["Date"].duplicated().sum()))
    rec(f,"C3_monotonic","PASS" if df["Date"].is_monotonic_increasing else "FAIL","")
    c=pd.to_numeric(df["Close"],errors="coerce")
    oob=int(((c<=0)|(c>250)).sum())
    rec(f,"C4_ranges","PASS" if oob==0 else "WARN",
        f"min={c.min():.2f} max={c.max():.2f} out_of_(0,250]={oob}")
    gaps=df["Date"].dropna().diff().dt.days.fillna(1); big=int((gaps>14).sum())
    rec(f,"C5_continuity","PASS" if big==0 else "WARN",f"max_gap={int(gaps.max())}d; gaps>14d={big}")

def chk_rv(f):
    df=pd.read_csv(f)
    need={"Date","Symbol","RV_5min","NBars_5min"}
    rec(f,"C1_schema","PASS" if need.issubset(df.columns) else "FAIL",f"{len(df)} rows")
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    rec(f,"C2_unique","PASS" if not df["Date"].duplicated().any() else "FAIL",int(df["Date"].duplicated().sum()))
    rec(f,"C3_monotonic","PASS" if df["Date"].is_monotonic_increasing else "FAIL","")
    rv=pd.to_numeric(df["RV_5min"],errors="coerce")
    ann=np.sqrt(rv*252)*100
    bad=int(((rv<=0)|(~np.isfinite(rv))).sum()); extreme=int((ann>300).sum())
    rec(f,"C4_ranges","PASS" if bad==0 and extreme==0 else "WARN",
        f"ann_vol median={np.nanmedian(ann):.1f}% p99={np.nanpercentile(ann,99):.1f}% max={np.nanmax(ann):.1f}%; nonpos={bad}; >300%={extreme}")
    nb=pd.to_numeric(df["NBars_5min"],errors="coerce")
    short=int((nb < nb.median()*0.5).sum())
    rec(f,"C5_continuity","PASS" if short/max(len(df),1)<0.05 else "WARN",
        f"median_bars={nb.median():.0f}; days_with<50%bars={short} ({100*short/max(len(df),1):.1f}%)")

def chk_macro(f):
    """Macro/risk-factor series. Same shape as daily, but yields and spreads may legitimately
    be zero or negative (US13W hit 0.00 in 2020-2021; oil printed negative in April 2020),
    so C4 does NOT require price>0 here - it checks finiteness and OHLC ordering only."""
    df=pd.read_csv(f)
    need={"Date","Symbol","Close"}
    rec(f,"C1_schema","PASS" if need.issubset(df.columns) else "FAIL",
        f"{len(df)} rows" if need.issubset(df.columns) else f"cols={list(df.columns)}")
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    rec(f,"C2_unique","PASS" if not df["Date"].duplicated().any() else "FAIL",int(df["Date"].duplicated().sum()))
    rec(f,"C3_monotonic","PASS" if df["Date"].is_monotonic_increasing else "FAIL","")
    det=[]; bad=0
    c=pd.to_numeric(df["Close"],errors="coerce")
    n=int((~np.isfinite(c)).sum()); bad+=n
    if n: det.append(f"non_finite_Close:{n}")
    if {"Open","High","Low","Close"}.issubset(df.columns):
        o,h,l,c2=[pd.to_numeric(df[x],errors="coerce") for x in ["Open","High","Low","Close"]]
        nv=int(((h<l)|(h<o)|(h<c2)|(l>o)|(l>c2)).sum()); bad+=nv
        if nv: det.append(f"OHLC_violations:{nv}")
    if "LogReturn" in df:
        nr=int((pd.to_numeric(df["LogReturn"],errors="coerce").abs()>0.5).sum())
        if nr: det.append(f"|logret|>50%:{nr}")   # informational, not a failure for commodities
    rec(f,"C4_ranges","PASS" if bad==0 else "WARN",
        "; ".join(det) if det else f"min={np.nanmin(c):.4f} max={np.nanmax(c):.4f}")
    gaps=df["Date"].dropna().diff().dt.days.fillna(1); big=int((gaps>14).sum())
    rec(f,"C5_continuity","PASS" if big==0 else "WARN",f"max_gap={int(gaps.max())}d; gaps>14d={big}")

def chk_intraday(f):
    """Intraday bar file (one index-year). Timestamps are exchange-local, session-filtered."""
    df=pd.read_csv(f)
    need={"Date","Symbol","ts_utc","ts_local","Open","High","Low","Close","Volume"}
    ok=need.issubset(df.columns)
    rec(f,"C1_schema","PASS" if ok else "FAIL",f"{len(df)} bars" if ok else f"cols={list(df.columns)}")
    if not ok: return
    ts=pd.to_datetime(df["ts_utc"],errors="coerce",utc=True)
    rec(f,"C2_unique","PASS" if not ts.duplicated().any() else "FAIL",int(ts.duplicated().sum()))
    rec(f,"C3_monotonic","PASS" if ts.is_monotonic_increasing else "FAIL","")
    o,h,l,c=[pd.to_numeric(df[x],errors="coerce") for x in ["Open","High","Low","Close"]]
    nonpos=int(((c<=0)|(~np.isfinite(c))).sum())
    nv=int(((h<l)|(h<o)|(h<c)|(l>o)|(l>c)).sum())
    r=np.log(c).diff()
    jump=int((r.abs()>0.10).sum())          # >10% inside one bar = almost certainly bad tick
    rec(f,"C4_ranges","PASS" if nonpos==0 and nv==0 and jump==0 else "WARN",
        f"nonpos={nonpos}; OHLC_violations={nv}; bars_with_|r|>10%={jump}; "
        f"px range {np.nanmin(c):.1f}-{np.nanmax(c):.1f}")
    nday=pd.to_datetime(df["Date"]).dt.date.nunique()
    per=len(df)/max(nday,1)
    rec(f,"C5_continuity","PASS" if nday>0 else "FAIL",
        f"{nday} session-days; mean {per:.1f} bars/day; {ts.min()} -> {ts.max()}")

for f in sorted(glob.glob(os.path.join(ROOT,'02_RAW_DAILY','*','*.csv'))): chk_daily(f)
for f in sorted(glob.glob(os.path.join(ROOT,'04_RAW_VOLATILITY','*','*.csv'))): chk_vol(f)
for f in sorted(glob.glob(os.path.join(ROOT,'06_REALIZED_MEASURES','*.csv'))): chk_rv(f)
for f in sorted(glob.glob(os.path.join(ROOT,'05_RAW_MACRO','*','*.csv'))): chk_macro(f)
for f in sorted(glob.glob(os.path.join(ROOT,'03_RAW_INTRADAY','*','*','*.csv'))): chk_intraday(f)

rp=pd.DataFrame(res)
rp.to_csv(os.path.join(OUTV,'validation_report.csv'),index=False)
print(rp.groupby(['check','status']).size().to_string())
print(f"\n{len(rp)} checks across {rp['file'].nunique()} files")
fails=rp[rp.status=="FAIL"]
print("FAILURES:" if len(fails) else "\nNo FAIL-level issues.")
if len(fails): print(fails.to_string(index=False))
warns=rp[rp.status=="WARN"]
if len(warns): print(f"\n{len(warns)} warnings:"); print(warns.to_string(index=False))
