"""
Phase 4: Dukascopy 1-minute candles, 6 Tier-1 index CFDs.
Binary (verified): >Iiiiif = t_offset_sec(u32), O,C,L,H (i32 /1000), volume(f32)
URL months are ZERO-INDEXED.
Each worker thread holds its OWN keep-alive requests.Session (critical: ~0.25s/req vs ~20s without).
Resumable: cached .npy per day is skipped on re-run.
"""
import os, sys, time, threading, queue, lzma
from datetime import date, timedelta
import numpy as np, pandas as pd, requests

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT,'12_CACHE_REGENERATION'); LOG = os.path.join(ROOT,'11_LOGS')
os.makedirs(CACHE,exist_ok=True); os.makedirs(LOG,exist_ok=True)

INSTR = {
 "SPX":"USA500IDXUSD", "NDX":"USATECHIDXUSD", "UKX":"GBRIDXGBP",
 "DAX":"DEUIDXEUR",    "NKY":"JPNIDXJPY",     "HSI":"HKGIDXHKD",
}
SIDE     = os.environ.get("DUKA_SIDE","BID")
NTHREAD  = int(os.environ.get("DUKA_THREADS","6"))
END      = date(2026,8,21)
DTYPE    = np.dtype([('t','>u4'),('o','>i4'),('c','>i4'),('l','>i4'),('h','>i4'),('v','>f4')])

def mksession():
    s = requests.Session()
    s.headers.update({'User-Agent':'Mozilla/5.0','Connection':'keep-alive'})
    s.mount('https://', requests.adapters.HTTPAdapter(pool_connections=2,pool_maxsize=2,max_retries=0))
    return s

def url(ins,d):
    return f"https://datafeed.dukascopy.com/datafeed/{ins}/{d.year:04d}/{d.month-1:02d}/{d.day:02d}/{SIDE}_candles_min_1.bi5"

def get(sess, ins, d, retries=4):
    for a in range(retries):
        try:
            r = sess.get(url(ins,d), timeout=30)
            if r.status_code == 404: return b""
            if r.status_code == 200: return r.content
            time.sleep(1.0*(a+1))
        except Exception:
            time.sleep(1.5*(a+1))
            try: sess.close()
            except Exception: pass
    return None

def parse(b):
    if not b: return None
    try: dec = lzma.LZMADecompressor().decompress(b)
    except Exception: return None
    n = len(dec)//24
    if n == 0: return None
    return np.frombuffer(dec[:n*24], dtype=DTYPE)

def find_start(ins, lo=2011, hi=2026):
    """Coarse probe: earliest year containing data. Samples 4 dates/year."""
    s = mksession()
    for y in range(lo, hi+1):
        for (m,dd) in [(3,12),(6,11),(9,10),(11,12)]:
            b = get(s, ins, date(y,m,dd), retries=2)
            if b: 
                r = parse(b)
                if r is not None and int(np.abs(r['o']).sum())>0:
                    s.close(); return y
    s.close(); return None

def worker(code, ins, q, counters, lock, failed):
    sess = mksession()
    while True:
        try: d = q.get_nowait()
        except queue.Empty: break
        cp = os.path.join(CACHE, code, f"{SIDE}_{d.isoformat()}.npy")
        if os.path.exists(cp):
            with lock: counters['cached'] += 1
            continue
        b = get(sess, ins, d)
        if b is None:
            with lock: counters['fail'] += 1; failed.append(d.isoformat())
            continue
        rec = parse(b)
        if rec is None:
            np.save(cp, np.zeros(0, dtype=DTYPE))
            with lock: counters['empty'] += 1
        else:
            np.save(cp, rec)
            with lock: counters['ok'] += 1
    sess.close()

def run(code):
    ins = INSTR[code]
    os.makedirs(os.path.join(CACHE,code), exist_ok=True)
    y0 = find_start(ins)
    if y0 is None:
        print(f"  [{code}] NO DATA FOUND", flush=True); return None
    print(f"  [{code}] first year with data = {y0}", flush=True)
    q = queue.Queue()
    d = date(y0,1,1)
    while d <= END:
        if d.weekday() < 5: q.put(d)
        d += timedelta(days=1)
    total = q.qsize()
    counters = dict(ok=0,empty=0,fail=0,cached=0); failed=[]; lock=threading.Lock()
    ts=[threading.Thread(target=worker,args=(code,ins,q,counters,lock,failed),daemon=True) for _ in range(NTHREAD)]
    t0=time.time(); [t.start() for t in ts]
    while any(t.is_alive() for t in ts):
        time.sleep(20)
        done=sum(counters.values()); el=time.time()-t0
        rate=done/el if el>0 else 0
        eta=(total-done)/rate/60 if rate>0 else -1
        print(f"  [{code}] {done}/{total} ({100*done/total:.1f}%) ok={counters['ok']} empty={counters['empty']} "
              f"fail={counters['fail']} cache={counters['cached']} {rate:.1f}req/s eta={eta:.0f}min", flush=True)
    [t.join() for t in ts]
    print(f"  [{code}] DONE {counters} in {(time.time()-t0)/60:.1f}min", flush=True)
    pd.DataFrame({'failed_day':failed}).to_csv(os.path.join(LOG,f'duka_failed_{code}_{SIDE}.csv'), index=False)
    return dict(code=code, start_year=y0, total_days=total, **counters)

if __name__ == "__main__":
    codes = sys.argv[1:] if len(sys.argv)>1 else list(INSTR)
    out=[]
    for c in codes:
        print(f"=== {c} ({INSTR[c]}) side={SIDE} threads={NTHREAD} ===", flush=True)
        r = run(c)
        if r: out.append(r); pd.DataFrame(out).to_csv(os.path.join(LOG,f'phase4_download_summary_{SIDE}.csv'), index=False)
    print("\nALL DONE", flush=True)
