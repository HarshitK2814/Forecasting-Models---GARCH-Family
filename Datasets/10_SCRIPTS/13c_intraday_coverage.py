# -*- coding: utf-8 -*-
"""
Diagnose intraday SESSION COVERAGE per index per year, straight from the Dukascopy cache.

The stage-1 audit found NKY sitting at exactly 48 bars on every day of 2016 against an
expected 60, and only 81 RV days in the whole of 2015. Either the feed is truncating the
session or our session window is wrong. RV computed on four fifths of a session is biased
downward by roughly the missing fifth, so this has to be resolved before anything is
modelled - it is not a rounding issue.

For every index-year this prints the local clock-time coverage actually present in the
cache, so a truncated feed is distinguishable from a mis-specified session window.
"""
import os
import glob
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, '12_CACHE_REGENERATION')
LOG = os.path.join(ROOT, '11_LOGS')

SESSIONS = {
    "SPX": ("America/New_York", [("09:30", "16:00")]),
    "NDX": ("America/New_York", [("09:30", "16:00")]),
    "UKX": ("Europe/London",    [("08:00", "16:30")]),
    "DAX": ("Europe/Berlin",    [("09:00", "17:30")]),
    "NKY": ("Asia/Tokyo",       [("09:00", "11:30"), ("12:30", "15:00")]),
    "HSI": ("Asia/Hong_Kong",   [("09:30", "12:00"), ("13:00", "16:00")]),
}
SCALE = 1000.0


def load_day(path):
    rec = np.load(path, allow_pickle=False)
    if rec.size == 0:
        return None
    day = os.path.basename(path).split('_')[1].replace('.npy', '')
    base = pd.Timestamp(day, tz='UTC')
    return pd.DataFrame({
        'ts_utc': base + pd.to_timedelta(rec['t'].astype('int64'), unit='s'),
        'Close': rec['c'].astype('float64') / SCALE,
        'High': rec['h'].astype('float64') / SCALE,
        'Low': rec['l'].astype('float64') / SCALE,
        'Volume': rec['v'].astype('float64'),
    })


def coverage(code, years=None):
    tz, windows = SESSIONS[code]
    files = sorted(glob.glob(os.path.join(CACHE, code, 'BID_*.npy')))
    rows = []
    for fp in files:
        day = os.path.basename(fp).split('_')[1].replace('.npy', '')
        yr = int(day[:4])
        if years and yr not in years:
            continue
        d = load_day(fp)
        if d is None or d.empty:
            rows.append(dict(Date=day, Year=yr, NRaw=0, NSess=0, First='', Last=''))
            continue
        d = d[d['Close'] > 0]
        if d.empty:
            rows.append(dict(Date=day, Year=yr, NRaw=0, NSess=0, First='', Last=''))
            continue
        loc = d['ts_utc'].dt.tz_convert(tz)
        # bars with genuine activity anywhere in the 24h day
        live = d[(d['Volume'] > 0) | (d['High'] != d['Low'])]
        lloc = live['ts_utc'].dt.tz_convert(tz)
        import datetime as _dt
        t = pd.DatetimeIndex(lloc).time
        m = np.zeros(len(live), dtype=bool)
        for a, b in windows:
            ha, ma = map(int, a.split(':'))
            hb, mb = map(int, b.split(':'))
            m |= (t >= _dt.time(ha, ma)) & (t < _dt.time(hb, mb))
        rows.append(dict(
            Date=day, Year=yr, NRaw=len(d), NLive=len(live), NSess=int(m.sum()),
            First=str(pd.DatetimeIndex(lloc).time.min()) if len(live) else '',
            Last=str(pd.DatetimeIndex(lloc).time.max()) if len(live) else ''))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 90)
    print("NKY - live-bar clock coverage in EXCHANGE-LOCAL time, by year")
    print("  NSess = 1-min bars falling inside our 09:00-11:30 + 12:30-15:00 window")
    print("=" * 90)
    nk = coverage('NKY')
    g = nk.groupby('Year').agg(Days=('Date', 'count'),
                               Days_NoData=('NLive', lambda s: int((s == 0).sum())),
                               Med_NLive=('NLive', 'median'),
                               Med_NSess=('NSess', 'median'),
                               Min_NSess=('NSess', 'min'))
    first_mode = nk[nk['NLive'] > 0].groupby('Year')['First'].agg(lambda s: s.mode().iloc[0])
    last_mode = nk[nk['NLive'] > 0].groupby('Year')['Last'].agg(lambda s: s.mode().iloc[0])
    g['FirstLiveBar'] = first_mode
    g['LastLiveBar'] = last_mode
    print(g.to_string())

    print()
    print("=" * 90)
    print("All six - median live bars inside the session window, by year")
    print("=" * 90)
    tab = {}
    for c in SESSIONS:
        cv = coverage(c)
        tab[c] = cv.groupby('Year')['NSess'].median()
    t = pd.DataFrame(tab)
    print(t.to_string())
    t.to_csv(os.path.join(LOG, 'phase12_session_coverage_by_year.csv'))

    print()
    print("=" * 90)
    print("All six - days per year with ZERO live bars in the session window")
    print("=" * 90)
    tab2 = {}
    for c in SESSIONS:
        cv = coverage(c)
        tab2[c] = cv.groupby('Year')['NSess'].apply(lambda s: int((s == 0).sum()))
    print(pd.DataFrame(tab2).to_string())
