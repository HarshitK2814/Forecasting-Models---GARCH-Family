# -*- coding: utf-8 -*-
"""
Pin down the NKY intraday defect and price the options.

Stage 1 -> NKY has 217 short sessions. Stage 13c -> the cause is that the Dukascopy
JPNIDXJPY feed does not cover the full Tokyo cash session between 2015 and 2020: 174 days
of 2015 have ZERO bars inside 09:00-15:00 local, and 2016/2017 run at 240/196 minutes
against an expected 300.

Two things have to be established before any cleaning rule is written:
  (a) WHICH clock hours are missing. If the feed drops a consistent block, the surviving
      window is a biased sample of the session - intraday volatility is U-shaped, so
      losing the open or the close removes far more variance than losing midday, and a
      pro-rata time scaling would be wrong.
  (b) WHAT a coverage threshold COSTS. Sample B is a balanced panel: dropping NKY days
      removes those dates from every index. The threshold is therefore not a free choice.
"""
import os
import glob
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, '12_CACHE_REGENERATION')
PAN = os.path.join(ROOT, '07_PANEL_INTERMEDIATE')
LOG = os.path.join(ROOT, '11_LOGS')
SCALE = 1000.0
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]
SESS = {
    "SPX": ("America/New_York", [("09:30", "16:00")], 390),
    "NDX": ("America/New_York", [("09:30", "16:00")], 390),
    "UKX": ("Europe/London",    [("08:00", "16:30")], 510),
    "DAX": ("Europe/Berlin",    [("09:00", "17:30")], 510),
    "NKY": ("Asia/Tokyo",       [("09:00", "11:30"), ("12:30", "15:00")], 300),
    "HSI": ("Asia/Hong_Kong",   [("09:30", "12:00"), ("13:00", "16:00")], 330),
}


def day_bars(code, fp):
    tz, windows, _ = SESS[code]
    rec = np.load(fp, allow_pickle=False)
    if rec.size == 0:
        return None
    day = os.path.basename(fp).split('_')[1].replace('.npy', '')
    base = pd.Timestamp(day, tz='UTC')
    d = pd.DataFrame({
        'ts': base + pd.to_timedelta(rec['t'].astype('int64'), unit='s'),
        'c': rec['c'].astype('float64') / SCALE,
        'h': rec['h'].astype('float64') / SCALE,
        'l': rec['l'].astype('float64') / SCALE,
        'v': rec['v'].astype('float64')})
    d = d[d['c'] > 0]
    d = d[(d['v'] > 0) | (d['h'] != d['l'])]
    if d.empty:
        return None
    loc = pd.DatetimeIndex(d['ts'].dt.tz_convert(tz))
    import datetime as _dt
    t = loc.time
    m = np.zeros(len(d), dtype=bool)
    for a, b in windows:
        ha, ma = map(int, a.split(':'))
        hb, mb = map(int, b.split(':'))
        m |= (t >= _dt.time(ha, ma)) & (t < _dt.time(hb, mb))
    return day, loc[m], int(m.sum())


print("=" * 84)
print("A. NKY - minutes present per LOCAL HOUR, averaged over each year")
print("   Expected: 60 in hours 09,10,13,14 ; 30 in hour 11 ; 30 in hour 12 ; 0 elsewhere")
print("=" * 84)
files = sorted(glob.glob(os.path.join(CACHE, 'NKY', 'BID_*.npy')))
rows = []
for fp in files:
    r = day_bars('NKY', fp)
    if r is None:
        continue
    day, loc, n = r
    if n == 0:
        continue
    h = pd.Series(loc.hour).value_counts()
    rows.append({'Year': int(day[:4]), **{f'h{k:02d}': v for k, v in h.items()}})
hh = pd.DataFrame(rows).fillna(0).groupby('Year').mean().round(0).astype(int)
cols = [c for c in sorted(hh.columns) if hh[c].sum() > 0]
print(hh[cols].to_string())

print()
print("=" * 84)
print("B. Coverage ratio per day (session minutes present / expected), all six indices")
print("=" * 84)
cov_rows = []
for c in CODES:
    exp = SESS[c][2]
    for fp in sorted(glob.glob(os.path.join(CACHE, c, 'BID_*.npy'))):
        r = day_bars(c, fp)
        if r is None:
            continue
        day, loc, n = r
        if n == 0:
            continue
        cov_rows.append(dict(Symbol=c, Date=day, NSess=n, Coverage=n / exp))
cov = pd.DataFrame(cov_rows)
cov['Date'] = pd.to_datetime(cov['Date'])
cov.to_csv(os.path.join(LOG, 'phase12_daily_coverage.csv'), index=False,
           date_format='%Y-%m-%d')
print(cov.groupby('Symbol')['Coverage'].describe(
    percentiles=[.01, .05, .10, .25, .5]).round(3).to_string())

print()
print("=" * 84)
print("C. What a coverage threshold COSTS the balanced sample B")
print("=" * 84)
panels = {c: pd.read_csv(os.path.join(PAN, f'{c}_panel_daily.csv'), parse_dates=['Date'])
          for c in CODES}
base = None
for c in CODES:
    d = set(panels[c].loc[panels[c]['InSample_B'], 'Date'])
    base = d if base is None else (base & d)
print(f"  current balanced sample B: {len(base)} days")
for thr in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99):
    inter = None
    for c in CODES:
        ok = set(cov.loc[(cov['Symbol'] == c) & (cov['Coverage'] >= thr), 'Date'])
        d = set(panels[c].loc[panels[c]['InSample_B'], 'Date']) & ok
        inter = d if inter is None else (inter & d)
    lost_by = {}
    for c in CODES:
        ok = set(cov.loc[(cov['Symbol'] == c) & (cov['Coverage'] >= thr), 'Date'])
        lost_by[c] = len(set(panels[c].loc[panels[c]['InSample_B'], 'Date']) - ok)
    print(f"  thr>={thr:.2f}  balanced={len(inter):5d}  "
          f"({100*len(inter)/len(base):5.1f}% of current)  "
          f"dropped per index: " + " ".join(f"{k}={v}" for k, v in lost_by.items()))

print()
print("=" * 84)
print("D. NKY coverage by year inside sample B")
print("=" * 84)
nk = cov[cov['Symbol'] == 'NKY'].copy()
b = set(panels['NKY'].loc[panels['NKY']['InSample_B'], 'Date'])
nk = nk[nk['Date'].isin(b)]
nk['Year'] = nk['Date'].dt.year
print(nk.groupby('Year')['Coverage'].agg(
    n='count', median='median', p10=lambda s: s.quantile(.1),
    frac_ge_90=lambda s: (s >= .9).mean()).round(3).to_string())
