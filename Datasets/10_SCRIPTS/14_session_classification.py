# -*- coding: utf-8 -*-
"""
EDA STAGE 2 - classify every intraday session as FULL, HALFDAY or DEFECT.

WHY A PLAIN COVERAGE THRESHOLD IS NOT ENOUGH
  Stage 13d showed NKY running at 80% and 66% of the expected session minutes in 2016 and
  2017. A naive rule - "drop any day below 90% coverage" - would delete those days, which is
  right, but it would ALSO delete every genuine exchange half-day: the 13:00 closes around
  US Independence Day and Christmas, the Christmas Eve and New Year's Eve half-sessions in
  London and Hong Kong. Those are not data faults. The market really did trade for three
  hours, and the realized variance measured over those three hours is a correct measurement
  of a short session.

  The two cases are distinguishable by WHERE the minutes are missing:

    HALFDAY  the session starts on time and stops early. Everything missing is one
             contiguous block at the END. Recurs on calendar-predictable dates.
    DEFECT   the opening minutes are absent, or there is a hole in the middle of the
             session. The market was open and we simply do not have the data.

  The distinction matters enormously for a volatility study because intraday volatility is
  U-shaped. Losing the last hour of a half-day removes a known, modest slice of the daily
  variance. Losing the OPENING hour - which is what happened to NKY in 2016-17 - removes the
  single most volatile part of the session and biases RV downward by far more than the
  missing time fraction suggests. That bias cannot be scaled away, because the scaling factor
  would itself have to be estimated from the volatile period we are missing.

COVERAGE IS MEASURED ON THE 5-MINUTE GRID, NOT THE 1-MINUTE GRID
  A first pass measured coverage in minutes and produced nonsense: 674 "half-days" for SPX,
  about 45 a year, when the US exchanges schedule roughly three. The cause is that a 1-min
  bar with no quote change and no volume is indistinguishable from padding and is dropped
  upstream, so a quiet session legitimately shows 93-97% minute coverage. That is not
  missing data - the price simply did not move.

  What actually matters for RV is whether each 5-MINUTE BLOCK contains at least one quote,
  because that is the grid the estimator samples on. A 5-min block with one bar in it
  contributes a perfectly good return. So coverage is defined here as

      covered 5-min blocks / expected 5-min blocks

  which is the quantity that governs whether RV is complete, and it makes the half-day and
  defect populations separate cleanly.

CLASSIFICATION RULE (on the 5-min grid)
  FULL     coverage >= 0.95
  HALFDAY  0.35 <= coverage < 0.95, opening block present, and everything missing is one
           contiguous block at the end (no interior hole beyond one block)
  DEFECT   anything else - missing open, an interior hole, or coverage below 0.35

Outputs: _validation/eda2_session_class.csv   (one row per index-day)
         _validation/eda2_session_class_summary.csv
"""
import os
import glob
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, '12_CACHE_REGENERATION')
VAL = os.path.join(ROOT, '08_VALIDATION')
os.makedirs(VAL, exist_ok=True)
SCALE = 1000.0

SESS = {
    "SPX": ("America/New_York", [("09:30", "16:00")]),
    "NDX": ("America/New_York", [("09:30", "16:00")]),
    "UKX": ("Europe/London",    [("08:00", "16:30")]),
    "DAX": ("Europe/Berlin",    [("09:00", "17:30")]),
    "NKY": ("Asia/Tokyo",       [("09:00", "11:30"), ("12:30", "15:00")]),  # pre-2024-11-05 close; see windows_for()
    "HSI": ("Asia/Hong_Kong",   [("09:30", "12:00"), ("13:00", "16:00")]),
}
NKY_SESSION_CHANGE = "2024-11-05"  # TSE extended cash-session close from 15:00 to 15:30 JST


def windows_for(code, date_str):
    """Per-day session windows - only NKY is date-dependent. Identical logic to
    05_build_intraday_and_RV.py / 12_extended_realized_measures.py's windows_for()."""
    tz, windows = SESS[code]
    if code == "NKY" and date_str >= NKY_SESSION_CHANGE:
        return tz, [("09:00", "11:30"), ("12:30", "15:30")]
    return tz, windows

GRID = 5           # minutes; the sampling grid RV is computed on
OPEN_TOL = 1       # 5-min blocks; the open counts as present within this many blocks
INTERIOR_GAP = 1   # 5-min blocks; a hole longer than this inside the session is a defect
FULL_COV = 0.95
MIN_COV = 0.35


def expected_blocks(windows):
    """Start-minute of every 5-min block the cash session is supposed to cover."""
    mins = []
    for a, b in windows:
        ha, ma = map(int, a.split(':'))
        hb, mb = map(int, b.split(':'))
        mins += list(range(ha * 60 + ma, hb * 60 + mb, GRID))
    return np.array(sorted(mins))


def classify(code):
    rows = []
    for fp in sorted(glob.glob(os.path.join(CACHE, code, 'BID_*.npy'))):
        rec = np.load(fp, allow_pickle=False)
        if rec.size == 0:
            continue
        day = os.path.basename(fp).split('_')[1].replace('.npy', '')
        tz, windows = windows_for(code, day)
        exp = expected_blocks(windows)
        nexp = len(exp)
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
            continue
        loc = pd.DatetimeIndex(d['ts'].dt.tz_convert(tz))
        mod = loc.hour * 60 + loc.minute
        # map each observed minute onto the start-minute of its 5-min block
        blocks = (mod // GRID) * GRID
        present = np.intersect1d(np.unique(blocks), exp)
        n = len(present)
        if n == 0:
            continue
        cov = n / nexp

        # positions on the expected block grid, so gaps are counted in BLOCKS
        pos = np.searchsorted(exp, present)
        has_open = bool(pos.min() <= OPEN_TOL)
        has_close = bool(pos.max() >= nexp - 1 - OPEN_TOL)
        span = np.arange(pos.min(), pos.max() + 1)
        miss_in_span = np.setdiff1d(span, pos)
        if len(miss_in_span):
            brk = np.where(np.diff(miss_in_span) > 1)[0]
            runs = np.split(miss_in_span, brk + 1)
            max_gap = max(len(r) for r in runs)
        else:
            max_gap = 0

        if cov >= FULL_COV:
            cls = 'FULL'
        elif cov < MIN_COV or not has_open or max_gap > INTERIOR_GAP:
            cls = 'DEFECT'
        else:
            cls = 'HALFDAY'

        rows.append(dict(Symbol=code, Date=day, NSess=n, NExpected=nexp,
                         Coverage=round(cov, 4), HasOpen=has_open, HasClose=has_close,
                         MaxInteriorGap_blocks=int(max_gap), Class=cls))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_rows = []
    for c in SESS:
        df = classify(c)
        all_rows.append(df)
        print(f"  [{c}] {len(df)} sessions classified")
    cl = pd.concat(all_rows, ignore_index=True)
    cl['Date'] = pd.to_datetime(cl['Date'])
    cl.to_csv(os.path.join(VAL, 'eda2_session_class.csv'), index=False,
              date_format='%Y-%m-%d')

    print()
    print("=" * 78)
    print("session class counts")
    print("=" * 78)
    ct = cl.pivot_table(index='Symbol', columns='Class', values='Date',
                        aggfunc='count').fillna(0).astype(int)
    ct['Total'] = ct.sum(axis=1)
    ct['Pct_DEFECT'] = (100 * ct.get('DEFECT', 0) / ct['Total']).round(2)
    print(ct.to_string())

    print()
    print("=" * 78)
    print("DEFECT days per index-year")
    print("=" * 78)
    cl['Year'] = cl['Date'].dt.year
    dd = cl[cl['Class'] == 'DEFECT'].pivot_table(
        index='Year', columns='Symbol', values='Date', aggfunc='count').fillna(0).astype(int)
    print(dd.to_string())

    print()
    print("=" * 78)
    print("HALFDAY days - the recurring calendar dates confirm these are real half-sessions")
    print("=" * 78)
    hd = cl[cl['Class'] == 'HALFDAY'].copy()
    hd['MD'] = hd['Date'].dt.strftime('%m-%d')
    for c in SESS:
        s = hd[hd['Symbol'] == c]['MD'].value_counts().head(5)
        if len(s):
            print(f"  {c} ({len(hd[hd['Symbol']==c])} half-days): " +
                  ", ".join(f"{k} x{v}" for k, v in s.items()))

    ct.to_csv(os.path.join(VAL, 'eda2_session_class_summary.csv'))
