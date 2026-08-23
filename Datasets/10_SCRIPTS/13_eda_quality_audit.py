# -*- coding: utf-8 -*-
"""
EDA STAGE 1 - data quality audit.

This runs BEFORE any statistical description. The order matters: a fat tail and a data
error look identical in a histogram, so every distributional claim later in the EDA is
worthless until the mechanical faults are ruled out first.

WHAT IS CHECKED, AND WHY EACH ONE EARNS ITS PLACE

  structure       duplicate dates, non-monotonic dates, weekend dates, calendar gaps.
                  A duplicated date silently double-counts a return and inflates every
                  volatility estimate on that day; a weekend date means the session/timezone
                  mapping is wrong somewhere upstream.

  missingness     per column, over the full history AND over sample B separately. A column
                  that is 40% missing overall but complete inside sample B is fine; the
                  aggregate number alone would wrongly condemn it.

  staleness       zero returns and runs of identical closes. This is the single most common
                  defect in free index data and it biases volatility DOWNWARD, which is
                  exactly the direction that would flatter our models. Counted as both a
                  share and a maximum run length.

  bounds          non-positive prices, negative variances, non-positive volatility indices,
                  RV exactly zero, |return| beyond a plausible bound.

  internal        OHLC ordering (High >= max(O,C), Low <= min(O,C), High >= Low), and
  consistency     whether the daily close-to-close return reconciles with the CFD session
                  return already validated in script 08.

  session depth   number of intraday bars per day against the exchange-calendar expectation.
                  A day with a quarter of the usual bars produces an RV that is biased low
                  by construction and must be findable later, so short sessions are counted
                  and the worst are listed rather than just tallied.

Nothing is modified. This stage only measures; script 18 decides what to do about it.

Outputs: _validation/eda1_quality_by_index.csv
         _validation/eda1_missingness.csv
         _validation/eda1_flagged_rows.csv
         _validation/eda1_staleness_runs.csv
"""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAN = os.path.join(ROOT, '07_PANEL_INTERMEDIATE')
VAL = os.path.join(ROOT, '08_VALIDATION')
os.makedirs(VAL, exist_ok=True)

CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

# expected number of 5-min BARS in a full cash session, from each exchange calendar
EXPECTED_BARS = {"SPX": 78, "NDX": 78, "UKX": 102, "DAX": 102, "NKY": 60, "HSI": 66}

# a daily log return larger than this is not impossible, but it is rare enough that every
# instance must be individually explainable. 1987 aside, the largest daily move in any of
# these six indices in the modern era is about -13% (NKY 2011-03-15, SPX 2020-03-16).
RET_BOUND = 0.15


def max_run(mask):
    """Longest run of consecutive True values."""
    if not mask.any():
        return 0
    m = mask.values.astype(int)
    best = cur = 0
    for v in m:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def audit(code):
    p = pd.read_csv(os.path.join(PAN, f'{code}_panel_daily.csv'), parse_dates=['Date'])
    p = p.sort_values('Date').reset_index(drop=True)
    b = p[p['InSample_B']].copy()
    flagged = []
    rec = {'Code': code, 'Rows': len(p), 'Rows_SampleB': len(b),
           'First': str(p['Date'].min().date()), 'Last': str(p['Date'].max().date())}

    # ---------------- structure ----------------
    rec['Dup_Dates'] = int(p['Date'].duplicated().sum())
    rec['Dates_Monotonic'] = bool(p['Date'].is_monotonic_increasing)
    wknd = p[p['Date'].dt.dayofweek >= 5]
    rec['Weekend_Dates'] = len(wknd)
    for _, r in wknd.iterrows():
        flagged.append(dict(Code=code, Date=r['Date'].date(), Issue='weekend_date', Value=''))
    gap = p['Date'].diff().dt.days
    rec['Max_Calendar_Gap_Days'] = int(gap.max()) if len(gap.dropna()) else 0
    big = p[gap > 7]
    rec['Gaps_Over_7d'] = len(big)
    for _, r in big.iterrows():
        flagged.append(dict(Code=code, Date=r['Date'].date(), Issue='calendar_gap',
                            Value=int(gap.loc[r.name])))

    # ---------------- staleness ----------------
    zr = p['Return'] == 0
    rec['ZeroReturn_N'] = int(zr.sum())
    rec['ZeroReturn_Pct'] = round(100 * zr.mean(), 3)
    rec['ZeroReturn_MaxRun'] = max_run(zr.fillna(False))
    same_close = p['Close'].diff() == 0
    rec['RepeatedClose_MaxRun'] = max_run(same_close.fillna(False))
    zrb = b['Return'] == 0
    rec['ZeroReturn_Pct_SampleB'] = round(100 * zrb.mean(), 3) if len(b) else np.nan

    # ---------------- bounds ----------------
    rec['NonPositive_Price'] = int((p[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1).sum())
    rec['Negative_RV'] = int((p['RV_5min'] < 0).sum())
    rec['Zero_RV'] = int((p['RV_5min'] == 0).sum())
    rec['NonPositive_VolIdx'] = int((p['VolUsed'] <= 0).sum())
    ext = p[p['Return'].abs() > RET_BOUND]
    rec['AbsReturn_Over_15pct'] = len(ext)
    for _, r in ext.iterrows():
        flagged.append(dict(Code=code, Date=r['Date'].date(), Issue='extreme_return',
                            Value=round(float(r['Return']), 5)))

    # ---------------- OHLC consistency ----------------
    o, h, l, c = p['Open'], p['High'], p['Low'], p['Close']
    bad = (h < l) | (h < o) | (h < c) | (l > o) | (l > c)
    rec['OHLC_Violations'] = int(bad.sum())
    for _, r in p[bad].iterrows():
        flagged.append(dict(Code=code, Date=r['Date'].date(), Issue='ohlc_violation',
                            Value=f"O{r['Open']:.2f} H{r['High']:.2f} L{r['Low']:.2f} C{r['Close']:.2f}"))

    # ---------------- session depth ----------------
    exp = EXPECTED_BARS[code]
    nb = p.loc[p['HasRV_t'] == True, 'NBars_5min']
    rec['NBars_Median'] = float(nb.median()) if len(nb) else np.nan
    rec['NBars_Expected'] = exp
    short = p[(p['HasRV_t'] == True) & (p['NBars_5min'] < 0.5 * exp)]
    rec['ShortSessions_Under50pct'] = len(short)
    rec['ShortSessions_Under80pct'] = int(((p['HasRV_t'] == True) &
                                           (p['NBars_5min'] < 0.8 * exp)).sum())
    for _, r in short.iterrows():
        flagged.append(dict(Code=code, Date=r['Date'].date(), Issue='short_session',
                            Value=int(r['NBars_5min'])))

    # ---------------- daily vs CFD session return ----------------
    m = p[p['HasRV_t'] == True].dropna(subset=['Return', 'CFD_SessionReturn'])
    if len(m) > 30:
        # the CFD session return excludes the overnight gap, so it is compared against the
        # daily return NET of the overnight move, not against the raw daily return
        intraday_daily = m['Return'] - m['Overnight_LogRet']
        d = (m['CFD_SessionReturn'] - intraday_daily).abs()
        rec['CFD_vs_Daily_MedAbsDiff_bps'] = round(float(d.median()) * 1e4, 2)
        rec['CFD_vs_Daily_P99_bps'] = round(float(d.quantile(0.99)) * 1e4, 2)
        rec['CFD_vs_Daily_Corr'] = round(float(np.corrcoef(
            m['CFD_SessionReturn'], intraday_daily)[0, 1]), 5)
    return rec, flagged, p


def main():
    recs, flags, miss = [], [], []
    for c in CODES:
        r, f, p = audit(c)
        recs.append(r)
        flags += f
        for col in p.columns:
            if col in ('Date', 'Symbol'):
                continue
            n_all = int(p[col].isna().sum())
            sb = p[p['InSample_B']]
            n_b = int(sb[col].isna().sum()) if len(sb) else 0
            miss.append(dict(Code=c, Column=col,
                             Missing_All=n_all,
                             Pct_All=round(100 * n_all / len(p), 3),
                             Missing_SampleB=n_b,
                             Pct_SampleB=round(100 * n_b / max(len(sb), 1), 3)))
        print(f"  [{c}] audited")

    q = pd.DataFrame(recs)
    q.to_csv(os.path.join(VAL, 'eda1_quality_by_index.csv'), index=False)
    md = pd.DataFrame(miss)
    md.to_csv(os.path.join(VAL, 'eda1_missingness.csv'), index=False)
    fl = pd.DataFrame(flags)
    if len(fl):
        fl = fl.sort_values(['Issue', 'Code', 'Date'])
    fl.to_csv(os.path.join(VAL, 'eda1_flagged_rows.csv'), index=False)

    pd.set_option('display.width', 250)
    show = ['Code', 'Rows', 'Rows_SampleB', 'Dup_Dates', 'Weekend_Dates', 'Gaps_Over_7d',
            'ZeroReturn_Pct', 'ZeroReturn_MaxRun', 'RepeatedClose_MaxRun', 'Zero_RV',
            'OHLC_Violations', 'AbsReturn_Over_15pct', 'NBars_Median', 'NBars_Expected',
            'ShortSessions_Under50pct', 'ShortSessions_Under80pct']
    print()
    print(q[show].to_string(index=False))
    print()
    print("CFD reconciliation (session return vs daily return net of overnight):")
    print(q[['Code', 'CFD_vs_Daily_Corr', 'CFD_vs_Daily_MedAbsDiff_bps',
             'CFD_vs_Daily_P99_bps']].to_string(index=False))
    print()
    if len(fl):
        print("flagged rows by issue:")
        print(fl.groupby('Issue').size().to_string())
    print()
    print("columns >5% missing INSIDE sample B:")
    bad = md[md['Pct_SampleB'] > 5]
    print(bad.to_string(index=False) if len(bad) else "  none")


if __name__ == "__main__":
    main()
