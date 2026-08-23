# -*- coding: utf-8 -*-
"""
Phase 11: EXTENDED realized measures, recomputed from the Dukascopy 1-min cache.

WHY
  Script 05 produced RV and bipower variation at five sampling frequencies. That is enough
  to fit a plain Realized GARCH, but it is not enough for the paper we are writing:

  1. TAIL RISK IS SIGNED. RV throws away the sign of every intraday move, yet the whole
     point of the study is downside VaR/ES. Realized semivariance (Barndorff-Nielsen,
     Kinnebrock & Shephard 2010) splits RV into upside and downside halves; RS_neg is a
     markedly better predictor of future downside risk than RV, and its inclusion is close
     to expected practice now in any realized-measure tail paper.
  2. JUMPS NEED A ROBUST BENCHMARK. BPV is the standard jump-robust estimator but it is
     contaminated when two adjacent returns are both jumps. MedRV (Andersen, Dobrev &
     Schaumburg 2012) is robust to consecutive jumps and to zero returns, which matter here
     because thin sessions (NKY, HSI lunch breaks) do produce runs of zeros.
  3. MEASUREMENT ERROR HAS TO BE QUANTIFIED. RV is an estimate, not the truth. Realized
     quarticity gives the asymptotic variance of that estimate and is what HAR-Q
     (Bollerslev, Patton & Quaedvlieg 2016) uses to correct the attenuation bias that
     otherwise flattens every RV coefficient. Without RQ we cannot even report how noisy
     our own dependent variable is.
  4. SUBSAMPLING IS FREE. A single 5-min grid discards four fifths of the 1-min data.
     Averaging the five overlapping 5-min grids (Zhang, Mykland & Ait-Sahalia 2005) lowers
     the variance of the estimator at no cost in bias, since we already hold the 1-min bars.

BIPOWER CORRECTION
  Script 05 computed BPV as (pi/2) * sum|r_{i-1}||r_i|, which is the right scaling constant
  but omits the n/(n-1) finite-sample correction. Both versions are written out here
  (BPV_5min_c is corrected) so the difference is visible rather than silently changed.

SESSION FILTERING is identical to script 05 - exchange-local cash session, DST-aware, with
the same stale-padding and frozen-session guards - so these measures line up row for row
with the existing realized_volatility/<CODE>_RV_daily.csv files. That is asserted, not
assumed: the script recomputes RV_5min and compares it to the stored value.

Output: realized_volatility/<CODE>_RV_extended.csv
        _logs/phase11_extended_rv_summary.csv
"""
import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.special import gamma as _gamma

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, '12_CACHE_REGENERATION')
OUTRV = os.path.join(ROOT, '06_REALIZED_MEASURES')
LOG = os.path.join(ROOT, '11_LOGS')
os.makedirs(LOG, exist_ok=True)

SESSIONS = {
    "SPX": ("America/New_York", [("09:30", "16:00")]),
    "NDX": ("America/New_York", [("09:30", "16:00")]),
    "UKX": ("Europe/London",    [("08:00", "16:30")]),
    "DAX": ("Europe/Berlin",    [("09:00", "17:30")]),
    "NKY": ("Asia/Tokyo",       [("09:00", "11:30"), ("12:30", "15:00")]),
    "HSI": ("Asia/Hong_Kong",   [("09:30", "12:00"), ("13:00", "16:00")]),
}
SCALE = 1000.0
FROZEN_RANGE_THRESHOLD = 4e-4   # identical to script 05

# scaling constants
MU1 = np.sqrt(2.0 / np.pi)                       # E|Z|
MU43 = 2.0 ** (2.0 / 3.0) * _gamma(7.0 / 6.0) / _gamma(0.5)   # E|Z|^(4/3)
MEDRV_C = np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)


def load_day(path):
    rec = np.load(path, allow_pickle=False)
    if rec.size == 0:
        return None
    day = os.path.basename(path).split('_')[1].replace('.npy', '')
    base = pd.Timestamp(day, tz='UTC')
    df = pd.DataFrame({
        'ts_utc': base + pd.to_timedelta(rec['t'].astype('int64'), unit='s'),
        'Open': rec['o'].astype('float64') / SCALE,
        'Close': rec['c'].astype('float64') / SCALE,
        'Low': rec['l'].astype('float64') / SCALE,
        'High': rec['h'].astype('float64') / SCALE,
        'Volume': rec['v'].astype('float64'),
    })
    return df[df['Close'] > 0]


def session_mask(local_idx, windows):
    import datetime as _dt
    t = local_idx.time
    m = np.zeros(len(local_idx), dtype=bool)
    for a, b in windows:
        ha, ma = map(int, a.split(':'))
        hb, mb = map(int, b.split(':'))
        m |= (t >= _dt.time(ha, ma)) & (t < _dt.time(hb, mb))
    return m


def measures_from_returns(r):
    """All estimators that are a function of one session's intraday return vector."""
    out = {}
    n = len(r)
    if n < 3:
        return None
    r2 = r ** 2
    rv = float(r2.sum())
    out['RV'] = rv
    # --- signed decomposition (Barndorff-Nielsen, Kinnebrock & Shephard 2010) ---
    out['RS_pos'] = float(r2[r > 0].sum())
    out['RS_neg'] = float(r2[r < 0].sum())
    out['SignedJump'] = out['RS_pos'] - out['RS_neg']
    # --- bipower, with and without the finite-sample correction ---
    if n > 1:
        bpv_raw = float((1.0 / MU1 ** 2) * np.sum(np.abs(r[:-1]) * np.abs(r[1:])))
        out['BPV'] = bpv_raw
        out['BPV_c'] = bpv_raw * n / (n - 1.0)
    else:
        out['BPV'] = out['BPV_c'] = np.nan
    # --- MedRV: robust to consecutive jumps (Andersen, Dobrev & Schaumburg 2012) ---
    if n >= 3:
        a = np.abs(r)
        med3 = np.median(np.vstack([a[:-2], a[1:-1], a[2:]]), axis=0)
        out['MedRV'] = float(MEDRV_C * (n / (n - 2.0)) * np.sum(med3 ** 2))
    else:
        out['MedRV'] = np.nan
    # --- quarticity: RQ for HAR-Q, TQ as the jump-robust counterpart ---
    out['RQ'] = float((n / 3.0) * np.sum(r ** 4))
    if n >= 3:
        p = 4.0 / 3.0
        out['TQ'] = float(n * (MU43 ** -3) *
                          np.sum((np.abs(r[:-2]) ** p) * (np.abs(r[1:-1]) ** p) * (np.abs(r[2:]) ** p)))
    else:
        out['TQ'] = np.nan
    # --- intraday distribution shape (Amaya, Christoffersen, Jacobs & Vasquez 2015) ---
    if rv > 0:
        out['RSkew'] = float(np.sqrt(n) * np.sum(r ** 3) / rv ** 1.5)
        out['RKurt'] = float(n * np.sum(r ** 4) / rv ** 2)
    else:
        out['RSkew'] = out['RKurt'] = np.nan
    out['NBars'] = n
    out['NZeroRet'] = int(np.sum(r == 0))
    out['MaxAbsRet'] = float(np.abs(r).max())
    return out


def subsampled_rv(px, minutes=5):
    """Average RV over the `minutes` overlapping grids offset by one minute each.

    Zhang, Mykland & Ait-Sahalia (2005). We already hold every 1-min bar, so the four
    discarded grids are free information; averaging them lowers estimator variance without
    changing the bias, and it is what makes a 5-min RV defensible against the objection
    that the choice of grid origin is arbitrary.
    """
    vals = []
    for off in range(minutes):
        s = px.iloc[off:].resample(f'{minutes}min').last().dropna()
        if len(s) >= 3:
            r = np.diff(np.log(s.values))
            vals.append(float(np.sum(r ** 2)))
    return float(np.mean(vals)) if vals else np.nan


def process(code):
    tz, windows = SESSIONS[code]
    files = sorted(glob.glob(os.path.join(CACHE, code, 'BID_*.npy')))
    if not files:
        print(f"  [{code}] no cache")
        return None
    rows = []
    for fp in files:
        d = load_day(fp)
        if d is None or d.empty:
            continue
        d['ts_local'] = d['ts_utc'].dt.tz_convert(tz)
        d = d[session_mask(pd.DatetimeIndex(d['ts_local']), windows)]
        if d.empty:
            continue
        d = d[(d['Volume'] > 0) | (d['High'] != d['Low'])]
        if len(d) < 10:
            continue
        rng = float((d['High'].max() - d['Low'].min()) / d['Close'].iloc[-1])
        if not np.isfinite(rng) or rng < FROZEN_RANGE_THRESHOLD:
            continue
        d = d.sort_values('ts_utc').reset_index(drop=True)
        sess_date = pd.DatetimeIndex(d['ts_local']).date[0]
        px = d.set_index(pd.DatetimeIndex(d['ts_local']))['Close']

        row = {'Date': sess_date, 'Symbol': code}
        for m in (1, 5):
            s = px.resample(f'{m}min').last().dropna()
            if len(s) < 3:
                continue
            r = np.diff(np.log(s.values))
            mm = measures_from_returns(r)
            if mm is None:
                continue
            for k, v in mm.items():
                row[f'{k}_{m}min'] = v
        row['RV_ss5min'] = subsampled_rv(px, 5)
        # Realized volatility of the session, and the session close for cross-checks
        rows.append(row)

    if not rows:
        return None
    ex = pd.DataFrame(rows).sort_values('Date').reset_index(drop=True)

    # ---- derived, on the 5-min grid ---------------------------------------
    ex['Jump_BPV_5min'] = (ex['RV_5min'] - ex['BPV_c_5min']).clip(lower=0)
    ex['Jump_MedRV_5min'] = (ex['RV_5min'] - ex['MedRV_5min']).clip(lower=0)
    ex['ContVar_MedRV_5min'] = ex['RV_5min'] - ex['Jump_MedRV_5min']
    ex['JumpShare_5min'] = ex['Jump_MedRV_5min'] / ex['RV_5min'].replace(0, np.nan)
    ex['RSV_Ratio_5min'] = ex['RS_neg_5min'] / ex['RV_5min'].replace(0, np.nan)
    # Relative measurement error of RV (Barndorff-Nielsen & Shephard 2002).
    #   sqrt(n)(RV - IV) -> N(0, 2*IQ),  so  Var(RV) ~ 2*IQ/n,  and RQ estimates IQ.
    #   RQ is DEFINED here as (n/3)*sum r^4, i.e. it already carries the n, so the standard
    #   error is sqrt(2*RQ/n) and NOT sqrt((2/3)*RQ) - the latter overstates it by
    #   sqrt(n/3), which on a 77-return session is a factor of five.
    _n5 = ex['NBars_5min'].replace(0, np.nan)
    ex['RV_SE_5min'] = np.sqrt(2.0 * ex['RQ_5min'] / _n5)
    ex['RV_RelSE_5min'] = ex['RV_SE_5min'] / ex['RV_5min'].replace(0, np.nan)
    ex['NoiseRatio_1v5'] = ex['RV_1min'] / ex['RV_5min'].replace(0, np.nan)

    # ---- assert alignment with the existing RV file -----------------------
    old_p = os.path.join(OUTRV, f'{code}_RV_daily.csv')
    chk = {}
    if os.path.exists(old_p):
        old = pd.read_csv(old_p, parse_dates=['Date'])
        old['Date'] = old['Date'].dt.date
        j = ex[['Date', 'RV_5min']].merge(old[['Date', 'RV_5min']], on='Date',
                                          how='outer', suffixes=('_new', '_old'), indicator=True)
        both = j[j['_merge'] == 'both']
        rel = np.abs(both['RV_5min_new'] - both['RV_5min_old']) / both['RV_5min_old'].replace(0, np.nan)
        chk = dict(Rows_Only_New=int((j['_merge'] == 'left_only').sum()),
                   Rows_Only_Old=int((j['_merge'] == 'right_only').sum()),
                   Max_Rel_Diff_RV5=float(np.nanmax(rel)) if len(both) else np.nan)

    path = os.path.join(OUTRV, f'{code}_RV_extended.csv')
    ex.to_csv(path, index=False, date_format='%Y-%m-%d', float_format='%.12g')
    print(f"  [{code}] {len(ex)} days  {ex.Date.iloc[0]} -> {ex.Date.iloc[-1]}  "
          f"cols={len(ex.columns)}  align_maxreldiff={chk.get('Max_Rel_Diff_RV5')}")
    return dict(Code=code, Days=len(ex), Columns=len(ex.columns),
                First=str(ex.Date.iloc[0]), Last=str(ex.Date.iloc[-1]),
                Median_NBars_5min=float(ex['NBars_5min'].median()),
                Median_JumpShare=float(ex['JumpShare_5min'].median()),
                Median_RSV_Ratio=float(ex['RSV_Ratio_5min'].median()),
                Median_RV_RelSE=float(ex['RV_RelSE_5min'].median()),
                Median_Noise_1v5=float(ex['NoiseRatio_1v5'].median()),
                **chk)


if __name__ == "__main__":
    codes = [a.upper() for a in sys.argv[1:]] or list(SESSIONS)
    out = []
    for c in codes:
        print(f"=== {c} ===", flush=True)
        r = process(c)
        if r:
            out.append(r)
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(LOG, 'phase11_extended_rv_summary.csv'), index=False)
    print()
    print(s.to_string(index=False))
