# -*- coding: utf-8 -*-
"""
Add CRISIS AND REGIME LABELS to the analysis dataset.

WHY THIS EXISTS
  The project Executive Summary lists this under "Preprocessing tasks (A leads)":

      "Crisis labels: Define 'event' periods by dates (e.g. 2008-2009, 2010-11, 2020, etc.)
       or volatility thresholds."

  It is a Researcher A deliverable and it was missing from the dataset. Researcher B leads
  the crisis/regime analysis (16h in the plan) and cannot start it without an agreed,
  documented definition of what counts as a crisis - and if B invents one independently, the
  regime split stops being reproducible from the dataset alone.

BOTH DEFINITIONS ARE PROVIDED, because they answer different questions and disagree in
useful ways.

  1. CrisisLabel / IsCrisis - NAMED EVENT WINDOWS, fixed calendar dates.
     Transparent, citable, identical across all six indices, and the natural way to report
     "performance during COVID". The dates are global market episodes rather than
     index-specific ones, so a European index carries the same label as a US one on the same
     day. That is deliberate: it makes cross-index comparison meaningful.

  2. VolRegime - DATA-DRIVEN, from the trailing 63-day realized volatility of each index.
     Quartile cut-points, so each index gets its own thresholds. This catches stress that no
     named event covers and adapts to markets with structurally different volatility levels.

LOOK-AHEAD WARNING, and why there are two versions of the regime column
  VolRegime uses FULL-SAMPLE quartile cut-points. That is in-sample information. It is the
  correct choice for ex-post subsample reporting - "how did each model do in the calmest
  quartile versus the most volatile" - which is exactly what the regime analysis needs, and
  it is what the literature does. It must NEVER be used as a predictor.

  VolRegime_ExAnte uses EXPANDING-WINDOW quantiles computed only from data strictly before
  each date, so it contains no future information and CAN be used as a conditioning variable
  or a regressor. It disagrees with VolRegime early in the sample, when the expanding window
  has seen little history. Both are supplied; pick deliberately.

Output: 01_ANALYSIS_READY/<CODE>_analysis.csv  (rewritten with 5 columns added)
        01_ANALYSIS_READY/CRISIS_PERIODS.csv
        11_LOGS/phase15_regime_summary.csv
"""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
LOG = os.path.join(ROOT, '11_LOGS')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

# label, start, end, one-line justification of the dates chosen
CRISES = [
    ("Asian_Crisis_LTCM", "1997-07-02", "1998-12-31",
     "Thai baht float 1997-07-02 through the Russia default and LTCM rescue"),
    ("DotCom_Bust", "2000-03-10", "2002-10-09",
     "Nasdaq peak 2000-03-10 to the S&P 500 trough 2002-10-09"),
    ("GFC", "2007-08-01", "2009-06-30",
     "BNP Paribas fund freeze Aug 2007 through the NBER recession trough Jun 2009"),
    ("Euro_Sovereign_Debt", "2010-04-23", "2012-07-26",
     "Greek bailout request to Draghi's 'whatever it takes'"),
    ("China_Deval_Oil", "2015-08-11", "2016-02-11",
     "RMB devaluation 2015-08-11 to the Feb 2016 equity/credit trough"),
    ("Volmageddon", "2018-02-02", "2018-02-14",
     "The VIX complex blow-up and XIV termination"),
    ("Q4_2018_Selloff", "2018-10-01", "2018-12-26",
     "Q4 2018 drawdown to the Christmas Eve trough"),
    ("COVID_Crash", "2020-02-20", "2020-04-30",
     "Global equity peak 2020-02-19 through the April stabilisation"),
    ("Rate_Shock_2022", "2022-01-03", "2022-10-12",
     "2022 peak to the October CPI-shock trough; inflation and rate repricing"),
    ("Yen_Carry_Unwind", "2024-08-01", "2024-08-09",
     "BoJ hike and the global carry-trade unwind; Nikkei -12.4% on 2024-08-05"),
]

QLAB = ['Calm', 'Normal', 'Stressed', 'Crisis']


def label_crises(dates):
    lab = pd.Series('Normal', index=dates.index, dtype=object)
    for name, s, e in [(c[0], c[1], c[2]) for c in CRISES]:
        m = (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
        lab[m] = name
    return lab


def expanding_quartile_regime(vol, min_obs=750):
    """Regime from quantile cut-points estimated ONLY on data strictly before each date.

    Implemented with expanding ranks rather than a re-sorted quantile at every step: the
    rank of today's volatility within all PRIOR observations is itself the empirical
    quantile, which is both exact and O(n log n) instead of O(n^2).
    """
    v = vol.values.astype(float)
    n = len(v)
    out = np.full(n, np.nan)
    seen = []
    import bisect
    for i in range(n):
        x = v[i]
        if np.isfinite(x):
            if len(seen) >= min_obs:
                # empirical quantile of x among strictly prior observations
                out[i] = bisect.bisect_left(seen, x) / len(seen)
            bisect.insort(seen, x)
    q = pd.Series(out, index=vol.index)
    return pd.cut(q, [-0.001, 0.25, 0.50, 0.75, 1.001], labels=QLAB).astype(object)


def main():
    rows = []
    for c in CODES:
        p = os.path.join(ANA, f'{c}_analysis.csv')
        a = pd.read_csv(p, parse_dates=['Date'])

        # ---- 1. named event windows ----
        a['CrisisLabel'] = label_crises(a['Date'])
        a['IsCrisis'] = a['CrisisLabel'].ne('Normal')

        # ---- 2. data-driven regime from trailing realised volatility ----
        # trailing 63 trading days of daily returns, annualised. Uses the RETURN series, not
        # RV, so it is defined on every day including those where the intraday feed failed -
        # otherwise the Nikkei would have no regime label at all for 2016-17.
        vol = a['Return'].rolling(63, min_periods=40).std() * np.sqrt(252) * 100
        a['TrailVol63_Ann'] = vol

        # ex-post: full-sample quartiles. For subsample REPORTING only.
        a['VolRegime'] = pd.qcut(vol, 4, labels=QLAB).astype(object)
        a.loc[vol.isna(), 'VolRegime'] = np.nan

        # ex-ante: expanding-window quantiles. Safe as a regressor.
        a['VolRegime_ExAnte'] = expanding_quartile_regime(vol)

        a.to_csv(p, index=False, date_format='%Y-%m-%d', float_format='%.10g')

        b = a[a['InSample_B']]
        rows.append(dict(
            Code=c, Rows=len(a), Cols=len(a.columns),
            Crisis_Days_All=int(a['IsCrisis'].sum()),
            Crisis_Days_SampleB=int(b['IsCrisis'].sum()),
            Pct_Crisis_SampleB=round(100 * b['IsCrisis'].mean(), 1),
            Vol_Calm=round(float(vol[a['VolRegime'] == 'Calm'].max()), 1),
            Vol_Crisis_Cut=round(float(vol[a['VolRegime'] == 'Crisis'].min()), 1),
            ExAnte_Labelled=int(a['VolRegime_ExAnte'].notna().sum())))
        print(f"  [{c}] crisis days {int(a['IsCrisis'].sum()):4d} "
              f"(sample B {int(b['IsCrisis'].sum()):4d}, {100*b['IsCrisis'].mean():.1f}%)")

    cp = pd.DataFrame(CRISES, columns=['CrisisLabel', 'Start', 'End', 'Basis'])
    # how many days of each event land inside sample B, using SPX as the reference calendar
    ref = pd.read_csv(os.path.join(ANA, 'SPX_analysis.csv'), parse_dates=['Date'])
    cp['Days_in_SampleB'] = [
        int(((ref['Date'] >= s) & (ref['Date'] <= e) & ref['InSample_B']).sum())
        for s, e in zip(cp['Start'], cp['End'])]
    cp['In_SampleB'] = cp['Days_in_SampleB'] > 0
    cp.to_csv(os.path.join(ANA, 'CRISIS_PERIODS.csv'), index=False)

    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(LOG, 'phase15_regime_summary.csv'), index=False)

    pd.set_option('display.width', 220)
    print()
    print(s.to_string(index=False))
    print()
    print("crisis windows and whether the primary sample sees them:")
    print(cp[['CrisisLabel', 'Start', 'End', 'Days_in_SampleB', 'In_SampleB']].to_string(index=False))
    missed = cp[~cp['In_SampleB']]['CrisisLabel'].tolist()
    print()
    print(f"NOT covered by sample B (daily-only models still see these): {', '.join(missed)}")


if __name__ == "__main__":
    main()
