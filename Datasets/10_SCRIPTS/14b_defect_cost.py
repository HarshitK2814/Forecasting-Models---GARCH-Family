# -*- coding: utf-8 -*-
"""
What do the DEFECT sessions cost, and what are the 2012-2014 ones?

Two distinct defect clusters came out of stage 2:
  - NKY 2015-2019 (672 days, peaking at 258 in 2016 and 257 in 2017)
  - SPX/NDX/UKX 2012-2014
The second cluster sits mostly BEFORE sample B starts (2013-09-30), so it may cost nothing.
This prices both and characterises the mechanism of the second.
"""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAN = os.path.join(ROOT, '07_PANEL_INTERMEDIATE')
VAL = os.path.join(ROOT, '08_VALIDATION')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

cl = pd.read_csv(os.path.join(VAL, 'eda2_session_class.csv'), parse_dates=['Date'])
P = {c: pd.read_csv(os.path.join(PAN, f'{c}_panel_daily.csv'), parse_dates=['Date'])
     for c in CODES}

print("=" * 80)
print("A. Mechanism of the 2012-2014 defects (SPX / NDX / UKX)")
print("=" * 80)
e = cl[(cl['Class'] == 'DEFECT') & (cl['Date'] < '2015-01-01') &
       (cl['Symbol'].isin(['SPX', 'NDX', 'UKX']))]
print(e.groupby('Symbol').agg(
    n=('Date', 'count'),
    med_cov=('Coverage', 'median'),
    frac_missing_open=('HasOpen', lambda s: round(1 - s.mean(), 3)),
    med_interior_gap=('MaxInteriorGap_blocks', 'median'),
    max_interior_gap=('MaxInteriorGap_blocks', 'max')).to_string())
print()
print("  Same for the NKY 2015-2019 cluster:")
n = cl[(cl['Class'] == 'DEFECT') & (cl['Symbol'] == 'NKY') &
       (cl['Date'] >= '2015-01-01') & (cl['Date'] < '2020-01-01')]
print(f"    n={len(n)}  med_cov={n['Coverage'].median():.3f}  "
      f"frac_missing_open={1-n['HasOpen'].mean():.3f}  "
      f"med_interior_gap={n['MaxInteriorGap_blocks'].median():.0f}")

print()
print("=" * 80)
print("B. DEFECT days that actually fall INSIDE sample B (2013-09-30 onward)")
print("=" * 80)
rows = []
for c in CODES:
    inb = set(P[c].loc[P[c]['InSample_B'], 'Date'])
    d = cl[(cl['Symbol'] == c) & (cl['Class'] == 'DEFECT')]
    d_in = d[d['Date'].isin(inb)]
    h = cl[(cl['Symbol'] == c) & (cl['Class'] == 'HALFDAY')]
    h_in = h[h['Date'].isin(inb)]
    rows.append(dict(Code=c, SampleB_days=len(inb),
                     DEFECT_in_B=len(d_in), HALFDAY_in_B=len(h_in),
                     Pct_DEFECT=round(100 * len(d_in) / len(inb), 2)))
print(pd.DataFrame(rows).to_string(index=False))

print()
print("=" * 80)
print("C. Balanced sample B under each cleaning rule")
print("=" * 80)
good_full = {c: set(cl.loc[(cl['Symbol'] == c) & (cl['Class'] == 'FULL'), 'Date'])
             for c in CODES}
good_fh = {c: set(cl.loc[(cl['Symbol'] == c) & (cl['Class'].isin(['FULL', 'HALFDAY'])),
                         'Date']) for c in CODES}


def balanced(keep, codes=CODES):
    inter = None
    for c in codes:
        d = set(P[c].loc[P[c]['InSample_B'], 'Date']) & keep[c]
        inter = d if inter is None else (inter & d)
    return inter


b0 = None
for c in CODES:
    d = set(P[c].loc[P[c]['InSample_B'], 'Date'])
    b0 = d if b0 is None else (b0 & d)
print(f"  R0  no cleaning                         : {len(b0):5d} days")
r1 = balanced(good_fh)
print(f"  R1  drop DEFECT, keep half-days         : {len(r1):5d} days  "
      f"({r1 and min(r1).date()} -> {r1 and max(r1).date()})")
r2 = balanced(good_full)
print(f"  R2  drop DEFECT and half-days           : {len(r2):5d} days  "
      f"({r2 and min(r2).date()} -> {r2 and max(r2).date()})")
r3 = balanced(good_fh, [c for c in CODES if c != 'NKY'])
print(f"  R3  drop DEFECT, keep half-days, NO NKY : {len(r3):5d} days (5 indices)")

print()
print("  How many days does EACH index remove from the R1 balanced panel?")
for c in CODES:
    others = None
    for k in CODES:
        if k == c:
            continue
        d = set(P[k].loc[P[k]['InSample_B'], 'Date']) & good_fh[k]
        others = d if others is None else (others & d)
    print(f"    {c}: {len(others) - len(r1):5d} days lost because of {c} alone")

print()
print("=" * 80)
print("D. NKY under R1, by year - is the loss concentrated or spread?")
print("=" * 80)
inb = set(P['NKY'].loc[P['NKY']['InSample_B'], 'Date'])
nk = cl[(cl['Symbol'] == 'NKY') & (cl['Date'].isin(inb))].copy()
nk['Year'] = nk['Date'].dt.year
t = nk.pivot_table(index='Year', columns='Class', values='Date',
                   aggfunc='count').fillna(0).astype(int)
t['Total'] = t.sum(axis=1)
t['Pct_Kept'] = (100 * (t['Total'] - t.get('DEFECT', 0)) / t['Total']).round(1)
print(t.to_string())

print()
print("=" * 80)
print("E. Does the NKY defect actually bias RV downward? (the test that matters)")
print("=" * 80)
p = P['NKY'].merge(cl[cl['Symbol'] == 'NKY'][['Date', 'Class', 'Coverage']],
                   on='Date', how='left')
p = p[p['InSample_B'] & p['RV_5min'].notna()].copy()
# benchmark RV against a measure that does NOT depend on intraday coverage: the squared
# close-to-close daily return, and the Parkinson high-low range estimator from the
# exchange daily file. If the defect days are biased low, their ratio to these
# coverage-independent benchmarks will be systematically below that of the FULL days.
p['ratio_r2'] = p['RV_5min'] / p['Return'].pow(2).replace(0, np.nan)
p['ratio_park'] = p['RV_5min'] / p['ParkinsonVar'].replace(0, np.nan)
g = p.groupby('Class').agg(n=('Date', 'count'),
                           med_RV=('RV_5min', 'median'),
                           med_ratio_vs_r2=('ratio_r2', 'median'),
                           med_ratio_vs_Parkinson=('ratio_park', 'median'))
print(g.round(4).to_string())
print()
print("  Same comparison for SPX as a control (its defects are few and inside sample B):")
ps = P['SPX'].merge(cl[cl['Symbol'] == 'SPX'][['Date', 'Class']], on='Date', how='left')
ps = ps[ps['InSample_B'] & ps['RV_5min'].notna()].copy()
ps['ratio_park'] = ps['RV_5min'] / ps['ParkinsonVar'].replace(0, np.nan)
print(ps.groupby('Class').agg(n=('Date', 'count'),
                              med_ratio_vs_Parkinson=('ratio_park', 'median')).round(4).to_string())
