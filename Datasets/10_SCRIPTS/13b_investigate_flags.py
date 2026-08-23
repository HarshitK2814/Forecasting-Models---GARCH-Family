# -*- coding: utf-8 -*-
"""Follow-up on every anomaly raised by the stage-1 audit. Diagnostic only, writes nothing."""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAN = os.path.join(ROOT, '07_PANEL_INTERMEDIATE')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]
EXPECTED = {"SPX": 78, "NDX": 78, "UKX": 102, "DAX": 102, "NKY": 60, "HSI": 66}

P = {c: pd.read_csv(os.path.join(PAN, f'{c}_panel_daily.csv'), parse_dates=['Date'])
     for c in CODES}

print("=" * 78)
print("1. CFD session return vs index daily return -- CORRECT comparison")
print("   (stage 1 wrongly subtracted the CFD overnight return from the INDEX return,")
print("    mixing two different instruments. Both series are close-to-close, so they")
print("    compare directly, subject to the script-08 alignment rule.)")
print("=" * 78)
for c in CODES:
    p = P[c]
    m = p[p['HasRV_t'] == True].dropna(subset=['Return', 'CFD_SessionReturn']).copy()
    # alignment: the CFD return spans the same one-day step as the index return only when
    # the previous RV day is also the previous index day
    m['prev_rv'] = m['Date'].shift(1)
    idx_prev = p.dropna(subset=['Return'])[['Date']].copy()
    idx_prev['prev_idx'] = idx_prev['Date'].shift(1)
    m = m.merge(idx_prev, on='Date', how='left')
    al = m[m['prev_rv'] == m['prev_idx']]
    r_raw = np.corrcoef(m['CFD_SessionReturn'], m['Return'])[0, 1]
    r_al = np.corrcoef(al['CFD_SessionReturn'], al['Return'])[0, 1]
    d = (al['CFD_SessionReturn'] - al['Return']).abs()
    print(f"  {c}  corr_naive={r_raw:.4f}  corr_aligned={r_al:.4f}  "
          f"n_aligned={len(al)}/{len(m)}  medAbsDiff={d.median()*1e4:.1f}bp  "
          f"p99={d.quantile(0.99)*1e4:.0f}bp")

print()
print("=" * 78)
print("2. NKY short sessions -- 217 days under 80% of the expected 60 bars")
print("=" * 78)
p = P['NKY']
nb = p[p['HasRV_t'] == True][['Date', 'NBars_5min']].copy()
nb['Year'] = nb['Date'].dt.year
t = nb.groupby('Year')['NBars_5min'].agg(['median', 'min', 'max', 'count'])
t['pct_under48'] = nb[nb['NBars_5min'] < 48].groupby('Year').size().reindex(t.index).fillna(0).astype(int)
print(t.to_string())
print()
print("  Tokyo Stock Exchange extended the cash close from 15:00 to 15:30 on 2024-11-05.")
print("  Bars per session before / after that date:")
pre = nb[nb['Date'] < '2024-11-05']['NBars_5min']
post = nb[nb['Date'] >= '2024-11-05']['NBars_5min']
print(f"    before  median={pre.median():.0f}  n={len(pre)}")
print(f"    after   median={post.median():.0f}  n={len(post)}")

print()
print("=" * 78)
print("3. Short sessions across all six -- are they real half-days?")
print("=" * 78)
for c in CODES:
    p = P[c]
    e = EXPECTED[c]
    s = p[(p['HasRV_t'] == True) & (p['NBars_5min'] < 0.8 * e)]
    if not len(s):
        continue
    md = s['Date'].dt.strftime('%m-%d').value_counts().head(6)
    print(f"  {c}: {len(s)} short days. Most frequent month-day:")
    print("     " + ", ".join(f"{k}x{v}" for k, v in md.items()))

print()
print("=" * 78)
print("4. Extreme returns, calendar gap, and the VolUsed / CreditStress missingness")
print("=" * 78)
for c in CODES:
    p = P[c]
    e = p[p['Return'].abs() > 0.15]
    for _, r in e.iterrows():
        print(f"  extreme  {c} {r['Date'].date()}  ret={r['Return']*100:+.2f}%  "
              f"close={r['Close']:.2f}")
    g = p['Date'].diff().dt.days
    for _, r in p[g > 7].iterrows():
        print(f"  gap      {c} {r['Date'].date()}  {int(g.loc[r.name])} days since previous")
print()
for c in CODES:
    p = P[c]
    b = p[p['InSample_B']]
    print(f"  {c}  VolUsed missing in B = {int(b['VolUsed'].isna().sum())}"
          f"   CreditStress missing in B = {int(b['CreditStress'].isna().sum())}"
          f"   ({100*b['CreditStress'].isna().mean():.1f}%)")

print()
print("=" * 78)
print("5. Macro missingness inside sample B, by column")
print("=" * 78)
mac = ['US10Y_pct', 'US13W_pct', 'DXY', 'WTI_usd', 'GOLD_usd', 'HYG_px', 'IEF_px',
       'TermSpread_pct', 'CreditStress']
rows = []
for c in CODES:
    b = P[c][P[c]['InSample_B']]
    rows.append({'Code': c, **{m: round(100 * b[m].isna().mean(), 2) for m in mac}})
print(pd.DataFrame(rows).to_string(index=False))
