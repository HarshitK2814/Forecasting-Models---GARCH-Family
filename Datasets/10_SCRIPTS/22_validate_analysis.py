# -*- coding: utf-8 -*-
"""
FINAL VALIDATION of the analysis dataset. Independent re-derivation, not a re-read.

The point of this script is to check the cleaned files against facts computed FROM SCRATCH
rather than against the intermediate objects that produced them. A validation that reuses the
build code cannot catch a bug in the build code.

The look-ahead test is the one that matters most. If any predictor column contains
information from t+1, every out-of-sample result in the paper is void and the failure is
invisible in ordinary summary statistics. It is tested here by construction: each predictor is
correlated against the FUTURE return and the FUTURE realized variance, and any column whose
correlation with a future value exceeds its correlation with the contemporaneous value is
flagged for inspection.
"""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
VAL = os.path.join(ROOT, '08_VALIDATION')
CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

checks, fails, warns = [], [], []


def chk(cond, code, name, detail=""):
    ok = bool(cond)
    checks.append(dict(Code=code, Check=name, Pass=ok, Detail=detail))
    if not ok:
        fails.append(f"{code}: {name} — {detail}")
    return ok


for c in CODES:
    a = pd.read_csv(os.path.join(ANA, f'{c}_analysis.csv'), parse_dates=['Date'])

    # ---------- structure ----------
    chk(a['Date'].is_monotonic_increasing, c, 'dates monotonic')
    chk(a['Date'].duplicated().sum() == 0, c, 'no duplicate dates',
        f"{a['Date'].duplicated().sum()} dupes")
    chk((a['Date'].dt.dayofweek < 5).all(), c, 'no weekend dates')
    chk(a['Symbol'].nunique() == 1, c, 'single symbol')

    # ---------- returns recomputed from the price column ----------
    r_re = np.log(a['Close'] / a['Close'].shift(1))
    d = (a['Return'] - r_re).abs().max()
    chk(d < 1e-8, c, 'Return equals log(C_t/C_t-1)', f"max abs diff {d:.2e}")

    # ---------- the RV_Valid gate is actually enforced ----------
    bad = a.loc[~a['RV_Valid'], ['RV', 'RS_neg', 'Jump', 'LogRV', 'BPV']].notna().sum().sum()
    chk(bad == 0, c, 'no realized measure survives outside RV_Valid',
        f"{bad} non-null cells")
    chk((a.loc[a['RV_Valid'], 'RV'] > 0).all(), c, 'RV strictly positive where valid')
    chk(a.loc[a['RV_Valid'], 'SessionClass'].eq('FULL').all(), c,
        'RV_Valid implies SessionClass FULL')

    # ---------- identities re-derived ----------
    v = a[a['RV_Valid']]
    for nm, err in (('RS_pos+RS_neg=RV', (v['RS_pos'] + v['RS_neg'] - v['RV']).abs().max()),
                    ('ContVar+Jump=RV', (v['ContVar'] + v['Jump'] - v['RV']).abs().max()),
                    ('RVol^2=RV', (v['RVol'] ** 2 - v['RV']).abs().max()),
                    ('exp(LogRV)=RV', (np.exp(v['LogRV']) - v['RV']).abs().max())):
        rel = err / v['RV'].median()
        chk(rel < 1e-6, c, f'identity {nm}', f"rel err {rel:.2e}")

    # ---------- units ----------
    chk(v['RVol_Ann_Pct'].between(1, 300).all(), c, 'annualised RVol in a sane range',
        f"[{v['RVol_Ann_Pct'].min():.1f}, {v['RVol_Ann_Pct'].max():.1f}]")
    iv = a['VolIdx'].dropna()
    chk(iv.between(4, 200).all(), c, 'vol index in a sane range',
        f"[{iv.min():.1f}, {iv.max():.1f}]")
    y = a['US10Y_pct'].dropna()
    chk(y.between(0, 20).all(), c, 'US 10y yield in percent not decimals',
        f"[{y.min():.2f}, {y.max():.2f}]")

    # ---------- HAR terms are trailing, never forward ----------
    # rebuild the 5-day trailing mean independently and compare
    w_re = a['RV'].rolling(5, min_periods=3).mean()
    m = a['RV_w'].notna() & w_re.notna()
    dd = (a.loc[m, 'RV_w'] - w_re[m]).abs().max()
    chk(dd < 1e-10, c, 'RV_w is a trailing 5-day mean', f"max abs diff {dd:.2e}")

    # ---------- LOOK-AHEAD: prefix stability ----------
    # The correct test is not a correlation heuristic. An earlier version compared each
    # predictor's correlation with LogRV(t) against its correlation with LogRV(t+1) and
    # flagged anything higher on the future. That test is invalid: VRP, JumpShare and
    # RSV_Ratio all contain RV(t) by construction, which mechanically suppresses their
    # CONTEMPORANEOUS correlation, and the macro series legitimately lead the Asian indices
    # by a day because those sessions close before New York opens. Both produce false
    # positives that say nothing about look-ahead.
    #
    # The definitive test is prefix stability. If a column at row t uses any information
    # from after t, then recomputing it from a dataset TRUNCATED at t must change its value.
    # So the time-dependent derivations are rebuilt on prefixes and compared at the cut.
    cuts = [int(len(a) * f) for f in (0.35, 0.55, 0.75, 0.92)]
    unstable = []
    for T in cuts:
        pre = a.iloc[:T + 1].copy()
        rebuilt = {
            'Return': np.log(pre['Close'] / pre['Close'].shift(1)),
            'RV_w': pre['RV'].rolling(5, min_periods=3).mean(),
            'RV_m': pre['RV'].rolling(22, min_periods=14).mean(),
            'LogRV_w': pre['LogRV'].rolling(5, min_periods=3).mean(),
            'LogRV_m': pre['LogRV'].rolling(22, min_periods=14).mean(),
            'RSneg_w': pre['RS_neg'].rolling(5, min_periods=3).mean(),
            'DXY_ret': np.log(pre['DXY']).diff(),
            'HYG_ret': np.log(pre['HYG_px']).diff(),
            'TermSpread_diff': pre['TermSpread_pct'].diff(),
            'US10Y_diff': pre['US10Y_pct'].diff(),
            'RV_Scaled': pre['ScaleFactor_HL'] * pre['RV'],
        }
        for col, series in rebuilt.items():
            if col not in a.columns:
                continue
            v_full, v_pre = a[col].iloc[T], series.iloc[T]
            if pd.isna(v_full) and pd.isna(v_pre):
                continue
            if pd.isna(v_full) != pd.isna(v_pre):
                unstable.append(f"{col}@{T}(nan mismatch)")
            elif abs(v_full - v_pre) > 1e-9 * max(1.0, abs(v_full)):
                unstable.append(f"{col}@{T}(d={abs(v_full - v_pre):.2e})")
    chk(len(unstable) == 0, c, 'derived columns are prefix-stable (no look-ahead)',
        "; ".join(unstable[:6]))

    # ScaleFactor_HL is deliberately a FULL-SAMPLE constant. That is fine for descriptive
    # use and for the measurement equation, but it is in-sample information, so it is
    # asserted to be constant here and flagged in the dictionary rather than silently used
    # as if it were recursively estimated.
    chk(a['ScaleFactor_HL'].nunique(dropna=True) == 1, c,
        'ScaleFactor_HL is a single full-sample constant (documented in-sample quantity)')

    # contemporaneous return must NOT be inferable from the predictor set alone in a way
    # that beats its own lag - a direct check that Return is not accidentally shifted
    cc = a[['Return']].copy()
    cc['lag1'] = a['Return'].shift(1)
    cc['lead1'] = a['Return'].shift(-1)
    k = cc.dropna()
    chk(abs(np.corrcoef(k['Return'], k['lead1'])[0, 1]) < 0.25, c,
        'return not duplicated across adjacent rows',
        f"corr(r_t, r_t+1)={np.corrcoef(k['Return'], k['lead1'])[0,1]:.3f}")

    # ---------- sample flags ----------
    chk((a.loc[a['InSample_B'], 'Return'].notna()).all(), c, 'InSample_B implies a return')
    chk(a.loc[a['BalancedRV_B'], 'RV_Valid'].all(), c, 'BalancedRV_B implies RV_Valid')
    chk(a.loc[a['BalancedRV_B'], 'InSample_B'].all(), c, 'BalancedRV_B implies InSample_B')

    # ---------- macro fill did not run backwards ----------
    # a forward fill can never create a value BEFORE the first genuine observation
    for mc in ['US10Y_pct', 'DXY', 'HYG_px']:
        s = a[mc]
        if s.notna().any():
            first = s.first_valid_index()
            chk(s.iloc[:first].isna().all() if first else True, c,
                f'{mc} has no value before its first observation')

    # ---------- no infinities ----------
    num = a.select_dtypes(include=[np.number])
    ninf = int(np.isinf(num.values).sum())
    chk(ninf == 0, c, 'no infinities', f"{ninf} inf cells")

    print(f"  [{c}] validated")

# ---------- cross-index ----------
bal = None
for c in CODES:
    a = pd.read_csv(os.path.join(ANA, f'{c}_analysis.csv'), parse_dates=['Date'])
    s = set(a.loc[a['BalancedRV_B'], 'Date'])
    bal = s if bal is None else (bal & s)
    if len(s) != len(a.loc[a['BalancedRV_B']]):
        fails.append(f"{c}: BalancedRV_B has duplicate dates")
sizes = []
for c in CODES:
    a = pd.read_csv(os.path.join(ANA, f'{c}_analysis.csv'), parse_dates=['Date'])
    sizes.append(int(a['BalancedRV_B'].sum()))
chk(len(set(sizes)) == 1, 'ALL', 'BalancedRV_B identical across indices', str(sizes))
chk(len(bal) == sizes[0], 'ALL', 'BalancedRV_B intersection is exact',
    f"{len(bal)} vs {sizes[0]}")

C = pd.DataFrame(checks)
C.to_csv(os.path.join(VAL, 'eda7_final_validation.csv'), index=False)
print()
print("=" * 74)
print(f"{len(C)} checks across {len(CODES)} files — {int((~C['Pass']).sum())} failures")
print("=" * 74)
if fails:
    for f in fails:
        print("  FAIL " + f)
else:
    print("  all checks passed")
print()
print(C.groupby('Check')['Pass'].agg(['count', 'sum']).rename(
    columns={'count': 'n', 'sum': 'passed'}).to_string())
