# -*- coding: utf-8 -*-
"""
Phase 10: freeze the ESTIMATION / EVALUATION SAMPLE definition onto every panel.

WHY THIS EXISTS
  The six indices do not start on the same day. Daily prices go back to 1990 for all of
  them, but the realized measures and the regional volatility indices do not:

      layer            SPX        NDX        UKX        DAX        NKY        HSI
      daily            1990       1990       1990       1990       1990       1990
      intraday / RV    2011-09    2011-09    2011-09    2013-09    2011-09    2011-09
      own vol index    1990 VIX   2009 VXN   2008 VXEFA 2005 V1X   2018 NKVI  2011 VXEEM

  Two things bind: DAX intraday (the Dukascopy DEUIDXEUR history begins 2013-09-30) and the
  Nikkei VI (free history only from 2018-01-22). Three defensible samples follow, and the
  choice materially changes what the paper can claim, so it is frozen here in code rather
  than being re-decided inside each modelling notebook.

  SAMPLE_A  five indices, DAX dropped, from the 2011-09 intraday start.
            Longest window but NO euro-area index, and it strips out the only non-US market
            with a native model-free volatility index (VDAX-NEW). Kept as a robustness run.
  SAMPLE_B  ** PRIMARY ** all six, from the DAX intraday start, NKY using its DECLARED
            fallback volatility proxy VXEFA. Full regional coverage at the cost of the
            pre-2013 stretch.
  SAMPLE_C  all six, from the Nikkei VI start, every index on its OWN volatility index.
            Costs a third of the sample to remove one declared proxy. Robustness run only.

WHAT THIS WRITES BACK
  Three boolean columns per panel  (InSample_A / InSample_B / InSample_C)  plus
  CommonDate_A/B/C, which are TRUE only on dates where EVERY index in that sample has a
  complete observation. The InSample_* flags are windows; the CommonDate_* flags are the
  balanced-panel intersection used for Diebold-Mariano and Model Confidence Set work, where
  the loss series has to be aligned across indices.
  Also resolves VolUsed / VolUsed_Symbol - the volatility index each sample actually uses.

  Nothing is deleted. Rows outside every sample stay in the file: the daily-only models
  (GARCH-EVT, Quantile Regression) legitimately estimate on history back to 1990. Only the
  FORECAST EVALUATION window has to be common across models, not the estimation window.

Outputs: panel/<CODE>_panel_daily.csv (rewritten, flags added)
         _logs/phase10_sample_summary.csv
         _logs/phase10_common_dates_<S>.csv
"""
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAN = os.path.join(ROOT, '07_PANEL_INTERMEDIATE')
LOG = os.path.join(ROOT, '11_LOGS')
os.makedirs(LOG, exist_ok=True)

CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]

# sample -> which volatility column each index uses
#   'own'      -> VolIdx           (the regional vol index of that market)
#   'fallback' -> VolIdx_Fallback  (declared proxy)
SAMPLES = {
    "A": dict(codes=["SPX", "NDX", "UKX", "NKY", "HSI"],
              vol={"SPX": "own", "NDX": "own", "UKX": "own",
                   "NKY": "fallback", "HSI": "own"},
              note="5 indices, no euro area, longest window. Robustness."),
    "B": dict(codes=CODES,
              vol={"SPX": "own", "NDX": "own", "UKX": "own", "DAX": "own",
                   "NKY": "fallback", "HSI": "own"},
              note="PRIMARY. All six. NKY on the declared VXEFA proxy."),
    "C": dict(codes=CODES,
              vol={c: "own" for c in CODES},
              note="All six on their own vol index, from the Nikkei VI start. Robustness."),
}


def complete_mask(p, volmode):
    """A row is COMPLETE for a sample when all three modelled layers exist on that date."""
    volcol = 'VolIdx' if volmode == 'own' else 'VolIdx_Fallback'
    if volcol not in p.columns:
        volcol = 'VolIdx'
    return p['Return'].notna() & p['RV_5min'].notna() & p[volcol].notna(), volcol


def main():
    panels = {c: pd.read_csv(os.path.join(PAN, f'{c}_panel_daily.csv'), parse_dates=['Date'])
              for c in CODES}

    summary = []
    for s, cfg in SAMPLES.items():
        firsts = {}
        for c in cfg['codes']:
            m, _ = complete_mask(panels[c], cfg['vol'][c])
            if not m.any():
                raise SystemExit(f"sample {s}: {c} has no complete rows")
            firsts[c] = panels[c].loc[m, 'Date'].min()
        start = max(firsts.values())
        binding = [c for c, d in firsts.items() if d == start]
        end = min(panels[c]['Date'].max() for c in cfg['codes'])

        inter = None
        for c in cfg['codes']:
            m, _ = complete_mask(panels[c], cfg['vol'][c])
            w = m & (panels[c]['Date'] >= start) & (panels[c]['Date'] <= end)
            d = set(panels[c].loc[w, 'Date'])
            inter = d if inter is None else (inter & d)
        inter = pd.DatetimeIndex(sorted(inter))
        pd.DataFrame({'Date': inter}).to_csv(
            os.path.join(LOG, f'phase10_common_dates_{s}.csv'),
            index=False, date_format='%Y-%m-%d')

        for c in CODES:
            p = panels[c]
            if c in cfg['codes']:
                m, _ = complete_mask(p, cfg['vol'][c])
                p[f'InSample_{s}'] = m & (p['Date'] >= start) & (p['Date'] <= end)
                p[f'CommonDate_{s}'] = p['Date'].isin(inter)
            else:
                p[f'InSample_{s}'] = False
                p[f'CommonDate_{s}'] = False

        summary.append(dict(
            Sample=s, N_Indices=len(cfg['codes']), Indices=" ".join(cfg['codes']),
            Start=str(start.date()), End=str(end.date()),
            Binding_Index=" ".join(binding), Common_Days=len(inter),
            Approx_Years=round(len(inter) / 252.0, 2), Note=cfg['note']))

    rows = []
    for c in CODES:
        p = panels[c]
        volmode = SAMPLES['B']['vol'][c]
        volcol = 'VolIdx' if volmode == 'own' else 'VolIdx_Fallback'
        p['VolUsed'] = p[volcol]
        sym = p[volcol + '_Symbol'].dropna()
        p['VolUsed_Symbol'] = sym.iloc[0] if len(sym) else ''
        p['VolUsed_IsProxy'] = (volmode == 'fallback')
        p.to_csv(os.path.join(PAN, f'{c}_panel_daily.csv'), index=False,
                 date_format='%Y-%m-%d', float_format='%.10g')
        rows.append(dict(Code=c, Rows=len(p),
                         VolUsed_Symbol=p['VolUsed_Symbol'].iloc[0],
                         IsProxy=(volmode == 'fallback'),
                         InSample_A=int(p['InSample_A'].sum()),
                         InSample_B=int(p['InSample_B'].sum()),
                         InSample_C=int(p['InSample_C'].sum()),
                         CommonDate_B=int(p['CommonDate_B'].sum())))

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(LOG, 'phase10_sample_summary.csv'), index=False)
    print(sm.to_string(index=False))
    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
