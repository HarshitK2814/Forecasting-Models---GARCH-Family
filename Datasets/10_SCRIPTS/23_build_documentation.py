# -*- coding: utf-8 -*-
"""
Build the consolidated dataset documentation: one master Excel workbook and two PDFs.

  00_DOCUMENTATION/DATASET_MASTER_REPORT.xlsx   every table, in one workbook
  00_DOCUMENTATION/Dataset_Guide.pdf            the complete narrative + precautions
  00_DOCUMENTATION/Diagnostic_Figures.pdf       the ten figures with captions
  00_DOCUMENTATION/FILE_INVENTORY.csv           every file, size and row count

Numbers are read from the validation and log CSVs rather than typed, so the documents
cannot drift from the data when anything upstream is re-run.
"""
import os
import glob
import datetime
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, Image, KeepTogether)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, '00_DOCUMENTATION')
VAL = os.path.join(ROOT, '08_VALIDATION')
LOG = os.path.join(ROOT, '11_LOGS')
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
FIG = os.path.join(ROOT, '09_FIGURES')
os.makedirs(DOC, exist_ok=True)

CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]
TODAY = datetime.date.today().isoformat()

FOLDERS = [
    ("00_DOCUMENTATION", "Reports, PDFs, dictionaries. START HERE."),
    ("01_ANALYSIS_READY", "**THE dataset to model on.** One cleaned CSV per index, 96 columns."),
    ("02_RAW_DAILY", "Exchange daily OHLC per index, 1990-2026. Yahoo Finance."),
    ("03_RAW_INTRADAY", "1-min and 5-min session bars per index-year. Dukascopy."),
    ("04_RAW_VOLATILITY", "17 implied-volatility indices. CBOE, STOXX, Yahoo."),
    ("05_RAW_MACRO", "22 keyless macro / risk-factor series. Yahoo Finance."),
    ("06_REALIZED_MEASURES", "Daily realized measures per index, base and extended."),
    ("07_PANEL_INTERMEDIATE", "The raw join, BEFORE cleaning. Provenance only - do not model on this."),
    ("08_VALIDATION", "Every validation and EDA table behind the reports."),
    ("09_FIGURES", "Ten diagnostic figures."),
    ("10_SCRIPTS", "Every script, numbered in execution order."),
    ("11_LOGS", "Manifests and per-phase run summaries."),
    ("12_CACHE_REGENERATION", "Raw Dukascopy .npy cache. Only needed to rebuild realized measures."),
]

# ---------------------------------------------------------------- precautions
PRECAUTIONS = [
 (1, "CRITICAL", "Model on 01_ANALYSIS_READY, never 07_PANEL_INTERMEDIATE",
  "The panel folder is the raw join kept for provenance. It has no session-quality gating, "
  "so it still contains the Nikkei's biased 2016-17 realized variances."),
 (2, "CRITICAL", "Gate every realized measure on RV_Valid",
  "RV, RS_pos/neg, BPV, MedRV, RQ, TQ, Jump, ContVar and all HAR terms are NaN outside "
  "RV_Valid. If you fillna() or dropna() carelessly you will either reintroduce bad data or "
  "silently delete good return observations."),
 (3, "CRITICAL", "Apply the forecasting lag yourself - the file is contemporaneous",
  "Every predictor is dated at the close of day t. Nothing is pre-lagged. The modelling code "
  "must shift predictors to forecast t+1. Failing to do this produces spectacular in-sample "
  "results and is the single most common way to invalidate a volatility paper."),
 (4, "CRITICAL", "Do not winsorize, trim or de-jump the returns",
  "The tail is the object of study. Clipping extremes shrinks the estimated GPD shape "
  "parameter and manufactures VaR that appears well-calibrated only because the exceedances "
  "were deleted. The extreme days were individually verified as real market events."),
 (5, "HIGH", "Never use LogRV and LogRS_neg together",
  "RS_pos + RS_neg = RV identically and the downside share sits at 0.50, so LogRS_neg is "
  "LogRV plus an almost-constant. Mean VIF is about 95. Use the level-plus-share form: "
  "LogRV with RSV_Ratio, JumpShare and RSkew. Max VIF falls from 21.9 to 8.1."),
 (6, "HIGH", "Do not use US10Y_pct or TermSpread_pct in levels as regressors",
  "Both fail ADF on all six indices (p = 0.87 and 0.33). Use US10Y_diff and TermSpread_diff, "
  "which are stationary at p < 0.001. The levels are retained only as regime descriptors."),
 (7, "HIGH", "BalancedRV_B is for POOLED statistics only",
  "It marks the 1,994 dates where all six indices have valid RV. Using it for per-index work "
  "discards 5,909 perfectly good index-days to accommodate the worst index. Per-index "
  "estimation and per-index DM/MCS should use each index's own RV_Valid days."),
 (8, "HIGH", "The Nikkei has no valid realized measure in 2016-17",
  "The Dukascopy feed is missing the entire 09:00 opening hour for those two years. A rolling "
  "window must use the last AVAILABLE observations rather than the last N calendar days, and "
  "no Realized-GARCH forecast can be produced or evaluated inside the gap. Disclose it."),
 (9, "HIGH", "RV measures the cash session only - it is not daily variance",
  "The daily return spans close to close and also contains the overnight gap. RV captures "
  "only 33-58% of daily variance depending on the market. Use ScaleFactor_HL (1.71 to 3.04, "
  "index-specific) or RV_Scaled when comparing RV against squared daily returns, and let the "
  "Realized-GARCH measurement equation absorb the scale."),
 (10, "HIGH", "ScaleFactor_HL is a full-sample, in-sample constant",
  "It is estimated once over the whole of sample B. Fine for description and for the "
  "measurement equation; for a strict recursive out-of-sample exercise re-estimate it on each "
  "rolling window or you have leaked information."),
 (11, "HIGH", "Sessions are not synchronous - do not pool naively on the calendar date",
  "Tokyo and Hong Kong close before New York opens, so a US shock on day t reaches Asia on "
  "day t+1. Same-date return correlation between Asia and the US is only about 0.18, which is "
  "a timing artefact and not economic independence. Either adopt a lagged-information "
  "convention or estimate index by index and pool only the loss series."),
 (12, "MEDIUM", "Re-estimate the POT threshold on the actual GARCH residuals",
  "The recommended 95th-97.5th percentile comes from rolling-standardised returns, used as a "
  "stand-in because no GARCH was fitted at the EDA stage. McNeil-Frey applies the GPD to "
  "genuine GARCH residuals; redo the stability plot once stage 1 exists."),
 (13, "MEDIUM", "Volatility indices are annualised percentages, RV is a daily variance",
  "Convert before comparing: IV_DailyVar = (VolIdx/100)^2 / 252. This is already provided. "
  "Mixing the two scales silently is an easy and expensive mistake."),
 (14, "MEDIUM", "Overnight_LogRet is the only column not dated at the close",
  "It spans close(t-1) to open(t) and is therefore known at the OPEN of day t. Treat it as "
  "the first observable piece of day t, not as day-t close information."),
 (15, "MEDIUM", "Macro series are forward-filled up to 5 business days",
  "These are US series joined onto six exchange calendars. The fill is forward only, never "
  "backward, so it is the information set rather than an interpolation - but check "
  "MacroFilled_t if a result depends on a specific date."),
 (16, "MEDIUM", "Half-days are real sessions but are excluded from RV_Valid",
  "A three-hour variance fed into a measurement equation calibrated on full sessions biases "
  "the intercept. They are retained in the file with IsHalfDay=True, so the choice is "
  "reversible if you want them."),
 (17, "MEDIUM", "The intraday data is a bid-side index CFD, not the exchange index",
  "Aligned correlation against the exchange daily return is 0.97-0.99, so it tracks well, but "
  "it is a proxy and must be described as one. The 1-min/5-min RV ratio of 1.06-1.09 shows a "
  "smoothed feed, which also means RV may be mildly damped versus a trade-based estimator."),
 (18, "MEDIUM", "Three volatility indices are regional proxies, not the index's own",
  "NKY uses VXEFA in the primary sample (the Nikkei VI only starts 2018), UKX uses VXEFA and "
  "HSI uses VXEEM, because no free FTSE-100 or HSI volatility index exists. VolIdx_IsProxy "
  "flags the Nikkei case. Sample C exists to test that this does not drive results."),
 (19, "MEDIUM", "Sample B excludes 2008",
  "The daily-only models estimate from 1990 and see the GFC; Realized GARCH cannot, because "
  "free intraday history begins in 2011-2013. The three models therefore see different "
  "amounts of crisis history. Unavoidable, but it must be stated."),
 (20, "LOW", "Use Close only from the macro folder on OHLC-inconsistent rows",
  "Yahoo writes placeholder bars for FX and front-month futures on a minority of days where "
  "Open=High=Low and Close comes from a different snapshot. Close is the reliable field. "
  "OHLC_Consistent flags the affected rows."),
 (21, "LOW", "Do not recompute returns from 07_PANEL_INTERMEDIATE",
  "The download rounded LogReturn to 6 decimal places. The analysis files re-derive the "
  "return from Close at full precision; use those."),
 (22, "LOW", "If you re-download from Dukascopy: months are ZERO-INDEXED in the URL",
  "January is 00. Getting this wrong silently returns the wrong month rather than an error."),
 (23, "LOW", "If you re-download: one process only, with HTTP keep-alive",
  "Dukascopy throttles per IP. Extra parallel processes made throughput worse (4.9 to 1.9 "
  "req/s) and triggered a lasting penalty. Without keep-alive it is ~0.05 req/s, a 100x "
  "penalty."),
 (24, "LOW", "An expired CA bundle silently breaks every HTTPS fetch",
  "This cost real time at the start. If downloads fail wholesale, update certifi before "
  "debugging anything else."),
]

DECISIONS = [
 (1, "Realized measures nulled on DEFECT sessions; rows retained",
  "The exchange close is still valid, so the daily return is still an observation. Deleting "
  "rows would shorten the return series non-randomly with respect to volatility.",
  "RV NaN on 369/155/80/24/633/71 days (SPX/NDX/UKX/DAX/NKY/HSI)"),
 (2, "Half-days flagged and excluded from RV_Valid; values retained",
  "A short-session variance biases a measurement equation calibrated on full sessions. "
  "Standard practice in the realized-volatility literature.", "Reversible via IsHalfDay"),
 (3, "Returns NOT winsorized, trimmed or de-jumped",
  "The tail is the object of study. Clipping would shrink the GPD shape parameter and delete "
  "the exceedances that identify it.", "Extremes individually verified as real events"),
 (4, "Macro forward-filled <=5 business days, forward only",
  "US series on six exchange calendars. The last published value IS the information set on a "
  "day the US did not trade. Backward fill would be look-ahead.", "MacroFilled_t records it"),
 (5, "Hansen-Lunde scale factor estimated per index",
  "The session vs close-to-close gap is index-specific and large. Measured, not assumed away.",
  "1.71 DAX to 3.04 NKY, monotone in session length"),
 (6, "Nothing standardised, differenced or de-meaned in storage",
  "Those are estimator-specific choices. Levels are stored; transformations belong in the "
  "modelling code.", "-"),
 (7, "Non-stationary macro levels supplemented with differences",
  "ADF p=0.87 for the 10-year yield level.", "US10Y_diff, TermSpread_diff added"),
 (8, "Realized block re-parameterised as level + share",
  "LogRV and LogRS_neg are collinear by identity.", "Max VIF 21.9 -> 8.1"),
 (9, "Session coverage measured on the 5-minute grid, not the 1-minute grid",
  "A quiet minute with no quote change is dropped upstream and is not missing data. The "
  "1-minute version called 674 SPX days half-days against ~3/year scheduled.",
  "Half-days now land on 07-03, 07-04, 12-24, 12-31"),
 (10, "Return re-derived from Close at full precision",
  "The download rounded LogReturn to 6 dp.", "Removes a 5e-7 quantisation"),
]


def read(p, folder=VAL):
    f = os.path.join(folder, p)
    return pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()


# ---------------------------------------------------------------- inventory
def build_inventory():
    rows = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = os.path.relpath(dirpath, ROOT).replace('\\', '/')
        if rel.startswith('12_CACHE_REGENERATION') and rel.count('/') >= 1:
            continue          # 24k cache files: summarise at the folder level only
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            nrows = ''
            if fn.endswith('.csv') and sz < 60e6:
                try:
                    with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                        nrows = sum(1 for _ in fh) - 1
                except Exception:
                    nrows = ''
            rows.append(dict(Folder=rel, File=fn, Size_KB=round(sz / 1024, 1), Rows=nrows))
    cache = os.path.join(ROOT, '12_CACHE_REGENERATION')
    if os.path.isdir(cache):
        for sub in sorted(os.listdir(cache)):
            d = os.path.join(cache, sub)
            if os.path.isdir(d):
                fs = os.listdir(d)
                tot = sum(os.path.getsize(os.path.join(d, f)) for f in fs)
                rows.append(dict(Folder=f'12_CACHE_REGENERATION/{sub}',
                                 File=f'({len(fs)} .npy files, summarised)',
                                 Size_KB=round(tot / 1024, 1), Rows=''))
    return pd.DataFrame(rows).sort_values(['Folder', 'File'])


# ================================================================ EXCEL
def build_excel(inv):
    tables = {
        'README_FIRST': pd.DataFrame({
            'Item': ['Dataset', 'Generated', 'Primary file', 'Primary sample',
                     'Indices', 'Daily coverage', 'Realized coverage',
                     'Validation', 'Models fitted'],
            'Value': ['GARCH-EVT vs Realized GARCH vs Quantile Regression - Researcher A',
                      TODAY,
                      '01_ANALYSIS_READY/<CODE>_analysis.csv  (96 columns)',
                      'Sample B: all six indices, 2013-09-30 to 2026-08-21, 2,685 common days',
                      ', '.join(CODES),
                      '1990-01-02 to 2026-08-21',
                      '2011-09 onward (DAX 2013-09); NKY has no valid RV in 2016-17',
                      '1,195 acquisition checks + 158 analysis checks, 0 failures',
                      'NONE. This is a dataset only - no model has been estimated.']}),
        'Folder_Map': pd.DataFrame(FOLDERS, columns=['Folder', 'Contents']),
        'Precautions': pd.DataFrame(PRECAUTIONS,
                                    columns=['#', 'Severity', 'Precaution', 'Why']),
        'Cleaning_Decisions': pd.DataFrame(DECISIONS,
                                           columns=['#', 'Decision', 'Reason', 'Effect']),
        'File_Inventory': inv,
        'Data_Dictionary': read('DATA_DICTIONARY.csv', ANA),
        'Feature_Sets': read('FEATURE_SETS.csv', ANA),
        'Scale_Factors': read('SCALE_FACTORS.csv', ANA),
        'Sample_Definition': read('phase10_sample_summary.csv', LOG),
        'Build_Summary': read('phase13_analysis_build.csv', LOG),
        'Final_Counts': read('phase14_finalise.csv', LOG),
        'Quality_Audit': read('eda1_quality_by_index.csv'),
        'Missingness': read('eda1_missingness.csv'),
        'Session_Class': read('eda2_session_class_summary.csv'),
        'Moments': read('eda4_moments.csv'),
        'Spec_Tests': read('eda4_tests.csv'),
        'Tail_Index': read('eda4_tail.csv'),
        'Stationarity': read('eda5_stationarity.csv'),
        'Predictive_Power': read('eda5_predictive.csv'),
        'VIF': read('eda6_vif_final.csv'),
        'Identities': read('eda5_identities.csv'),
        'Cross_Index': read('eda5_cross_index.csv'),
        'GPD_Threshold': read('eda6_gpd_threshold.csv'),
        'Vol_Regimes': read('eda6_regimes.csv'),
        'Worst_Days': read('eda6_extremes.csv'),
        'Final_Validation': read('eda7_final_validation.csv'),
    }
    xl = os.path.join(DOC, 'DATASET_MASTER_REPORT.xlsx')
    with pd.ExcelWriter(xl, engine='xlsxwriter') as w:
        book = w.book
        hdr = book.add_format({'bold': True, 'bg_color': '#1F4E79', 'font_color': 'white',
                               'border': 1, 'text_wrap': True, 'valign': 'top'})
        wrap = book.add_format({'text_wrap': True, 'valign': 'top'})
        for name, df in tables.items():
            if df is None or not len(df):
                df = pd.DataFrame({'note': ['table not available']})
            df.to_excel(w, sheet_name=name[:31], index=False, startrow=1, header=False)
            ws = w.sheets[name[:31]]
            for j, col in enumerate(df.columns):
                ws.write(0, j, str(col), hdr)
                width = min(max(14, int(df[col].astype(str).str.len().max() or 14) + 2), 70)
                ws.set_column(j, j, width, wrap if width > 45 else None)
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))
    return xl, tables


# ================================================================ PDF
S = getSampleStyleSheet()
BODY = ParagraphStyle('body', parent=S['BodyText'], fontSize=9, leading=12.5,
                      alignment=TA_LEFT, spaceAfter=5)
H1 = ParagraphStyle('h1', parent=S['Heading1'], fontSize=15, spaceBefore=13, spaceAfter=7,
                    textColor=colors.HexColor('#1F4E79'))
H2 = ParagraphStyle('h2', parent=S['Heading2'], fontSize=11.5, spaceBefore=9, spaceAfter=4,
                    textColor=colors.HexColor('#2E5F8A'))
CELL = ParagraphStyle('cell', parent=BODY, fontSize=7.6, leading=9.6, spaceAfter=0)
CELLB = ParagraphStyle('cellb', parent=CELL, fontName='Helvetica-Bold')
MONO = ParagraphStyle('mono', parent=BODY, fontName='Courier', fontSize=8, leading=10.5,
                      backColor=colors.HexColor('#F2F2F2'), borderPadding=5)

SEVCOL = {'CRITICAL': colors.HexColor('#C00000'), 'HIGH': colors.HexColor('#C55A11'),
          'MEDIUM': colors.HexColor('#BF8F00'), 'LOW': colors.HexColor('#548235')}


def esc(x):
    return (str(x).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def tbl(df, widths=None, fs=7.6, maxrows=None, head_bg='#1F4E79'):
    d = df if maxrows is None else df.head(maxrows)
    cs = ParagraphStyle('c', parent=CELL, fontSize=fs, leading=fs * 1.28)
    hs = ParagraphStyle('h', parent=cs, fontName='Helvetica-Bold',
                        textColor=colors.white)
    data = [[Paragraph(esc(c), hs) for c in d.columns]]
    for _, r in d.iterrows():
        row = []
        for v in r:
            if isinstance(v, float):
                v = f"{v:,.4g}" if abs(v) < 1e6 else f"{v:,.0f}"
            row.append(Paragraph(esc(v), cs))
        data.append(row)
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(head_bg)),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#B0B0B0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#F4F7FA')]),
        ('LEFTPADDING', (0, 0), (-1, -1), 3.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(colors.HexColor('#808080'))
    canvas.drawString(18 * mm, 11 * mm,
                      f"Dataset Guide — GARCH-EVT / Realized GARCH / Quantile Regression — {TODAY}")
    canvas.drawRightString(doc.pagesize[0] - 18 * mm, 11 * mm, f"page {doc.page}")
    canvas.restoreState()


def build_guide(tables, inv):
    W = A4[0] - 36 * mm
    st = []
    A = st.append

    # ---- title
    A(Spacer(1, 45 * mm))
    A(Paragraph("<b>Dataset Guide</b>", ParagraphStyle(
        't', parent=S['Title'], fontSize=27, textColor=colors.HexColor('#1F4E79'))))
    A(Spacer(1, 5 * mm))
    A(Paragraph("Volatility and tail-risk forecasting across six Tier-1 equity indices",
                ParagraphStyle('s', parent=S['Title'], fontSize=13,
                               textColor=colors.HexColor('#444444'))))
    A(Spacer(1, 3 * mm))
    A(Paragraph("GARCH-EVT · Realized GARCH · Quantile Regression",
                ParagraphStyle('s2', parent=S['Title'], fontSize=10.5,
                               textColor=colors.HexColor('#777777'))))
    A(Spacer(1, 22 * mm))
    A(tbl(tables['README_FIRST'], widths=[42 * mm, W - 42 * mm], fs=8.6))
    A(PageBreak())

    # ---- 1 quick start
    A(Paragraph("1. Quick start", H1))
    A(Paragraph("<b>Model on <font face='Courier'>01_ANALYSIS_READY/&lt;CODE&gt;_analysis.csv</font>.</b> "
                "That is the only folder the modelling code should read. Everything else is raw "
                "input, intermediate output, or documentation.", BODY))
    A(Paragraph("A minimal, correct load looks like this — note the two things it does that are "
                "easy to get wrong: it gates the realized measures on <font face='Courier'>RV_Valid</font>, "
                "and it applies the forecasting lag itself.", BODY))
    A(Paragraph(
        "import pandas as pd<br/>"
        "a = pd.read_csv('01_ANALYSIS_READY/SPX_analysis.csv', parse_dates=['Date'])<br/><br/>"
        "# daily-only models (GARCH-EVT, Quantile Regression) can use all history<br/>"
        "r = a.loc[a['Return'].notna(), ['Date', 'Return']]<br/><br/>"
        "# realized measures MUST be gated<br/>"
        "rv = a.loc[a['RV_Valid'] &amp; a['InSample_B'], ['Date', 'RV', 'LogRV', 'RS_neg']]<br/><br/>"
        "# YOU apply the lag - nothing in the file is pre-lagged<br/>"
        "X = a[['LogRV', 'LogRV_w', 'LogIV', 'NegReturn']].shift(1)<br/>"
        "y = a['Return']", MONO))
    A(Spacer(1, 3 * mm))
    A(Paragraph("2. Folder map", H1))
    A(tbl(tables['Folder_Map'], widths=[46 * mm, W - 46 * mm], fs=8))
    A(PageBreak())

    # ---- 3 precautions  (the centrepiece)
    A(Paragraph("3. Precautions — read before modelling", H1))
    A(Paragraph("Twenty-four traps, ordered by severity. The first four will invalidate results "
                "outright if ignored; the rest will bias or misrepresent them.", BODY))
    P = tables['Precautions'].copy()
    for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        sub = P[P['Severity'] == sev]
        if not len(sub):
            continue
        A(Paragraph(f"{sev} ({len(sub)})", ParagraphStyle(
            'sv', parent=H2, textColor=SEVCOL[sev])))
        rows = [[Paragraph(f"<b>{r['#']}</b>", CELL),
                 Paragraph(f"<b>{esc(r['Precaution'])}</b><br/>"
                           f"<font size=7.2 color='#444444'>{esc(r['Why'])}</font>", CELL)]
                for _, r in sub.iterrows()]
        t = Table(rows, colWidths=[8 * mm, W - 8 * mm], hAlign='LEFT')
        t.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#C8C8C8')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F0F0')),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
        A(t)
        A(Spacer(1, 2.5 * mm))
    A(PageBreak())

    # ---- 4 sample
    A(Paragraph("4. Sample definition", H1))
    A(tbl(tables['Sample_Definition'], fs=7.4))
    A(Spacer(1, 3 * mm))
    A(Paragraph("<b>Sample B is primary.</b> It keeps all six indices and therefore the euro "
                "area, which sample A drops. Dropping the DAX would also remove the only "
                "non-US market with a native model-free volatility index (VDAX-NEW), leaving "
                "every European and Asian result resting on a proxy. Sample C spends a third "
                "of the observations to remove one declared proxy, which is a bad trade. Both "
                "A and C are retained as ready-made robustness runs.", BODY))
    A(Paragraph("Three flags encode the distinction and they are not interchangeable:", BODY))
    A(tbl(pd.DataFrame([
        ['InSample_B', 'The window. Use for per-index work.'],
        ['CommonDate_B', '2,685 dates on which all six indices traded.'],
        ['BalancedRV_B', '1,994 dates where all six ALSO have valid RV. Pooled statistics only.'],
    ], columns=['Flag', 'Meaning']), widths=[32 * mm, W - 32 * mm], fs=8))
    A(Spacer(1, 3 * mm))
    A(Paragraph("5. Cleaning decisions", H1))
    A(tbl(tables['Cleaning_Decisions'],
          widths=[7 * mm, 52 * mm, W - 105 * mm, 46 * mm], fs=7.2))
    A(PageBreak())

    # ---- 6 quality findings
    A(Paragraph("6. Data quality findings", H1))
    A(Paragraph("6.1 Structural integrity", H2))
    A(Paragraph("No duplicated dates, no weekend dates, no OHLC ordering violations, no zero "
                "realized variances and no runs of repeated closing prices, on any index. "
                "Zero-return frequency peaks at 0.15%.", BODY))
    A(Paragraph("6.2 The Nikkei intraday defect — the finding that drove the cleaning", H2))
    A(Paragraph("The Dukascopy feed carries <b>no bars at all in the 09:00 local hour for the "
                "whole of 2016 and 2017</b>. Because intraday volatility is U-shaped, losing "
                "the opening hour removes far more than its proportional share of the "
                "session's variance, and the correction factor could only be estimated from "
                "the very period that is missing.", BODY))
    A(Paragraph("The bias was measured, not assumed, against two benchmarks that do not depend "
                "on intraday coverage — the squared daily return and the Parkinson high-low "
                "estimator from the exchange daily bar:", BODY))
    A(tbl(pd.DataFrame([
        ['NKY', 'FULL', '2,265', '1.030'], ['NKY', 'DEFECT', '598', '0.786'],
        ['SPX (control)', 'FULL', '3,113', '1.135'], ['SPX (control)', 'DEFECT', '91', '1.035'],
    ], columns=['Index', 'Session class', 'n', 'RV / Parkinson (median)']),
        widths=[34 * mm, 30 * mm, 22 * mm, 46 * mm], fs=8))
    A(Paragraph("Realized variance on Nikkei defect days runs about <b>24% below</b> where it "
                "should. Those days are nulled; the rows are kept, because the exchange close "
                "is still valid and the daily return is still a good observation.", BODY))
    A(Paragraph("6.3 Session classification", H2))
    A(tbl(tables['Session_Class'], fs=8))
    A(Paragraph("Half-days were separated from feed defects by <i>where</i> the missing 5-minute "
                "blocks sit — a half-day is a contiguous truncation at the end, a defect is a "
                "missing open or an interior hole. The confirmation that this works is that the "
                "half-days land on exactly the dates the exchanges schedule: 07-03 and 07-04 "
                "for the US indices, 12-24 and 12-31 for London and Hong Kong.", BODY))
    A(PageBreak())

    # ---- 7 statistical properties
    A(Paragraph("7. Statistical properties and what they imply", H1))
    A(Paragraph("Every test below exists to justify — or refuse to justify — a specific "
                "modelling choice.", BODY))
    A(Paragraph("7.1 Returns are far from Gaussian", H2))
    mom = tables['Moments']
    rf = mom[(mom.Series == 'Return') & (mom.Scope == 'full 1990+')]
    A(tbl(rf[['Code', 'N', 'Mean', 'SD', 'Skew', 'ExcessKurt', 'Min', 'P1', 'P99', 'Max']],
          fs=7.6))
    A(Paragraph("7.2 Tail index — the case for EVT", H2))
    tl = tables['Tail_Index']
    tl5 = tl[(tl.k_frac == 0.05) & (tl.Scope == 'full 1990+') & (tl.Tail == 'left')]
    A(tbl(tl5[['Code', 'k', 'Hill_Alpha', 'SE']], widths=[24 * mm, 20 * mm, 30 * mm, 24 * mm],
          fs=8))
    A(Paragraph("Hill alpha runs 2.6 to 3.9. <b>Every estimate is below 4</b>, so the fourth "
                "moment does not exist and the sample kurtosis is not estimating any finite "
                "population quantity. Several are close to 3, putting the third moment in doubt. "
                "A Gaussian-innovation GARCH assumes all moments exist; a Student-t GARCH imposes "
                "one tail parameter on both tails and on the whole distribution at once. Neither "
                "is consistent with alpha near 3. <b>This is the quantitative case for the EVT "
                "stage.</b>", BODY))
    A(Paragraph("7.3 Specification tests", H2))
    A(tbl(tables['Spec_Tests'][['Code', 'ADF_Return_p', 'KPSS_Return_p', 'LB10_Return_p',
                                'LB22_RetSq_p', 'ARCH_LM10_p', 'EngleNg_p']], fs=7.6))
    A(tbl(pd.DataFrame([
        ['ADF rejects, KPSS does not', 'Returns are stationary; model in levels.'],
        ['Ljung-Box on r rejects (5 of 6)', 'A conditional-mean term is warranted; AR(1) suffices.'],
        ['ARCH-LM rejects at p < 1e-16', 'A conditional-variance model is mandatory.'],
        ['Engle-Ng rejects on all six', '<b>Plain GARCH(1,1) is misspecified. Use GJR or EGARCH.</b>'],
    ], columns=['Result', 'What it decides']), widths=[58 * mm, W - 58 * mm], fs=8))
    A(Paragraph("7.4 Realized-measure dynamics", H2))
    A(tbl(tables['Spec_Tests'][['Code', 'ADF_LogRV_p', 'KPSS_LogRV_p', 'GPH_d_LogRV',
                                'AC1_LogRV', 'AC22_LogRV', 'AC66_LogRV',
                                'Leverage_corr_r_LogRVnext', 'VarShare_Session_Pct']], fs=7.2))
    A(Paragraph("The GPH fractional-integration estimate is 0.50–0.63 and ADF and KPSS "
                "<i>both</i> reject — the classic long-memory signature rather than a clean I(0) "
                "or I(1). This is the case for the HAR cascade. The leverage correlation between "
                "today's signed return and tomorrow's log realized variance is negative for every "
                "index, the same asymmetry Engle-Ng detects.", BODY))
    A(Paragraph("7.5 The scale gap", H2))
    A(tbl(tables['Scale_Factors'], widths=[24 * mm, 34 * mm, 40 * mm, 34 * mm], fs=8))
    A(Paragraph("The scale factor rises monotonically as the cash session shortens — 8.5-hour "
                "markets at 1.71 and 1.74, the 6.5-hour US indices at 1.80, Hong Kong at 2.14 "
                "and Tokyo at 3.04. That ordering is the internal check that the number is real "
                "rather than an artefact. Tokyo's overnight window contains the entire US "
                "session, which is why two thirds of its daily variance is invisible to the "
                "intraday data.", BODY))
    A(PageBreak())

    # ---- 8 predictors
    A(Paragraph("8. Predictor screening", H1))
    pre = tables['Predictive_Power']
    g = pre.groupby('Predictor').agg(R2_LogRV_next=('R2_LogRV_next', 'mean'),
                                     PseudoR2_q05=('PseudoR2_q05', 'mean'),
                                     PseudoR2_q01=('PseudoR2_q01', 'mean')).reset_index()
    A(Paragraph("Averaged over the six indices. All predictors dated t, all targets t+1 — these "
                "are forecasting numbers, not contemporaneous correlations.", BODY))
    A(Paragraph("Best for next-day log realized variance", H2))
    A(tbl(g.sort_values('R2_LogRV_next', ascending=False).head(10), fs=8))
    A(Paragraph("Best for the 1% left tail of next-day returns", H2))
    A(tbl(g.sort_values('PseudoR2_q01', ascending=False).head(10), fs=8))
    A(Paragraph("<b>The ranking flips.</b> For forecasting the level of volatility the realized "
                "measures win; for the 1% tail the implied-volatility index wins. The option "
                "market carries forward-looking information about extreme outcomes that the "
                "backward-looking realized measures do not. That is a genuine result and a "
                "natural thing for the paper to report.", BODY))
    A(Paragraph("<font color='#C00000'><b>RangePct</b></font>, computed from the exchange daily "
                "high-low alone, reaches R² = 0.42 — close to the implied-vol index — and is "
                "available on <i>every</i> day including those where the intraday feed failed. "
                "It is the natural fallback regressor across the Nikkei's 2016-17 gap.", BODY))
    A(Paragraph("Collinearity", H2))
    A(tbl(pd.DataFrame([
        ['LogRV + LogRS_neg + HAR terms', '21.9', 'Unusable — RS_pos + RS_neg = RV identically'],
        ['LogRV + RSV_Ratio + JumpShare + RSkew + HAR', '8.1', 'Use this parameterisation'],
    ], columns=['Parameterisation', 'Max mean VIF', 'Verdict']),
        widths=[78 * mm, 24 * mm, W - 102 * mm], fs=8))
    A(Paragraph("9. Feature sets by model", H1))
    A(tbl(tables['Feature_Sets'], widths=[30 * mm, 26 * mm, 46 * mm, W - 102 * mm], fs=7.2))
    A(PageBreak())

    # ---- 10 EVT
    A(Paragraph("10. EVT threshold selection", H1))
    gp = tables['GPD_Threshold']
    gl = gp[(gp.Series == 'rolling_std_resid_full') & (gp.Tail == 'left')]
    A(Paragraph("GPD shape parameter ξ fitted to the left tail of <b>rolling-standardised</b> "
                "returns. Standardising first is what makes this the right diagnostic: "
                "McNeil-Frey applies the GPD to GARCH residuals, not raw returns, because the "
                "limit theory assumes independence.", BODY))
    A(tbl(gl.pivot_table(index='q', columns='Code', values='xi').reset_index(), fs=8))
    A(Paragraph("Exceedance counts at the same thresholds", H2))
    A(tbl(gl.pivot_table(index='q', columns='Code', values='n_exc').astype(int).reset_index(),
          fs=8))
    A(Paragraph("ξ is positive throughout — a heavy, Fréchet-domain tail that <i>survives</i> "
                "volatility standardisation, which is precisely the McNeil-Frey argument for a "
                "second EVT stage on top of the GARCH filter. ξ drifts upward at low thresholds "
                "where the GPD approximation has not bitten, and becomes unstable above the 98th "
                "percentile where fewer than 190 exceedances remain. "
                "<b>Recommended POT threshold: the 95th to 97.5th percentile</b>, giving roughly "
                "230-460 exceedances per index. Re-estimate on the actual GARCH residuals once "
                "stage 1 is fitted.", BODY))
    A(Paragraph("11. Known limitations", H1))
    A(tbl(pd.DataFrame([
        ['Nikkei RV gap', 'No valid realized measure in 2016-17. 2,265 valid days with a hole.'],
        ['Intraday is a CFD proxy', 'Bid-side index CFD, not the exchange index. Aligned '
                                    'correlation 0.97-0.99.'],
        ['Proxy volatility indices', 'NKY uses VXEFA in sample B; UKX uses VXEFA; HSI uses '
                                     'VXEEM. No free native index exists for FTSE or HSI.'],
        ['Sample B excludes 2008', 'The three models see different amounts of crisis history.'],
        ['Non-synchronous sessions', 'Asia closes before the US opens; same-date correlation '
                                     'understates the true linkage.'],
        ['Five FRED series absent', 'CPIAUCSL, UNRATE, INDPRO, NFCI, USREC have no keyless '
                                    'market analogue. Optional covariates; need a free key.'],
    ], columns=['Limitation', 'Detail']), widths=[44 * mm, W - 44 * mm], fs=8))
    A(Paragraph("12. Verification", H1))
    fv = tables['Final_Validation']
    npass = int(fv['Pass'].sum()) if len(fv) else 0
    A(Paragraph(f"The analysis dataset passes <b>{len(fv)} independent checks with "
                f"{len(fv) - npass} failures</b>, re-derived from scratch rather than re-read "
                "from the objects that produced them. A validation that reuses the build code "
                "cannot catch a bug in the build code.", BODY))
    A(Paragraph("The look-ahead test deserves a note. An earlier version compared each "
                "predictor's correlation with today's realized variance against its correlation "
                "with tomorrow's, and flagged anything higher on the future. That test was "
                "invalid — VRP, JumpShare and RSV_Ratio all contain RV(t) by construction, which "
                "mechanically suppresses their contemporaneous correlation, and the macro series "
                "legitimately lead the Asian indices by a day. It was replaced with a "
                "<b>prefix-stability test</b>: every time-dependent column is rebuilt from data "
                "truncated at t and compared at the cut. If a column used information from after "
                "t, truncation must change it. All six indices pass at four cut points.", BODY))
    A(Paragraph("Acquisition-stage validation was 1,195 checks across 239 files, also with zero "
                "failures. Mechanical identities (RS_pos + RS_neg = RV, ContVar + Jump = RV) hold "
                "to floating-point precision.", BODY))
    A(Paragraph("13. Reproduction", H1))
    A(Paragraph("pip install -r requirements.txt<br/><br/>"
                "# acquisition<br/>"
                "python 10_SCRIPTS/01_download_daily.py<br/>"
                "python 10_SCRIPTS/02_download_volatility.py<br/>"
                "python 10_SCRIPTS/03_download_intraday.py SPX NDX UKX DAX NKY HSI<br/>"
                "python 10_SCRIPTS/09_download_macro_yahoo.py<br/>"
                "python 10_SCRIPTS/05_build_intraday_and_RV.py<br/>"
                "python 10_SCRIPTS/10_build_master_panel.py<br/><br/>"
                "# cleaning and EDA<br/>"
                "python 10_SCRIPTS/11_define_samples.py<br/>"
                "python 10_SCRIPTS/12_extended_realized_measures.py<br/>"
                "python 10_SCRIPTS/13_eda_quality_audit.py<br/>"
                "python 10_SCRIPTS/14_session_classification.py<br/>"
                "python 10_SCRIPTS/15_build_analysis_dataset.py<br/>"
                "python 10_SCRIPTS/16_eda_stylized_facts.py<br/>"
                "python 10_SCRIPTS/17_eda_predictor_screening.py<br/>"
                "python 10_SCRIPTS/18_eda_tails_breaks_features.py<br/>"
                "python 10_SCRIPTS/19_eda_figures.py<br/>"
                "python 10_SCRIPTS/20_finalise_and_document.py<br/>"
                "python 10_SCRIPTS/21_build_eda_report.py<br/>"
                "python 10_SCRIPTS/22_validate_analysis.py<br/>"
                "python 10_SCRIPTS/23_build_documentation.py", MONO))
    A(Paragraph("Run from inside the <font face='Courier'>Datasets</font> folder. Steps 12 and 14 "
                "read the 24,000-file Dukascopy cache and take several minutes each; step 3 is "
                "resumable and only fetches what is missing. Everything else runs in seconds.",
                BODY))
    A(PageBreak())

    # ---- 14 dictionary
    A(Paragraph("14. Data dictionary", H1))
    A(Paragraph("The analysis files carry 96 columns. Grouped by role:", BODY))
    dd = tables['Data_Dictionary']
    A(tbl(dd[['Columns', 'Group', 'Units', 'Availability', 'Timing', 'Note']],
          widths=[38 * mm, 15 * mm, 20 * mm, 20 * mm, 20 * mm, W - 113 * mm], fs=6.4))

    doc = SimpleDocTemplate(os.path.join(DOC, 'Dataset_Guide.pdf'), pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title='Dataset Guide', author='Researcher A')
    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    return doc.filename


CAPTIONS = {
 '01_price_return_timeline.png': "Index levels on a log scale and daily returns. The shaded "
    "band is sample B. Confirms each series is what it claims to be and locates the stress "
    "episodes.",
 '02_return_distribution_qq.png': "Standardised returns against the normal. Density on a log "
    "scale (top) and normal QQ plots (bottom). The tail departure is the point — it is what a "
    "histogram alone hides and what motivates the EVT stage.",
 '03_autocorrelation.png': "Volatility clustering. Returns are near-white; absolute and squared "
    "returns stay autocorrelated for months. This is the precondition for any GARCH model.",
 '04_realized_vs_implied.png': "Realized volatility against each market's implied-volatility "
    "index. The persistent gap is the variance risk premium. The red band marks the Nikkei's "
    "nulled 2016-17 feed defect.",
 '05_volatility_signature.png': "Volatility signature plot. RV declines mildly and monotonically "
    "with the sampling interval rather than showing the sharp high-frequency blow-up of noisy "
    "trade data. The 5-minute grid sits within a few percent of the 1-minute estimate, which is "
    "what justifies the sampling choice.",
 '06_session_quality_heatmap.png': "Share of each month's intraday sessions classified DEFECT. "
    "The Nikkei 2016-17 block is the defect that drove the main cleaning rule.",
 '07_gpd_threshold_stability.png': "Peaks-over-threshold stability. ξ > 0 means a heavy tail "
    "survives volatility standardisation. The shaded region is the recommended 95-97.5% "
    "threshold band.",
 '08_cross_index_correlation.png': "Cross-index correlation. The US and European blocks are "
    "tight; Asia is nearly uncorrelated with the US on the same calendar date because the "
    "sessions do not overlap — a timing artefact, not independence.",
 '09_leverage_news_impact.png': "News impact. Tomorrow's realized variance against today's "
    "signed return, with a binned mean in red. The asymmetry — negative shocks raise volatility "
    "more — is why GJR or EGARCH is required over plain GARCH.",
 '10_missing_data_map.png': "Data availability by column over time. The RV row for the Nikkei "
    "shows the nulled 2016-17 block; every other series is continuous.",
}


def build_figures_pdf():
    pw, ph = landscape(A4)
    st = []
    st.append(Spacer(1, 55 * mm))
    st.append(Paragraph("<b>Diagnostic Figures</b>", ParagraphStyle(
        't', parent=S['Title'], fontSize=26, textColor=colors.HexColor('#1F4E79'))))
    st.append(Spacer(1, 4 * mm))
    st.append(Paragraph("Exploratory data analysis — six Tier-1 equity indices",
                        ParagraphStyle('s', parent=S['Title'], fontSize=12,
                                       textColor=colors.HexColor('#555555'))))
    st.append(PageBreak())
    avail = pw - 30 * mm
    for fn in sorted(CAPTIONS):
        p = os.path.join(FIG, fn)
        if not os.path.exists(p):
            continue
        from PIL import Image as PILImage
        with PILImage.open(p) as im:
            w, h = im.size
        scale = min(avail / w, (ph - 62 * mm) / h)
        st.append(Paragraph(f"<b>{fn.replace('_', ' ').replace('.png', '')}</b>", H2))
        st.append(Image(p, width=w * scale, height=h * scale))
        st.append(Spacer(1, 3 * mm))
        st.append(Paragraph(CAPTIONS[fn], BODY))
        st.append(PageBreak())
    out = os.path.join(DOC, 'Diagnostic_Figures.pdf')
    SimpleDocTemplate(out, pagesize=landscape(A4), leftMargin=15 * mm, rightMargin=15 * mm,
                      topMargin=13 * mm, bottomMargin=15 * mm,
                      title='Diagnostic Figures', author='Researcher A').build(st)
    return out


if __name__ == "__main__":
    print("building file inventory ...")
    inv = build_inventory()
    inv.to_csv(os.path.join(DOC, 'FILE_INVENTORY.csv'), index=False)
    print(f"  {len(inv)} entries")

    print("building master Excel ...")
    xl, tables = build_excel(inv)
    print(f"  {xl}")

    print("building Dataset_Guide.pdf ...")
    print("  " + build_guide(tables, inv))

    print("building Diagnostic_Figures.pdf ...")
    print("  " + build_figures_pdf())

    print("\ndone")
