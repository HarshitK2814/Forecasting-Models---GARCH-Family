# -*- coding: utf-8 -*-
"""
Build two documents:

  00_DOCUMENTATION/Figure_Guide.pdf          every figure explained in full
  00_DOCUMENTATION/Handoff_to_Researcher_B.pdf   plan-vs-delivered and the handoff

The handoff document is checked against the project Executive Summary, so the status table
reflects the agreed division of labour rather than an after-the-fact description of whatever
happened to get done.
"""
import os
import warnings
warnings.filterwarnings('ignore')
import datetime
import pandas as pd
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, Image)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, '00_DOCUMENTATION')
FIG = os.path.join(ROOT, '09_FIGURES')
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
LOG = os.path.join(ROOT, '11_LOGS')
VAL = os.path.join(ROOT, '08_VALIDATION')
TODAY = datetime.date.today().isoformat()

S = getSampleStyleSheet()
BODY = ParagraphStyle('body', parent=S['BodyText'], fontSize=9, leading=12.5,
                      alignment=TA_LEFT, spaceAfter=5)
SMALL = ParagraphStyle('sm', parent=BODY, fontSize=8, leading=10.8)
H1 = ParagraphStyle('h1', parent=S['Heading1'], fontSize=15, spaceBefore=12, spaceAfter=6,
                    textColor=colors.HexColor('#1F4E79'))
H2 = ParagraphStyle('h2', parent=S['Heading2'], fontSize=11.5, spaceBefore=9, spaceAfter=4,
                    textColor=colors.HexColor('#2E5F8A'))
H3 = ParagraphStyle('h3', parent=S['Heading3'], fontSize=9.5, spaceBefore=6, spaceAfter=2,
                    textColor=colors.HexColor('#444444'))
CELL = ParagraphStyle('cell', parent=BODY, fontSize=7.6, leading=9.8, spaceAfter=0)
MONO = ParagraphStyle('mono', parent=BODY, fontName='Courier', fontSize=8, leading=10.5,
                      backColor=colors.HexColor('#F2F2F2'), borderPadding=5)


def esc(x):
    return str(x).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def tbl(df, widths=None, fs=7.6, head='#1F4E79'):
    cs = ParagraphStyle('c', parent=CELL, fontSize=fs, leading=fs * 1.3)
    hs = ParagraphStyle('h', parent=cs, fontName='Helvetica-Bold', textColor=colors.white)
    data = [[Paragraph(esc(c), hs) for c in df.columns]]
    for _, r in df.iterrows():
        data.append([Paragraph(str(v) if not isinstance(v, float) else f"{v:,.4g}", cs)
                     for v in r])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(head)),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#B0B0B0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F4F7FA')]),
        ('LEFTPADDING', (0, 0), (-1, -1), 3.5), ('RIGHTPADDING', (0, 0), (-1, -1), 3.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5)]))
    return t


def mkfooter(text):
    def f(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor('#808080'))
        canvas.drawString(16 * mm, 10 * mm, f"{text} — {TODAY}")
        canvas.drawRightString(doc.pagesize[0] - 16 * mm, 10 * mm, f"page {doc.page}")
        canvas.restoreState()
    return f


# ==================================================================== FIGURE GUIDE
# file -> (title, what it plots, how to read it, what OUR data shows, what it decides, caveat)
FIGURES = [
 ("01_price_return_timeline.png", "Price and return timeline",
  "Twelve panels. Left column: the closing level of each index on a LOG scale, so equal "
  "vertical distances are equal percentage moves. Right column: the daily log return in per "
  "cent, capped at +/-16 for readability. The blue shaded band on both is sample B.",
  "Read the left column for trend and the right for risk. On a log scale a straight line is "
  "constant compound growth. In the return panels look for VERTICAL THICKENING - bands of "
  "large moves clustered together - rather than for individual spikes.",
  "All six show the same three thickenings: 2000-02, 2008-09 and 2020. The shaded band makes "
  "the central compromise visible at a glance - sample B starts after the GFC, so the "
  "realized-measure models never see 2008, while the daily-only models do.",
  "Confirms each series is the index it claims to be, with no level breaks, splits or "
  "stitching errors, and locates the stress episodes that the regime analysis will use.",
  "The +/-16% clip hides the exact height of the very largest moves. Read those off the "
  "Worst_Days table, not this figure."),

 ("02_return_distribution_qq.png", "Return distribution and normal QQ plots",
  "Top row: histogram of standardised returns with the standard normal density in red, "
  "plotted with a LOGARITHMIC y-axis. Bottom row: normal quantile-quantile plots - empirical "
  "quantiles against theoretical normal quantiles, with the red line showing where points "
  "would fall if returns were Gaussian.",
  "The log y-axis is the whole point: on a linear scale the tails are invisible because the "
  "density is tiny there. On a log scale a Gaussian is a downward parabola, so anything that "
  "sits ABOVE the red curve out in the wings is excess tail mass. In the QQ plot, points "
  "bending BELOW the line on the left and ABOVE on the right mean both tails are fatter than "
  "normal; the further they bend, the fatter.",
  "Every index bends away from the line at both ends, and the left end bends further than the "
  "right - fat tails plus negative skew. Excess kurtosis runs 5.9 to 10.9. The most extreme "
  "points sit five to ten standard deviations out, which under a Gaussian would be "
  "essentially impossible.",
  "Rules out a Gaussian innovation and motivates the EVT stage. Together with the Hill "
  "estimates it is the visual half of the argument that a normal-innovation GARCH will "
  "understate 1% VaR.",
  "Standardising by the FULL-SAMPLE mean and standard deviation, as here, mixes calm and "
  "crisis periods, so some of the apparent fat-tailedness is volatility clustering rather "
  "than true unconditional tail weight. That is exactly why the EVT stage is applied to GARCH "
  "residuals and not to these raw returns."),

 ("03_autocorrelation.png", "Autocorrelation of returns, absolute returns and squared returns",
  "Three panels sharing a lag axis of 1 to 60 trading days, with all six indices overlaid. "
  "Left: autocorrelation of the return itself. Middle: of the absolute return. Right: of the "
  "squared return. The dotted grey lines are the approximate 95% confidence band around zero.",
  "Compare the three panels against each other. If a series were independent, all three would "
  "sit inside the band. What you should see is the left panel near zero and the other two "
  "clearly and persistently positive.",
  "Exactly that. Returns are close to white noise - a small negative first-order "
  "autocorrelation and little else. Absolute and squared returns stay well above the "
  "confidence band out to sixty days and decay slowly rather than geometrically.",
  "This is the precondition for GARCH. Returns are nearly unpredictable in the MEAN but "
  "strongly predictable in the VARIANCE, which is the entire premise of conditional "
  "volatility modelling. The slow, non-geometric decay is also the first hint of long memory, "
  "which the GPH estimates confirm.",
  "The confidence band assumes independence and is therefore too narrow for the |r| and r^2 "
  "panels. Treat those bands as indicative; the formal evidence is the Ljung-Box and ARCH-LM "
  "tests, which reject at p < 1e-16."),

 ("04_realized_vs_implied.png", "Realized volatility against the implied volatility index",
  "One panel per index. Dark red is realized volatility from the 5-minute intraday data, "
  "annualised into per cent. Dark blue is that market's implied-volatility index. Both on the "
  "same axis, capped at 95.",
  "Look at the GAP between the two lines, not just their co-movement. Implied volatility sits "
  "persistently above realized volatility; that wedge is the variance risk premium, the "
  "compensation option sellers earn. Then look at the SPIKES: implied typically jumps first "
  "and overshoots.",
  "The premium is positive nearly everywhere and widens sharply in stress. The red block on "
  "the Nikkei panel marks 2016-17, where the intraday feed is broken and realized volatility "
  "has been nulled - the red line simply stops.",
  "Justifies including VRP and the implied-volatility level as quantile-regression "
  "predictors, and gives the visual context for the screening result that implied volatility "
  "beats every realized measure for the 1% tail.",
  "The two series are NOT on identical footing. Realized volatility here covers the cash "
  "session only, while the implied index prices a full 30 calendar days including overnight "
  "and weekend risk. Part of the visible gap is that mismatch rather than a risk premium - "
  "use ScaleFactor_HL before quantifying the premium."),

 ("05_volatility_signature.png", "Volatility signature plot",
  "Left: mean realized variance at 1, 5, 10, 15 and 30-minute sampling, divided by the "
  "5-minute value so all six indices share one axis. Right: the same in absolute annualised "
  "volatility per cent. The red dashed line marks the 5-minute grid actually used.",
  "This is the standard microstructure-noise diagnostic. Sampling too finely makes RV pick up "
  "bid-ask bounce and quote noise instead of true volatility, which INFLATES it. The textbook "
  "picture is a sharp rise as the interval shortens toward zero, flattening out somewhere "
  "around 5 to 30 minutes. You choose the coarsest frequency that is still on the flat part.",
  "The profile declines mildly and monotonically - roughly 15 to 20% in variance between 1 "
  "and 30 minutes - with no sharp high-frequency blow-up. The 1-minute to 5-minute ratio is "
  "only 1.06 to 1.09, against 1.3 to 2.0 typical of trade data. The feed is a smoothed CFD "
  "quote, so it carries little microstructure noise.",
  "Justifies the 5-minute sampling choice: it sits within a few per cent of the 1-minute "
  "estimate, so almost nothing is lost, while remaining the literature standard and immune to "
  "the noise objection.",
  "Low noise is not automatically good news. A smoothed quote may also DAMP genuine "
  "high-frequency variation, which would bias RV downward relative to a trade-based "
  "estimator. This is disclosed as a limitation, not presented as a strength."),

 ("06_session_quality_heatmap.png", "Intraday session quality by month",
  "One panel per index, months down the vertical axis and years across the horizontal. Colour "
  "is the SHARE of that month's sessions classified DEFECT: green is clean, red is wholly "
  "unusable. White means no data was cached for that month.",
  "Scan for red blocks. Isolated red cells are ordinary one-off feed interruptions. A solid "
  "red rectangle spanning many months is a systematic outage and is a completely different "
  "problem.",
  "The Nikkei has a solid red block covering the whole of 2016 and 2017 - the Dukascopy feed "
  "carries no bars at all in the 09:00 local hour for those two years. The S&P shows a "
  "smaller cluster in 2012-13. Everything else is green.",
  "This is the figure behind the single most consequential cleaning decision in the project: "
  "nulling the Nikkei's realized measures for 2016-17. Because intraday volatility is "
  "U-shaped, losing the opening hour biases RV downward by far more than the missing time "
  "fraction, measured at 24% low against the Parkinson benchmark.",
  "An earlier version of this figure averaged a three-level class code and passed it through "
  "a three-colour palette, which silently rendered a month that was half broken as if it were "
  "a half-day. It now plots the defect SHARE on a continuous scale. If you regenerate it, do "
  "not revert to the categorical version."),

 ("07_gpd_threshold_stability.png", "Generalised Pareto threshold stability",
  "Two panels, left tail and right tail. The horizontal axis is the threshold expressed as a "
  "quantile from 0.80 to 0.99; the vertical axis is the fitted GPD shape parameter xi with "
  "asymptotic error bars. One line per index. The green band marks the recommended 95 to "
  "97.5% region.",
  "You are looking for a PLATEAU. Below the right threshold the GPD limit theory has not "
  "taken hold and xi drifts as the threshold rises; above it, so few exceedances remain that "
  "the estimate becomes noisy and the error bars widen. The correct threshold is the lowest "
  "one inside the stable stretch, because that keeps the most data.",
  "xi is positive at every threshold for every index - a heavy Frechet-domain tail. It rises "
  "from around 0.03-0.10 at the 80th percentile and settles around 0.15-0.25 from roughly the "
  "95th, before becoming erratic above the 98th where fewer than 190 exceedances are left.",
  "Sets the peaks-over-threshold cutoff at the 95th to 97.5th percentile, giving 230 to 460 "
  "exceedances per index. Critically, xi stays positive AFTER volatility standardisation, "
  "which is precisely the McNeil-Frey argument that a GARCH filter alone does not remove the "
  "tail and a second EVT stage is needed.",
  "This is fitted to returns standardised by a rolling 252-day volatility, used as a stand-in "
  "because no GARCH had been fitted at the EDA stage. Genuine GARCH residuals will differ. "
  "Treat the band as a starting expectation and REDO this plot on the real residuals."),

 ("08_cross_index_correlation.png", "Cross-index correlation",
  "Two heatmaps on the balanced sample. Left: correlation of log realized variance between "
  "each pair of indices. Right: correlation of daily returns. Warmer colours are higher.",
  "Look for BLOCKS. Groups of indices that move together will show as bright squares along "
  "the diagonal. Then compare the two panels - volatility and returns need not share the same "
  "structure.",
  "Two tight blocks: the US pair at 0.96 and the European pair at 0.85. Asia is largely "
  "detached, and Asia-to-US return correlation is only about 0.13-0.18. The first principal "
  "component of the log-RV matrix explains 62% of the variation.",
  "Establishes that the six indices are not six independent experiments - a pooled test that "
  "treats them as such will overstate its own significance. Supports using Hansen's Model "
  "Confidence Set, which handles dependence across the compared series.",
  "THE LOW ASIA-US CORRELATION IS A TIMING ARTEFACT, NOT ECONOMIC INDEPENDENCE. Tokyo and "
  "Hong Kong close before New York opens, so a US shock on day t reaches Asia on day t+1. Do "
  "not interpret this figure as evidence that Asian markets are decoupled."),

 ("09_leverage_news_impact.png", "News impact curve",
  "One scatter per index: today's signed return in per cent on the horizontal axis against "
  "TOMORROW's log realized variance on the vertical. The red line joins the means of twenty "
  "equal-count bins, so it traces the average volatility response to a shock of each size.",
  "The red line is the empirical news impact curve. Under a symmetric GARCH it would be a "
  "symmetric V centred on zero: a -2% day and a +2% day would raise tomorrow's variance "
  "equally. Compare the steepness of the left arm against the right.",
  "For all six the left arm is visibly steeper than the right. A negative shock raises "
  "tomorrow's volatility more than a positive shock of the same size. The correlation between "
  "today's signed return and tomorrow's log RV runs -0.12 to -0.20.",
  "This is the evidence that PLAIN GARCH(1,1) IS MISSPECIFIED for this data. A symmetric "
  "model cannot reproduce an asymmetric news impact curve. GJR-GARCH or EGARCH is required, "
  "and the formal Engle-Ng sign-bias test rejects on all six indices.",
  "The binned mean is sensitive to the tails, where few observations sit, so the extreme ends "
  "of the red line are less reliable than the middle. The x-axis is clipped at +/-6% and a "
  "handful of points lie outside it."),

 ("10_missing_data_map.png", "Data availability map",
  "One panel per index, 2011 onward. Each row is a column of the dataset, each horizontal "
  "position a trading day. Green means the value is present, white means missing.",
  "Read across each row for continuity. A clean row is solid green. Gaps show as white "
  "vertical stripes, and a systematic outage as a wide white block.",
  "The RV, RS_neg and Jump rows for the Nikkei show a clear white block over 2016-17 - the "
  "nulled defect period. Return, VolIdx and the range and macro rows are continuous "
  "everywhere, which is exactly the intended outcome: the intraday failure removed the "
  "realized measures without touching the daily observations.",
  "Documents honest missingness. It is the visual proof that rows were not deleted when the "
  "intraday feed failed - only the affected columns were nulled - so the return series that "
  "GARCH-EVT and quantile regression estimate on remains complete.",
  "Green means PRESENT, not VALID. A realized measure can be present and still fail the "
  "RV_Valid gate on a half-day. Always filter on RV_Valid rather than on notna()."),
]


def build_figure_guide():
    pw, ph = landscape(A4)
    avail = pw - 30 * mm
    st = []
    A = st.append
    A(Spacer(1, 50 * mm))
    A(Paragraph("<b>Figure Guide</b>", ParagraphStyle(
        't', parent=S['Title'], fontSize=28, textColor=colors.HexColor('#1F4E79'))))
    A(Spacer(1, 4 * mm))
    A(Paragraph("What every diagnostic figure shows, how to read it, and what it decides",
                ParagraphStyle('s', parent=S['Title'], fontSize=12.5,
                               textColor=colors.HexColor('#555555'))))
    A(Spacer(1, 3 * mm))
    A(Paragraph("Volatility and tail-risk forecasting — six Tier-1 equity indices",
                ParagraphStyle('s2', parent=S['Title'], fontSize=10,
                               textColor=colors.HexColor('#888888'))))
    A(PageBreak())

    A(Paragraph("What these figures are — and are not", H1))
    A(Paragraph("These ten figures are <b>diagnostics, not data</b>. They were generated from "
                "the dataset during exploratory analysis to test specific questions, and they "
                "exist as evidence for the cleaning and modelling decisions that follow. "
                "Nothing in the pipeline reads them; deleting them would not affect the dataset. "
                "<font face='Courier'>10_SCRIPTS/19_eda_figures.py</font> regenerates all ten in "
                "about thirty seconds.", BODY))
    A(Paragraph("Two of them are load-bearing rather than illustrative. Figure 06 is the reason "
                "the Nikkei's 2016-17 realized measures were nulled, and figure 09 is the reason "
                "a symmetric GARCH is not an acceptable baseline. If either were wrong, the "
                "corresponding decision would have to be revisited.", BODY))
    A(Paragraph("Likely use in the paper", H2))
    A(tbl(pd.DataFrame([
        ["02, 03, 09", "Main text", "The stylised facts that motivate the model class"],
        ["07", "Methodology", "Justifies the EVT threshold choice"],
        ["04, 08", "Main text or appendix", "Variance risk premium; cross-index dependence"],
        ["01, 05, 06, 10", "Appendix / supplementary", "Data provenance and quality evidence"],
    ], columns=["Figure", "Where", "Why"]),
        widths=[30 * mm, 42 * mm, avail - 72 * mm], fs=8.5))
    A(PageBreak())

    for i, (fn, title, plots, how, shows, decides, caveat) in enumerate(FIGURES, 1):
        p = os.path.join(FIG, fn)
        if not os.path.exists(p):
            continue
        A(Paragraph(f"Figure {i:02d} — {title}", H1))
        with PILImage.open(p) as im:
            w, h = im.size
        scale = min(avail / w, (ph * 0.42) / h)
        A(Image(p, width=w * scale, height=h * scale))
        A(Spacer(1, 2.5 * mm))
        rows = [["What it plots", plots], ["How to read it", how],
                ["What our data shows", shows], ["What it decides", decides],
                ["Caveat", caveat]]
        cs = ParagraphStyle('c2', parent=CELL, fontSize=8, leading=10.5)
        hs = ParagraphStyle('h2c', parent=cs, fontName='Helvetica-Bold')
        data = [[Paragraph(a_, hs), Paragraph(b_, cs)] for a_, b_ in rows]
        t = Table(data, colWidths=[34 * mm, avail - 34 * mm], hAlign='LEFT')
        t.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#C0C0C0')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#EDF2F7')),
            ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#FFF6F6')),
            ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
        A(t)
        A(PageBreak())

    out = os.path.join(DOC, 'Figure_Guide.pdf')
    SimpleDocTemplate(out, pagesize=landscape(A4), leftMargin=15 * mm, rightMargin=15 * mm,
                      topMargin=13 * mm, bottomMargin=14 * mm,
                      title='Figure Guide', author='Researcher A').build(
        st, onFirstPage=mkfooter('Figure Guide'), onLaterPages=mkfooter('Figure Guide'))
    return out


# ==================================================================== HANDOFF
TASKS = [
 ("Data acquisition (daily & intraday) & cleaning", "A", "40 h", "COMPLETE",
  "Six indices, 1990-2026 daily; intraday 2011-09 onward (DAX 2013-09). 1,195 acquisition "
  "checks, 0 failures."),
 ("Realized volatility construction", "A", "16 h", "COMPLETE",
  "RV at five frequencies plus semivariance, MedRV, quarticity, subsampled RV, jumps. "
  "Exceeds the plan, which specified only RV = sum of squared intraday returns."),
 ("Baseline GARCH & GJR/EGARCH coding", "A", "24 h", "NOT STARTED",
  "Researcher A leads this in the plan. EDA has established the specification: Engle-Ng "
  "rejects on all six indices, so GJR or EGARCH is required, not plain GARCH."),
 ("Realized GARCH model coding", "A", "24 h", "NOT STARTED",
  "Researcher A leads. Inputs are ready; the measurement equation must absorb ScaleFactor_HL."),
 ("Rolling out-of-sample forecast engine", "A", "24 h", "NOT STARTED",
  "Researcher A leads. Must handle the Nikkei 2016-17 RV gap by using the last AVAILABLE "
  "observations rather than the last N calendar days."),
 ("Robustness checks (thresholds, windows, horizons)", "A", "16 h", "NOT STARTED",
  "Researcher A leads. Samples A and C are pre-built as ready-made robustness runs."),
 ("Environment setup & reproducibility", "A", "part of 16 h", "PARTIAL",
  "requirements.txt exists and all 26 scripts are numbered and rerunnable. NO git repository, "
  "no Dockerfile, no seed policy yet. The 'arch' package is not installed."),
 ("GARCH-EVT (filtering + EVT tail fitting)", "B", "24 h", "READY TO START",
  "Threshold diagnostics done: recommend the 95th-97.5th percentile, 230-460 exceedances. "
  "Must be re-estimated on genuine GARCH residuals."),
 ("Quantile regression (with predictors)", "B", "16 h", "READY TO START",
  "Predictors screened, collinearity resolved, non-stationary levels flagged and differenced "
  "forms supplied."),
 ("Evaluation metrics (RMSE, QLIKE, VaR/ES backtest)", "B", "24 h", "READY TO START",
  "Realized measure available as the volatility proxy; use RV_Scaled for like-for-like "
  "comparison against squared daily returns."),
 ("Crisis/regime analysis", "B", "16 h", "READY TO START",
  "CrisisLabel (10 named windows), VolRegime and VolRegime_ExAnte now shipped in the dataset."),
 ("Statistical tests (Diebold-Mariano etc.)", "B", "12 h", "READY TO START",
  "BalancedRV_B flag supplied for pooled comparison; per-index tests should use each index's "
  "own RV_Valid days."),
 ("Figures & tables preparation", "A/B", "16 h", "PARTIAL",
  "Ten EDA diagnostic figures complete and documented. Results figures - forecast vs realized, "
  "VaR exceedance, performance heatmaps - cannot exist until models are fitted."),
]

MILESTONES = [
 ("Clean data files & code", "2", "A", "DELIVERED", "Exceeds scope: 6 indices, not 1-3"),
 ("Realized volatility series", "3", "A", "DELIVERED", "Exceeds scope: extended measures"),
 ("Baseline GARCH code", "4", "A", "OUTSTANDING", "Researcher A"),
 ("GARCH-EVT code", "5", "B", "OUTSTANDING", "Researcher B — inputs ready"),
 ("Realized GARCH code", "6", "A", "OUTSTANDING", "Researcher A"),
 ("Quantile regression code", "6", "B", "OUTSTANDING", "Researcher B — inputs ready"),
 ("Rolling forecast engine", "8", "A", "OUTSTANDING", "Researcher A"),
 ("Evaluation scripts", "9", "B", "OUTSTANDING", "Researcher B"),
 ("Initial results & figures", "10", "A/B", "OUTSTANDING", "Blocked on models"),
 ("Regime-specific analysis", "11", "B", "OUTSTANDING", "Labels now shipped"),
 ("Robustness analysis", "12", "A", "OUTSTANDING", "Samples A and C pre-built"),
 ("Statistical comparison tests", "12", "B", "OUTSTANDING", "Researcher B"),
 ("Code repository & environment", "13", "A/B", "PARTIAL", "No git repo yet"),
 ("Reproducibility audit report", "14", "A/B", "OUTSTANDING", "Needs seeds + env lock"),
 ("Handoff package for writers", "15", "A/B", "OUTSTANDING", "Blocked on results"),
]

DEVIATIONS = [
 ("Assets", "1-3 indices (S&P 500, NASDAQ-100, maybe a commodity or FX)",
  "6 indices across 5 regions: SPX, NDX, UKX, DAX, NKY, HSI",
  "Broader regional coverage was requested. Also makes a Model Confidence Set across a "
  "six-index panel possible, which a 1-3 asset study cannot support.",
  "Six times the model fits. Budget compute accordingly."),
 ("Intraday source", "NYSE TAQ via WRDS (academic licence), or commercial tick feeds",
  "Dukascopy index CFD, bid side, free",
  "The project operates under a strict free-data constraint. TAQ requires a WRDS "
  "subscription.",
  "RV is built from a PROXY, not the exchange index. Aligned correlation against the exchange "
  "daily return is 0.97-0.99. This must be disclosed in the paper."),
 ("Intraday coverage", "2010-2020, 10+ years including 2008 GFC and 2011 Euro crisis",
  "2011-09 onward; DAX only from 2013-09",
  "This is where free intraday history actually begins. Verified by probing earlier years and "
  "receiving empty responses.",
  "CRITICAL: the realized-measure models CANNOT see the GFC, the dot-com bust, the Asian "
  "crisis or the Euro sovereign crisis. GARCH-EVT and quantile regression can, since they "
  "estimate on daily data back to 1990. The three models therefore see different crisis "
  "histories and the paper must say so."),
 ("Daily coverage", "2010-2025", "1990-2026", "Free and available; more tail observations.",
  "Longer estimation sample for the daily-only models."),
 ("Risk indicators", "VIX as an exogenous variable",
  "17 implied-volatility indices plus 22 macro / risk-factor series",
  "Each index gets a regional volatility index where one exists; macro factors support the "
  "quantile-regression feature set.",
  "Richer predictor set, already screened for stationarity and collinearity."),
 ("Outlier handling", "\"Winsorize or filter erroneous ticks\"",
  "Erroneous INTRADAY sessions removed; daily RETURNS NOT winsorized at all",
  "Winsorizing returns would destroy the object of study. It shrinks the estimated GPD shape "
  "parameter and deletes the exceedances that identify the tail, producing VaR that looks "
  "well-calibrated only because the violations were removed. The extreme days were "
  "individually verified as real market events.",
  "DELIBERATE DEVIATION FROM THE PLAN. Researcher B must not reintroduce winsorization at the "
  "modelling stage."),
 ("Realized variance", "RV = sum of squared intraday returns",
  "That, plus realized semivariance, MedRV, bipower variation, realized quarticity, "
  "tripower quarticity, subsampled RV, jumps, realized skew and kurtosis",
  "Signed and jump-robust measures are close to expected practice in a modern realized-measure "
  "tail paper.", "HAR-Q and signed-jump specifications are available without new data work."),
 ("Crisis labels", "Define event periods by dates or volatility thresholds",
  "Both: 10 named CrisisLabel windows, plus VolRegime and a no-look-ahead VolRegime_ExAnte",
  "The plan left the choice open; both answer different questions.",
  "Regime analysis can start immediately. Note VolRegime uses full-sample cut-points and is "
  "for ex-post reporting only."),
 ("Storage", "CSV or Parquet", "CSV throughout", "Universally readable, no engine dependency.",
  "Files are larger. Convert to Parquet if load time becomes an issue."),
 ("Version control", "All code in a Git repo, branches for major features",
  "NOT a git repository",
  "Not yet initialised.",
  "GAP. Should be fixed before the modelling work starts, or the model code will have no "
  "history either."),
 ("Environment", "requirements.txt or environment.yml, plus a Dockerfile",
  "requirements.txt only",
  "Partial.",
  "GAP. Also: the 'arch' package the plan names is not installed."),
 ("Estimation window", "Initial estimation window e.g. first 2000 days",
  "Recommend 1000 days for the realized-measure models",
  "Sample B has about 3,100 valid RV days per index. A 2000-day window would leave barely "
  "1,100 forecast days and only ~11 expected exceedances at the 1% level.",
  "Researcher B's call, but state it explicitly. 1000 days is also the McNeil-Frey choice."),
]

FIRST_STEPS = [
 (1, "Install the environment", "pip install -r requirements.txt, then pip install arch. The "
  "plan names 'arch' for GARCH and VaR tests and it is currently absent."),
 (2, "Initialise version control", "git init at the project root and commit the Datasets "
  "folder state before writing any model code, excluding 12_CACHE_REGENERATION (810 MB)."),
 (3, "Read Dataset_Guide.pdf section 3", "Twenty-four precautions. The first four invalidate "
  "results outright if ignored."),
 (4, "Agree the sample and window", "Sample B is primary: all six indices, 2013-09-30 onward, "
  "2,685 common days. Recommend a 1000-day rolling estimation window."),
 (5, "Confirm the evaluation design", "Per-index estimation on each index's own RV_Valid days; "
  "BalancedRV_B (1,994 days) only for pooled statistics. Do not force a balanced panel "
  "per-index."),
 (6, "Decide the non-synchronous-session convention", "Asia closes before the US opens. Either "
  "adopt a lagged-information convention or estimate index by index and pool only losses. "
  "This must be settled BEFORE any cross-index result is produced."),
 (7, "Re-run the EVT threshold diagnostic", "On genuine GARCH residuals. The shipped 95-97.5% "
  "recommendation comes from rolling-standardised returns as a stand-in."),
 (8, "Handle the Nikkei gap explicitly", "No valid RV in 2016-17. Rolling windows must use the "
  "last AVAILABLE observations; no Realized-GARCH forecast can be produced or evaluated "
  "inside the gap."),
]


def build_handoff():
    W = A4[0] - 36 * mm
    st = []
    A = st.append
    A(Spacer(1, 42 * mm))
    A(Paragraph("<b>Handoff to Researcher B</b>", ParagraphStyle(
        't', parent=S['Title'], fontSize=25, textColor=colors.HexColor('#1F4E79'))))
    A(Spacer(1, 4 * mm))
    A(Paragraph("Dataset delivery, and how it compares with the project Executive Summary",
                ParagraphStyle('s', parent=S['Title'], fontSize=12,
                               textColor=colors.HexColor('#555555'))))
    A(Spacer(1, 16 * mm))
    A(tbl(pd.DataFrame({
        'Item': ['From', 'To', 'Date', 'Delivered', 'Status of the data',
                 'Status of the models', 'Reference plan'],
        'Value': ['Researcher A (data acquisition, cleaning, preprocessing)',
                  'Researcher B (GARCH-EVT, quantile regression, evaluation)',
                  TODAY,
                  'Datasets/ — 6 indices, 96 columns, 1990-2026, fully documented',
                  'COMPLETE and validated: 1,195 + 158 checks, 0 failures',
                  'NONE FITTED. No model has been estimated.',
                  'Executive Summary.pdf, project root']}),
        widths=[36 * mm, W - 36 * mm], fs=8.6))
    A(PageBreak())

    A(Paragraph("1. Read this first — the handoff is not a clean transfer", H1))
    A(Paragraph("The Executive Summary divides the work between two researchers, and it "
                "assigns <b>more to Researcher A than data</b>. Specifically, A leads:", BODY))
    A(tbl(pd.DataFrame([
        ["Baseline GARCH & GJR/EGARCH coding", "24 h", "Week 4", "NOT STARTED"],
        ["Realized GARCH model coding", "24 h", "Week 6", "NOT STARTED"],
        ["Rolling out-of-sample forecast engine", "24 h", "Week 8", "NOT STARTED"],
        ["Robustness checks", "16 h", "Week 12", "NOT STARTED"],
    ], columns=["Researcher A module", "Budget", "Due", "Status"]),
        widths=[70 * mm, 20 * mm, 20 * mm, W - 110 * mm], fs=8.5, head='#C00000'))
    A(Spacer(1, 2 * mm))
    A(Paragraph("<b>That is 88 hours of Researcher A work still outstanding, and three of the "
                "five model-and-engine modules.</b> Handing the whole remaining project to "
                "Researcher B would move roughly a third of the total budget onto one person "
                "and leave B leading every module, which is not what the plan agreed. Either "
                "re-agree the split with B in writing, or A retains the baseline GARCH, "
                "Realized GARCH and rolling-engine work. Flagging it here rather than letting "
                "it surface at the week-8 checkpoint.", BODY))
    A(Paragraph("What IS complete is the data half: acquisition, realized-volatility "
                "construction, cleaning, exploratory analysis and documentation — the two "
                "Researcher A deliverables due in weeks 2 and 3, both exceeding their "
                "specified scope.", BODY))

    A(Paragraph("2. Task status against the plan", H1))
    A(tbl(pd.DataFrame(TASKS, columns=['Task area', 'Lead', 'Budget', 'Status', 'Notes']),
          widths=[46 * mm, 10 * mm, 13 * mm, 20 * mm, W - 89 * mm], fs=7))
    A(PageBreak())

    A(Paragraph("3. Deliverables and milestones", H1))
    A(tbl(pd.DataFrame(MILESTONES,
                       columns=['Deliverable', 'Week', 'Lead', 'Status', 'Note']),
          widths=[52 * mm, 12 * mm, 12 * mm, 24 * mm, W - 100 * mm], fs=7.4))
    A(Spacer(1, 2 * mm))
    A(Paragraph("<b>2 of 15 delivered.</b> Both are Researcher A data deliverables and both "
                "exceed their specified scope. The remaining 13 depend on models that do not "
                "yet exist.", BODY))

    A(Paragraph("4. Where the delivered data departs from the plan", H1))
    A(Paragraph("Every deviation below is deliberate and has a reason. Two of them change what "
                "the paper can claim and must be carried into the write-up.", BODY))
    A(tbl(pd.DataFrame(DEVIATIONS,
                       columns=['Area', 'Plan said', 'Delivered', 'Why', 'What it means for B']),
          widths=[19 * mm, 38 * mm, 38 * mm, 42 * mm, W - 137 * mm], fs=6.5))
    A(PageBreak())

    A(Paragraph("5. The two deviations that matter most", H1))
    A(Paragraph("5.1 The realized-measure models cannot see the 2008 crisis", H2))
    A(Paragraph("The plan called for 10+ years of intraday data covering the GFC. Free intraday "
                "history begins in 2011 (2013 for the DAX), so this was not achievable at zero "
                "cost. The consequence is structural, not cosmetic:", BODY))
    cp = os.path.join(ANA, 'CRISIS_PERIODS.csv')
    if os.path.exists(cp):
        c = pd.read_csv(cp)
        A(tbl(c[['CrisisLabel', 'Start', 'End', 'Days_in_SampleB', 'In_SampleB']],
              widths=[44 * mm, 24 * mm, 24 * mm, 28 * mm, W - 120 * mm], fs=8))
    A(Paragraph("Sample B contains 6 of the 10 named crisis windows, about 14% of its days. It "
                "misses the Asian crisis, the dot-com bust, the GFC and the Euro sovereign "
                "crisis. GARCH-EVT and quantile regression still see all of them because they "
                "estimate on daily data from 1990; Realized GARCH does not. <b>State this "
                "asymmetry explicitly in the paper — a referee will otherwise read the "
                "comparison as unfair.</b> It does still contain COVID, the 2022 rate shock, "
                "the 2015-16 China devaluation, Volmageddon, Q4 2018 and the 2024 yen-carry "
                "unwind, which is enough distinct stress to identify a tail.", BODY))
    A(Paragraph("5.2 Returns are not winsorized, and must not be", H2))
    A(Paragraph("The Executive Summary lists \"Outliers: Winsorize or filter erroneous ticks\" "
                "under preprocessing. Erroneous intraday sessions WERE removed. Daily returns "
                "were deliberately NOT winsorized, and this is a considered departure rather "
                "than an oversight. In a tail-risk study the extremes are the object of "
                "measurement: clipping them shrinks the estimated GPD shape parameter, deletes "
                "the exceedances that identify it, and produces VaR forecasts that appear "
                "well-calibrated precisely because the violations were removed. The largest "
                "returns were each verified as real market events. <b>Researcher B should not "
                "reintroduce winsorization at the modelling stage.</b>", BODY))

    A(Paragraph("6. What Researcher B receives", H1))
    A(tbl(pd.DataFrame([
        ["01_ANALYSIS_READY/<CODE>_analysis.csv", "The model-ready dataset. 96 columns, "
         "6 indices, 1990-2026, cleaned, gated and validated."],
        ["Dataset_Guide.pdf", "14 pages. Section 3 is 24 precautions — read before coding."],
        ["Figure_Guide.pdf", "Every diagnostic figure explained, with what it decides."],
        ["DATASET_MASTER_REPORT.xlsx", "26 sheets: dictionary, precautions, all EDA tables."],
        ["EDA_REPORT.md", "The full cleaning and exploratory analysis record."],
        ["CRISIS_PERIODS.csv", "10 named crisis windows with justification for each date."],
        ["FEATURE_SETS.csv", "Which fields feed which of the three models."],
        ["10_SCRIPTS/", "26 numbered, rerunnable scripts covering the whole pipeline."],
    ], columns=['Item', 'What it is']), widths=[62 * mm, W - 62 * mm], fs=8))

    A(Paragraph("7. First eight steps", H1))
    A(tbl(pd.DataFrame(FIRST_STEPS, columns=['#', 'Step', 'Detail']),
          widths=[7 * mm, 46 * mm, W - 53 * mm], fs=8))

    A(Paragraph("8. Specification already settled by the EDA", H1))
    A(Paragraph("These are not suggestions; they are conclusions from tests in the EDA report, "
                "and each is reproducible from the shipped validation tables.", BODY))
    A(tbl(pd.DataFrame([
        ["Plain GARCH(1,1) is insufficient", "Engle-Ng sign-bias rejects on all six indices "
         "(worst p = 7e-5). Use GJR or EGARCH."],
        ["A Gaussian innovation is insufficient", "Hill tail index 2.6-3.9, every estimate "
         "below 4 — the fourth moment does not exist."],
        ["The EVT stage is justified", "GPD shape xi stays positive AFTER volatility "
         "standardisation, at 0.15-0.25."],
        ["Long memory is present", "GPH d on log RV is 0.50-0.63; ADF and KPSS both reject."],
        ["An AR(1) mean term is warranted", "Ljung-Box on returns rejects for 5 of 6 indices."],
        ["Use the level-plus-share realized block", "LogRV with LogRS_neg gives VIF ~95; "
         "LogRV with RSV_Ratio, JumpShare and RSkew gives max VIF 8.1."],
        ["Differenced macro only", "US10Y_pct and TermSpread_pct fail ADF in levels "
         "(p = 0.87, 0.33)."],
        ["Implied vol leads for the tail", "For the 1% quantile the vol index beats every "
         "realized measure; for the level of volatility the realized measures win."],
    ], columns=['Conclusion', 'Evidence']), widths=[58 * mm, W - 58 * mm], fs=8))

    A(Paragraph("9. Open questions for Researcher B", H1))
    A(tbl(pd.DataFrame([
        ["Rolling window length", "1000 days recommended (McNeil-Frey). 2000 as in the plan "
         "would leave ~1,100 forecast days and ~11 exceedances at 1%."],
        ["Fixed or expanding window", "The plan mentions both. Fixed is the cleaner comparison; "
         "expanding uses more of the 1990-2013 daily history."],
        ["Non-synchronous sessions", "Must be settled before any cross-index result."],
        ["Half-days", "Currently excluded from RV_Valid. Reversible via IsHalfDay if you "
         "prefer to keep them."],
        ["Pooled vs per-index MCS", "BalancedRV_B supports pooling; per-index keeps more data."],
        ["ES backtest choice", "The plan names Acerbi-Szekely. Not implemented."],
    ], columns=['Question', 'Context']), widths=[44 * mm, W - 44 * mm], fs=8))

    out = os.path.join(DOC, 'Handoff_to_Researcher_B.pdf')
    SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=15 * mm, bottomMargin=16 * mm,
                      title='Handoff to Researcher B', author='Researcher A').build(
        st, onFirstPage=mkfooter('Handoff to Researcher B'),
        onLaterPages=mkfooter('Handoff to Researcher B'))
    return out


if __name__ == "__main__":
    print("building Figure_Guide.pdf ...")
    print("  " + build_figure_guide())
    print("building Handoff_to_Researcher_B.pdf ...")
    print("  " + build_handoff())
    print("\ndone")
