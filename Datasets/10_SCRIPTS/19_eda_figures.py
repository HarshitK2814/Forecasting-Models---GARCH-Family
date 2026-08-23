# -*- coding: utf-8 -*-
"""
EDA STAGE 7 - the diagnostic figure set.

Each figure answers a question that a table answers badly. Nothing decorative is produced.

  01 price and return timeline      does the series look like what it claims to be, and
                                    where are the stress episodes
  02 return distribution + QQ       how far from Gaussian, and specifically in the tails,
                                    where a histogram is useless and a QQ plot is decisive
  03 autocorrelation                the volatility-clustering fact: r is near-white, |r| and
                                    r^2 are strongly and persistently autocorrelated
  04 realized vs implied vol        the variance risk premium, and whether the vol index
                                    tracks the realized measure of its own market
  05 volatility signature plot      the microstructure-noise diagnostic that justifies the
                                    5-minute sampling choice
  06 session coverage heatmap       makes the NKY 2016-17 feed defect visible at a glance
  07 GPD threshold stability        where the peaks-over-threshold cutoff should sit
  08 cross-index correlation        the regional block structure of volatility
  09 leverage / news impact         asymmetry of the volatility response to signed returns
  10 missing-data map               what is actually available, per column, over time

Output: _figures/*.png
"""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
from statsmodels.tsa.stattools import acf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, '01_ANALYSIS_READY')
RVD = os.path.join(ROOT, '06_REALIZED_MEASURES')
VAL = os.path.join(ROOT, '08_VALIDATION')
FIG = os.path.join(ROOT, '09_FIGURES')
os.makedirs(FIG, exist_ok=True)

CODES = ["SPX", "NDX", "UKX", "DAX", "NKY", "HSI"]
NAMES = {"SPX": "S&P 500", "NDX": "Nasdaq-100", "UKX": "FTSE 100",
         "DAX": "DAX 40", "NKY": "Nikkei 225", "HSI": "Hang Seng"}
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 140, 'font.size': 8.5,
                     'axes.grid': True, 'grid.alpha': 0.25, 'axes.titlesize': 9.5,
                     'axes.spines.top': False, 'axes.spines.right': False})

A = {c: pd.read_csv(os.path.join(ANA, f'{c}_analysis.csv'), parse_dates=['Date'])
     for c in CODES}


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------- 01
fig, ax = plt.subplots(6, 2, figsize=(12, 13), sharex='col')
for i, c in enumerate(CODES):
    a = A[c]
    ax[i, 0].semilogy(a['Date'], a['Close'], lw=0.6, color='#1f4e79')
    ax[i, 0].set_ylabel(c)
    ax[i, 1].plot(a['Date'], 100 * a['Return'], lw=0.25, color='#8b1a1a')
    b = a[a['InSample_B']]
    if len(b):
        for k in (0, 1):
            ax[i, k].axvspan(b['Date'].min(), b['Date'].max(), color='#4472c4', alpha=0.07)
    ax[i, 1].set_ylim(-16, 16)
ax[0, 0].set_title('Index level (log scale); shaded = sample B')
ax[0, 1].set_title('Daily log return, %')
save(fig, '01_price_return_timeline.png')

# ---------------------------------------------------------------- 02
fig, ax = plt.subplots(2, 6, figsize=(16, 5.5))
for i, c in enumerate(CODES):
    r = A[c]['Return'].dropna().values
    z = (r - r.mean()) / r.std()
    ax[0, i].hist(z, bins=140, density=True, color='#4472c4', alpha=.75)
    g = np.linspace(-8, 8, 400)
    ax[0, i].plot(g, stats.norm.pdf(g), 'r-', lw=1.1, label='N(0,1)')
    ax[0, i].set_yscale('log')
    ax[0, i].set_xlim(-9, 9)
    ax[0, i].set_title(f'{c}  kurt={stats.kurtosis(r):.1f}')
    if i == 0:
        ax[0, i].legend(fontsize=7)
        ax[0, i].set_ylabel('density (log)')
    stats.probplot(z, dist='norm', plot=ax[1, i])
    ax[1, i].set_title('')
    ax[1, i].get_lines()[0].set_markersize(1.2)
    ax[1, i].get_lines()[0].set_color('#1f4e79')
    ax[1, i].get_lines()[1].set_color('red')
    if i:
        ax[1, i].set_ylabel('')
fig.suptitle('Standardised daily returns vs the normal distribution: density on a log scale '
             '(top) and normal QQ plot (bottom). Tail departure is the point.', y=1.02)
save(fig, '02_return_distribution_qq.png')

# ---------------------------------------------------------------- 03
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
L = 60
for c in CODES:
    r = A[c]['Return'].dropna().values
    ax[0].plot(range(1, L + 1), acf(r, nlags=L)[1:], lw=1.1, label=c)
    ax[1].plot(range(1, L + 1), acf(np.abs(r), nlags=L)[1:], lw=1.1)
    ax[2].plot(range(1, L + 1), acf(r ** 2, nlags=L)[1:], lw=1.1)
band = 1.96 / np.sqrt(len(A['SPX']['Return'].dropna()))
for k, t in enumerate(['returns  r(t)', 'absolute returns  |r(t)|', 'squared returns  r(t)^2']):
    ax[k].axhline(0, color='k', lw=.6)
    ax[k].axhline(band, color='grey', ls=':', lw=.8)
    ax[k].axhline(-band, color='grey', ls=':', lw=.8)
    ax[k].set_title(t)
    ax[k].set_xlabel('lag (days)')
ax[0].legend(ncol=2, fontsize=7)
ax[0].set_ylabel('autocorrelation')
fig.suptitle('Volatility clustering: returns are near-white, but |r| and r^2 stay '
             'autocorrelated for months. This is the precondition for a GARCH model.', y=1.04)
save(fig, '03_autocorrelation.png')

# ---------------------------------------------------------------- 04
fig, ax = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
for i, c in enumerate(CODES):
    b = A[c][A[c]['InSample_B']]
    ax[i].plot(b['Date'], b['RVol_Ann_Pct'], lw=.5, color='#8b1a1a',
               label='realized vol (session, annualised %)')
    ax[i].plot(b['Date'], b['VolIdx'], lw=.7, color='#1f4e79',
               label=f"implied vol index ({b['VolIdx_Symbol'].iloc[0]})")
    ax[i].set_ylabel(c)
    ax[i].set_ylim(0, 95)
    if i == 0:
        ax[i].legend(fontsize=7.5, loc='upper left')
    if c == 'NKY':
        ax[i].axvspan(pd.Timestamp('2016-01-01'), pd.Timestamp('2018-01-01'),
                      color='red', alpha=.10)
        ax[i].text(pd.Timestamp('2016-02-01'), 78, 'intraday feed defect\n(RV nulled)',
                   fontsize=7, color='darkred')
fig.suptitle('Realized volatility against the implied volatility index. The persistent gap '
             'is the variance risk premium.', y=1.005)
save(fig, '04_realized_vs_implied.png')

# ---------------------------------------------------------------- 05
# Plotted as a RATIO to the 5-min estimate so all six indices share one axis and the size
# of the effect is readable directly. Absolute levels are in the right-hand panel.
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
freqs = [1, 5, 10, 15, 30]
cl_ = pd.read_csv(os.path.join(VAL, 'eda2_session_class.csv'), parse_dates=['Date'])
for c in CODES:
    rv = pd.read_csv(os.path.join(RVD, f'{c}_RV_daily.csv'), parse_dates=['Date'])
    keep = set(cl_.loc[(cl_['Symbol'] == c) & (cl_['Class'] == 'FULL'), 'Date'])
    rv = rv[rv['Date'].isin(keep)]
    m = np.array([rv[f'RV_{f}min'].mean() for f in freqs])
    ax[0].plot(freqs, m / m[1], 'o-', ms=4, lw=1.2, label=c)
    ax[1].plot(freqs, 100 * np.sqrt(252 * m), 'o-', ms=4, lw=1.2, label=c)
for k, (yl, ttl) in enumerate([
        ('mean RV / mean RV at 5 min', 'relative to the 5-minute estimate'),
        ('mean annualised RVol %', 'absolute level')]):
    ax[k].axvline(5, color='red', ls='--', lw=.9)
    ax[k].set_xlabel('sampling interval (min)')
    ax[k].set_ylabel(yl)
    ax[k].set_title(ttl)
ax[0].axhline(1.0, color='k', lw=.6)
ax[0].legend(ncol=3, fontsize=7)
fig.suptitle('Volatility signature plot. RV declines mildly and monotonically with the '
             'sampling interval — about 15-20% in variance between 1 and 30 minutes — rather '
             'than showing the sharp high-frequency blow-up of noisy trade data. The 5-minute '
             'grid (red) sits within a few percent of the 1-minute estimate.', y=1.12)
save(fig, '05_volatility_signature.png')

# ---------------------------------------------------------------- 06
cl = pd.read_csv(os.path.join(VAL, 'eda2_session_class.csv'), parse_dates=['Date'])
cl['Year'] = cl['Date'].dt.year
cl['Month'] = cl['Date'].dt.month
# Shown as the FRACTION of each month's sessions that are DEFECT, on a continuous scale.
# An earlier version averaged a 0/1/2 class code and mapped it through a three-colour
# palette, which silently rendered a month that was half broken as if it were a half-day.
fig, ax = plt.subplots(2, 3, figsize=(14, 6))
for i, c in enumerate(CODES):
    s = cl[cl['Symbol'] == c].copy()
    s['bad'] = s['Class'].eq('DEFECT').astype(float)
    piv = s.pivot_table(index='Month', columns='Year', values='bad', aggfunc='mean')
    a_ = ax[i // 3, i % 3]
    im = a_.imshow(piv.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1,
                   extent=[piv.columns.min() - .5, piv.columns.max() + .5, 12.5, .5])
    a_.set_title(f'{c} — {NAMES[c]}')
    a_.set_ylabel('month')
    a_.grid(False)
    fig.colorbar(im, ax=a_, fraction=.046, pad=.02)
fig.suptitle('Share of each month\'s intraday sessions classified DEFECT (feed failure). '
             'Green = clean, red = wholly unusable. The NKY 2016-17 block is the defect that '
             'drove the cleaning rule; white = no data cached for that month.', y=1.03)
save(fig, '06_session_quality_heatmap.png')

# ---------------------------------------------------------------- 07
G = pd.read_csv(os.path.join(VAL, 'eda6_gpd_threshold.csv'))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k, (tail, ttl) in enumerate([('left', 'LEFT tail (losses)'),
                                 ('right', 'RIGHT tail (gains)')]):
    s = G[(G['Series'] == 'rolling_std_resid_full') & (G['Tail'] == tail)]
    for c in CODES:
        d = s[s['Code'] == c]
        ax[k].errorbar(d['q'], d['xi'], yerr=d['xi_se'], lw=1.1, marker='o', ms=3,
                       capsize=2, label=c, alpha=.85)
    ax[k].axhline(0, color='k', lw=.6)
    ax[k].axvspan(0.95, 0.975, color='green', alpha=.08)
    ax[k].set_title(f'GPD shape xi vs POT threshold — {ttl}')
    ax[k].set_xlabel('threshold quantile')
    ax[k].set_ylabel('xi (shape)')
ax[0].legend(ncol=2, fontsize=7)
fig.suptitle('Peaks-over-threshold stability. xi > 0 means a heavy (Frechet) tail even AFTER '
             'volatility standardisation. Shaded = the recommended 95-97.5% region.', y=1.04)
save(fig, '07_gpd_threshold_stability.png')

# ---------------------------------------------------------------- 08
X = pd.read_csv(os.path.join(VAL, 'eda5_cross_index.csv'))
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for k, col in enumerate(['Corr_LogRV', 'Corr_Return']):
    M = X.pivot_table(index='A', columns='B', values=col).reindex(index=CODES, columns=CODES)
    im = ax[k].imshow(M.values, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax[k].set_xticks(range(6), CODES)
    ax[k].set_yticks(range(6), CODES)
    ax[k].grid(False)
    for i in range(6):
        for j in range(6):
            ax[k].text(j, i, f'{M.values[i, j]:.2f}', ha='center', va='center', fontsize=7.5,
                       color='white' if M.values[i, j] > .65 else 'black')
    ax[k].set_title('log realized variance' if k == 0 else 'daily returns')
    fig.colorbar(im, ax=ax[k], fraction=.046)
fig.suptitle('Cross-index correlation. The US and European blocks are tight; Asia is nearly '
             'uncorrelated with the US on the SAME calendar date because the sessions do not '
             'overlap.', y=1.03)
save(fig, '08_cross_index_correlation.png')

# ---------------------------------------------------------------- 09
fig, ax = plt.subplots(1, 6, figsize=(16, 3.0))
for i, c in enumerate(CODES):
    b = A[c][A[c]['InSample_B'] & A[c]['RV_Valid']].copy()
    b['LogRV_next'] = b['LogRV'].shift(-1)
    d = b.dropna(subset=['Return', 'LogRV_next'])
    x = 100 * d['Return'].values
    y = d['LogRV_next'].values
    ax[i].scatter(x, y, s=1.5, alpha=.22, color='#1f4e79')
    bins = np.quantile(x, np.linspace(0, 1, 21))
    idx = np.digitize(x, bins[1:-1])
    mx = [x[idx == k].mean() for k in range(20) if (idx == k).sum() > 5]
    my = [y[idx == k].mean() for k in range(20) if (idx == k).sum() > 5]
    ax[i].plot(mx, my, 'r-o', ms=3, lw=1.3)
    ax[i].set_title(f'{c}  corr={np.corrcoef(x, y)[0,1]:.2f}')
    ax[i].set_xlabel('return(t), %')
    ax[i].set_xlim(-6, 6)
    if i == 0:
        ax[i].set_ylabel('log RV(t+1)')
fig.suptitle('News impact: tomorrow\'s realized variance against today\'s signed return. '
             'The red binned mean is asymmetric — negative shocks raise volatility more. '
             'This is why GJR/EGARCH is required over plain GARCH.', y=1.10)
save(fig, '09_leverage_news_impact.png')

# ---------------------------------------------------------------- 10
cols = ['Return', 'RV', 'RS_neg', 'Jump', 'VolIdx', 'US10Y_pct', 'DXY_ret',
        'CreditStress', 'ParkinsonVar']
fig, ax = plt.subplots(2, 3, figsize=(15, 6))
for i, c in enumerate(CODES):
    a = A[c]
    a = a[a['Date'] >= '2011-01-01']
    m = np.vstack([a[k].notna().values.astype(float) for k in cols])
    a_ = ax[i // 3, i % 3]
    a_.imshow(m, aspect='auto', cmap='Greens', vmin=0, vmax=1,
              extent=[0, len(a), len(cols) - .5, -.5])
    a_.set_yticks(range(len(cols)), cols, fontsize=7)
    step = max(len(a) // 6, 1)
    ticks = list(range(0, len(a), step))
    a_.set_xticks(ticks, [str(a['Date'].iloc[t].year) for t in ticks], fontsize=7)
    a_.set_title(c)
    a_.grid(False)
fig.suptitle('Data availability map, 2011 onward. Green = present. The RV row for NKY shows '
             'the nulled 2016-17 block; every other series is continuous.', y=1.02)
save(fig, '10_missing_data_map.png')

print(f"\nfigures written to {FIG}")
