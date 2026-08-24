# Researcher A — modelling decisions

**Purpose.** Three decisions the Executive Summary assigns to Researcher A and requires to be
settled *before* Researcher B's cross-index work (crisis/regime analysis, DM, MCS) can start.
Each is a choice, not a fact — stated here so it is made once and applied identically by both
of us. If you disagree with one, say so before building on it; changing it after B has written
code against it is expensive.

---

## 1. Non-synchronous sessions

**The problem.** Tokyo and Hong Kong close (07:00–08:00 UTC) before New York opens
(14:30 UTC). A US shock on trading day *t* cannot reach the Asian close until day *t+1*. The
EDA measured same-calendar-date return correlation between Asia and the US at **~0.18** — a
timing artefact of the closing-time mismatch, not weak economic linkage. (Contrast: SPX–NDX
same-day correlation is 0.96, because both close at the same instant.)

**Decision.** All six indices in the analysis files use their **own local exchange calendar**
— `Date` is the local trading day, `Close` the local close. No shift is applied inside
`01_ANALYSIS_READY/`. This is deliberate: shifting would corrupt each index's own
autocorrelation structure (its own GARCH cares about its own calendar, not New York's).

**What this means for each model:**

- **Per-index models** (baseline GARCH, GJR/EGARCH, Realized GARCH, GARCH-EVT, quantile
  regression estimated separately per index): **no action needed.** Each index's own calendar
  is already correct for its own dynamics.
- **Any pooled or cross-index result** (contagion, lead-lag, spillover, a regression that
  puts two indices' returns on the same row and calls it "same day"): **must lag Asia by one
  session relative to the US/Europe close**, i.e. compare `US_Return[t]` against
  `Asia_Return[t+1]`, not `Asia_Return[t]`. Report same-day correlations only as evidence of
  the timing artefact, never as a spillover estimate.
- **Model Confidence Set / pooled Diebold-Mariano** across indices: build the loss series
  per index first (each on its own calendar), then align by `CommonDate_B` for the pooled
  test. `CommonDate_B` is a shared trading-day flag, not a same-instant flag — see §3.

**Do not** attempt to build one single "world clock" and resample every index onto it. It
was considered and rejected: it either drops real local-holiday variation or interpolates
prices that were never traded, either of which is worse than stating the convention above.

---

## 2. Forecast-file schema

See `10_SCRIPTS/26_forecast_io.py` — the schema is enforced in code, not just described here,
so a malformed file cannot silently reach a backtest. Read the module docstring for the full
rationale. Summary of the load-bearing conventions:

| Rule | Statement |
|---|---|
| Units | Every return-space number is a **decimal log return** (0.01, not "1%"). |
| Scale | `SigmaHat` is the conditional SD of the **close-to-close** daily return — not session-only. Realized GARCH fitted on raw session RV must be rescaled before it lands in a forecast file (see `ScaleFactor_HL`, and precaution below). |
| VaR sign | VaR is the **signed, negative** quantile. `VaR_01 <= VaR_025 <= VaR_05 < 0`. A breach is `Realized < VaR_01`. |
| Date semantics | `Date` = day being forecast. `OriginDate` = last day of information used. `OriginDate < Date` always — this is checked and raises on violation. |
| Missing forecasts | Rows for undeliverable forecasts (e.g. NKY 2016–17 under Realized GARCH) are **kept**, `Valid=False`, `Reason` filled. Never delete the row — deleting breaks paired-length tests (DM) silently. |
| Evaluation window | Rows span `InSample_B`. Pooled tests additionally restrict to `CommonDate_B`; use `BalancedRV_B` when the loss series itself needs realized variance across all six indices simultaneously. |
| One file per model per index | `20_FORECASTS/<MODEL>__<CODE>_forecasts.csv` |

`26_forecast_io.py` provides `write_forecasts()` (A calls this) and `read_forecasts()` /
`eval_frame()` (B calls these). `eval_frame()` joins a forecast file to the actuals and
returns QLIKE, squared error, and VaR breach indicators already computed the same way for
every model — so two models are never scored by two slightly different formulas.

Synthetic placeholder files are in `20_FORECASTS/_SYNTHETIC/` in the exact contract format,
so B's evaluation code can be written and unit-tested against them before A's real GARCH
output exists.

---

## 3. Sample and crisis-coverage statement

**Sample B is primary**: all six indices, `2013-09-30 → 2026-08-21`, `2,685` common trading
days (`CommonDate_B`), binding on DAX's later realized-volatility start. `BalancedRV_B`
(`1,994` days) is a stricter subset for pooled statistics that need realized volatility valid
on **every** index simultaneously — use it only where that is required (e.g. a pooled QLIKE
average across indices); using it everywhere discards 691 good index-days for no reason.

**Crisis coverage — the fact that must be disclosed in the paper.** Ten named crisis windows
are labelled in `CrisisLabel` (`01_ANALYSIS_READY/CRISIS_PERIODS.csv`). Sample B's start date
of 2013-09-30 means it captures only:

| Crisis | In sample B? |
|---|---|
| Asian Crisis / LTCM (1997) | **No** |
| DotCom Bust (2000–02) | **No** |
| Global Financial Crisis (2007–09) | **No** |
| Euro Sovereign Debt (2010–12) | **No** |
| China Deval / Oil (2015–16) | Yes — 128 days |
| Volmageddon (Feb 2018) | Yes — 9 days |
| Q4 2018 Selloff | Yes — 60 days |
| COVID Crash (2020) | Yes — 50 days |
| Rate Shock (2022) | Yes — 196 days |
| Yen Carry Unwind (Aug 2024) | Yes — 7 days |

**Consequence for the paper.** GARCH-EVT and quantile regression are estimated on daily data
back to 1990 and therefore see all ten crises in estimation, even though their *out-of-sample
evaluation window* is restricted to sample B for comparability with Realized GARCH (which
needs intraday RV and cannot start before 2011, DAX 2013). **State explicitly in the paper
that the three model families see different crisis histories in estimation, and that the
out-of-sample comparison itself only evaluates on the four post-2013 stress episodes** (China
deval, Volmageddon, Q4 2018, COVID, 2022 rate shock, yen carry — 450 of 2,685 days, ~17%).
Do not claim the models were "tested through the GFC" — they were not, on sample B.

If a reviewer specifically wants pre-2013 crisis performance, that can only come from a
daily-only robustness table (GARCH-EVT / QR on the full 1990+ history, no Realized GARCH
comparator) — flagged as a possible robustness-check addition, not built by default.

---

*Owner: Researcher A. Last updated 2026-08-24. Changes here must be reflected in
`10_SCRIPTS/26_forecast_io.py` and `Handoff_to_Researcher_B.pdf` if they alter either.*
