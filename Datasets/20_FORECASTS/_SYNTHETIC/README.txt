These files are NOT model output. They exist so Researcher B's evaluation code
(QLIKE, MSE, VaR backtests, DM, MCS) can be written and unit-tested against the
exact forecast-file contract (see 10_SCRIPTS/26_forecast_io.py) before Researcher
A's real GARCH-family forecasts exist. Do not cite or evaluate these numbers -
SigmaHat here is nothing more than a 21-day trailing return std, lagged one day.
Swap in 20_FORECASTS/<REAL_MODEL>__<CODE>_forecasts.csv as soon as they exist.
