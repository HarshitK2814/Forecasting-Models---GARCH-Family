# Unit tests

Closes the Executive Summary's "Reproducibility & Environment" item: *"Write
unit tests for core functions (e.g. GPD fit, quantile regression prediction).
Document how to run tests."*

Covers the shared statistical library (`Datasets/10_SCRIPTS/40_b_common.py`:
GPD fit/VaR/ES, Kupiec/Christoffersen/DQ backtests, QLIKE/pinball loss,
Diebold-Mariano, Model Confidence Set), the forecast-file contract
(`26_forecast_io.py`'s `validate()` — the single interface between
Researcher A and B), and the Basel traffic-light zone function in
`49_model_comparison.py` (added as a regression guard for the off-by-one
found in code review of PR #1, 2026-08-26).

Run all tests:
```
pip install pytest
pytest tests/ -v
```

These are function-level tests against synthetic/known-parameter inputs —
they do not re-run any GARCH fit or re-download data, so they finish in
seconds and can be run on every commit.
