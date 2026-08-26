# -*- coding: utf-8 -*-
"""
Unit tests for basel_zone() in Datasets/10_SCRIPTS/49_model_comparison.py.

Regression guard for the off-by-one found in code review of PR #1
(2026-08-26): the function must classify on P(X <= n_breach), inclusive of
n_breach, matching the published Basel Committee cumulative-binomial table
for n=250, alpha=0.01 (green through 4 exceptions, amber 5-9, red 10+).
"""
import importlib.util
import os
import sys

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "Datasets", "10_SCRIPTS")
sys.path.insert(0, SCRIPTS)


def _load_basel_zone():
    spec = importlib.util.spec_from_file_location("mc49", os.path.join(SCRIPTS, "49_model_comparison.py"))
    mod = importlib.util.module_from_spec(spec)
    # 49_model_comparison.py does real work at import if run as __main__ only
    # inside main(); importing the module just defines functions, so this is safe.
    spec.loader.exec_module(mod)
    return mod.basel_zone


basel_zone = _load_basel_zone()


def test_basel_zone_official_boundaries_n250_alpha01():
    """Official Basel Committee (1996) traffic-light boundaries at n=250,
    alpha=1%: green for 0-4 exceptions, amber for 5-9, red for 10+."""
    n, alpha = 250, 0.01
    for k in range(0, 5):
        assert basel_zone(k, n, alpha) == "green", f"k={k} should be green"
    for k in range(5, 10):
        assert basel_zone(k, n, alpha) == "amber", f"k={k} should be amber"
    for k in (10, 15, 20):
        assert basel_zone(k, n, alpha) == "red", f"k={k} should be red"


def test_basel_zone_inclusive_of_n_breach():
    """The boundary itself (P(X<=n_breach)) must be evaluated, not
    P(X<=n_breach-1) -- this is exactly the bug found in review. At n=250,
    alpha=1%, k=4 is the last green cell; k=4 must not spill into amber."""
    assert basel_zone(4, 250, 0.01) == "green"
    assert basel_zone(9, 250, 0.01) == "amber"


def test_basel_zone_monotonic_in_breach_count():
    n, alpha = 3243, 0.01  # a realistic n from this project's forecast files
    order = {"green": 0, "amber": 1, "red": 2}
    prev = "green"
    for k in range(0, 60, 3):
        z = basel_zone(k, n, alpha)
        assert order[z] >= order[prev]
        prev = z
