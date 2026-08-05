"""Tests for the seeded bootstrap CI (bench_stats.resampling.bootstrap_ci).

The bootstrap is the CI engine for non-proportion metrics, so the load-bearing
property is reproducibility: the same seed must produce the same interval. Also
checks the empty-input contract, that the interval brackets the point estimate,
and that a custom statistic is honored.
"""

import pytest

from bench_stats.resampling import bootstrap_ci


def test_empty_values_returns_seeded_empty():
    t = bootstrap_ci([], seed=42)
    assert t.method == "bootstrap"
    assert t.n == 0
    assert t.seed == 42
    assert t.ci_low is None and t.ci_high is None


def test_same_seed_is_reproducible():
    values = [0.1, 0.4, 0.4, 0.7, 0.9, 0.2, 0.55]
    a = bootstrap_ci(values, seed=7)
    b = bootstrap_ci(values, seed=7)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert a.seed == b.seed == 7


def test_interval_brackets_point_estimate():
    values = [0.2, 0.3, 0.5, 0.6, 0.8]
    t = bootstrap_ci(values, seed=1)
    assert t.n == 5
    assert t.ci_low <= t.statistic <= t.ci_high


def test_constant_values_collapse_the_interval():
    # Every resample of a constant list has the same mean, so the CI is degenerate.
    t = bootstrap_ci([0.5, 0.5, 0.5], seed=3)
    assert t.statistic == pytest.approx(0.5)
    assert t.ci_low == pytest.approx(0.5)
    assert t.ci_high == pytest.approx(0.5)


def test_custom_statistic_is_honored():
    # The default statistic is the mean; passing max uses the max instead.
    t = bootstrap_ci([0.1, 0.9, 0.5], seed=2, statistic=max)
    assert t.statistic == pytest.approx(0.9)
    assert t.ci_low <= t.statistic <= t.ci_high
