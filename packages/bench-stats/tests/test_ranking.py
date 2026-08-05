"""Tests for the Bradley-Terry ranking in `bench_stats.ranking`.

Checked against hand-reasoned fixed points of the MM fit (a balanced pair gives
equal strengths; a strict win-chain gives a strictly monotone order) and against
the schema/reproducibility contract (method == "bradley_terry", seed echoed,
deterministic CIs). Stdlib-only, so nothing is skipped.
"""

import pytest

from bench_stats import bradley_terry
from bench_stats.ranking import BradleyTerryRanking


def test_balanced_pair_has_equal_strengths():
    # A and B split 5-5 -> the fit's fixed point is equal, normalized to 0.5 each.
    games = [("A", "B")] * 5 + [("B", "A")] * 5
    r = bradley_terry(games, seed=0)
    assert isinstance(r, BradleyTerryRanking)
    assert r.strengths["A"] == pytest.approx(0.5)
    assert r.strengths["B"] == pytest.approx(0.5)
    assert sum(r.strengths.values()) == pytest.approx(1.0)


def test_win_chain_is_strictly_monotone():
    # A beats B and C, B beats C: a total order A > B > C with C winless.
    games = [("A", "B"), ("A", "C"), ("B", "C")]
    r = bradley_terry(games, seed=0)
    assert r.systems == ["A", "B", "C"]
    assert r.winner == "A"
    assert r.strengths["A"] > r.strengths["B"] > r.strengths["C"]
    assert r.strengths["C"] == pytest.approx(0.0, abs=1e-12)


def test_dominant_system_leads():
    games = [("A", "B")] * 9 + [("B", "A")]
    r = bradley_terry(games, seed=0)
    assert r.winner == "A"
    assert r.strengths["A"] > r.strengths["B"]


def test_strengths_sum_to_one():
    games = [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C"), ("B", "A")]
    r = bradley_terry(games, seed=1)
    assert sum(r.strengths.values()) == pytest.approx(1.0)


def test_stattest_contract_and_bounds():
    games = [("A", "B")] * 6 + [("B", "A")] * 4
    r = bradley_terry(games, seed=3)
    for name in ("A", "B"):
        t = r.tests[name]
        assert t.method == "bradley_terry"
        assert t.seed == 3
        assert t.n == 10
        assert t.statistic == pytest.approx(r.strengths[name])
        assert 0.0 <= t.ci_low <= t.ci_high <= 1.0


def test_same_seed_is_deterministic():
    games = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "B"), ("A", "B")]
    a = bradley_terry(games, seed=42)
    b = bradley_terry(games, seed=42)
    assert a.strengths == b.strengths
    for s in a.systems:
        assert (a.tests[s].ci_low, a.tests[s].ci_high) == (b.tests[s].ci_low, b.tests[s].ci_high)


def test_self_games_are_ignored():
    # A row where a system "beats itself" is degenerate and must not count.
    clean = bradley_terry([("A", "B"), ("B", "A")], seed=0)
    with_noise = bradley_terry([("A", "B"), ("A", "A"), ("B", "A")], seed=0)
    assert with_noise.n_games == clean.n_games == 2
    assert with_noise.strengths == clean.strengths


def test_empty_games_is_empty_ranking():
    r = bradley_terry([], seed=7)
    assert r.systems == []
    assert r.strengths == {}
    assert r.tests == {}
    assert r.winner is None
    assert r.n_games == 0
    assert r.seed == 7
