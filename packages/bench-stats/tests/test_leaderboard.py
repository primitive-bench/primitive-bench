"""Tests for the leaderboard derivation (winner / TIE band / thin / saturated)."""

from __future__ import annotations

from bench_schemas import SliceResult
from bench_schemas.models import Primitive

from bench_stats.leaderboard import (
    PrimitiveReport,
    build_primitive_report,
    build_slice_report,
)


def _report(slice_key, counts, **kw):
    return build_slice_report(Primitive.WEBSEARCH, "run", slice_key, counts,
                              metric_name="hit_rate", **kw)


def test_sole_winner_when_interval_clears_runnerup():
    # exa 88% vs everyone at 0% (company_lookup) -> exa is the sole winner.
    r = _report("company_lookup",
                [("exa", 22, 25), ("serpapi", 0, 25), ("brave", 0, 25), ("tavily", 0, 25)])
    assert r.winner == "exa"
    assert r.band == ["exa"]
    assert r.status == "published"
    assert r.thin is False and r.saturated is False
    # leader carries the separability verdict; runners-up do not.
    leader = next(x for x in r.results if x.rank == 1)
    assert leader.adapter == "exa" and leader.separable is True


def test_tie_band_when_intervals_overlap():
    # government_registry: exa 76.7%, serpapi 66.7% overlap -> TIE; brave/tavily excluded.
    r = _report("government_registry",
                [("exa", 69, 90), ("serpapi", 60, 90), ("brave", 44, 90), ("tavily", 36, 90)])
    assert r.winner is None
    assert r.band == ["exa", "serpapi"]
    assert "brave" not in r.band and "tavily" not in r.band
    leader = next(x for x in r.results if x.rank == 1)
    assert leader.separable is False  # tied at the top


def test_thin_slice_is_flagged_not_called():
    # n=1 must never produce a real winner.
    r = _report("b2b_tools",
                [("brave", 1, 1), ("exa", 1, 1), ("serpapi", 1, 1), ("tavily", 1, 1)])
    assert r.thin is True
    assert r.status == "thin_data"
    assert r.saturated is False


def test_saturated_slice_is_flagged():
    # Everyone ~100% at healthy n -> saturated (rotation candidate).
    r = build_slice_report(Primitive.EXTRACTION, "run", "sec_edgar",
                           [("firecrawl", 25, 25), ("jina", 25, 25),
                            ("tavily", 25, 25), ("exa_live", 25, 25)],
                           metric_name="token_survival")
    assert r.saturated is True
    assert r.status == "saturated"


def test_separated_loser_drops_out_of_band():
    # fed_register: top three ~100% TIE; exa_live (blocked, 0%) separates out below.
    r = build_slice_report(Primitive.EXTRACTION, "run", "fed_register",
                           [("firecrawl", 100, 100), ("jina", 99, 99),
                            ("tavily", 100, 100), ("exa_live", 0, 100)],
                           metric_name="token_survival")
    assert r.winner is None
    assert set(r.band) == {"firecrawl", "jina", "tavily"}
    assert "exa_live" not in r.band
    exa = next(x for x in r.results if x.adapter == "exa_live")
    assert exa.rank == 4 and exa.point_estimate == 0.0


def test_ranks_are_contiguous_and_results_are_sliceresults():
    r = _report("technical_docs",
                [("exa", 25, 25), ("serpapi", 23, 25), ("tavily", 23, 25), ("brave", 18, 25)])
    assert [x.rank for x in r.results] == [1, 2, 3, 4]
    assert all(isinstance(x, SliceResult) for x in r.results)


def test_primitive_report_round_trips_through_json():
    rep = build_primitive_report(
        Primitive.WEBSEARCH, "run",
        {"company_lookup": [("exa", 22, 25), ("brave", 0, 25)]},
        metric_name="hit_rate", citation="report.md",
    )
    again = PrimitiveReport.model_validate(rep.model_dump(mode="json"))
    assert again.primitive == Primitive.WEBSEARCH
    assert again.slices[0].winner == "exa"
