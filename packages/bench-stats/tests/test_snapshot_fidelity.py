"""Fidelity guard: the curated snapshot counts must reproduce the PUBLISHED rates.

The expected values below are read off the published reports
(packages/eval-websearch/reports/{slices-search,snapshot-v2-analysis}.md), NOT the
TOMLs — so a transcription typo in a *.counts.toml diverges from these and fails CI.
This is the cheap provenance guard while the counts are still hand-curated.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from bench_schemas.models import Primitive

from bench_stats.leaderboard import PrimitiveReport, build_primitive_report

ROOT = Path(__file__).resolve().parents[3]


def _load(rel: str) -> PrimitiveReport:
    spec = tomllib.loads((ROOT / rel).read_text())
    slices_raw = {k: [tuple(r) for r in v] for k, v in spec["slices"].items()}
    return build_primitive_report(
        primitive=Primitive(spec["primitive"]),
        run_id=spec["run_id"],
        slices_raw=slices_raw,
        metric_name=spec["metric_name"],
        citation=spec.get("citation"),
        as_of=spec.get("as_of"),
    )


@pytest.fixture(scope="module")
def websearch() -> PrimitiveReport:
    return _load("packages/eval-websearch/snapshots/websearch.counts.toml")


@pytest.fixture(scope="module")
def extraction() -> PrimitiveReport:
    return _load("packages/eval-extraction/snapshots/extraction.counts.toml")


def _slice(rep: PrimitiveReport, key: str):
    return next(s for s in rep.slices if s.slice == key)


def _point(rep: PrimitiveReport, slice_key: str, adapter: str) -> float:
    return next(r.point_estimate for r in _slice(rep, slice_key).results if r.adapter == adapter)


# --- websearch (reports/slices-search.md) ----------------------------------- #

def test_websearch_published_rates(websearch):
    assert _point(websearch, "company_lookup", "exa") == pytest.approx(0.880, abs=0.006)
    assert _point(websearch, "government_registry", "exa") == pytest.approx(0.767, abs=0.006)
    assert _point(websearch, "government_registry", "serpapi") == pytest.approx(0.667, abs=0.006)
    assert _point(websearch, "technical_docs", "exa") == pytest.approx(1.000, abs=0.006)
    assert _point(websearch, "citation_needed", "exa") == pytest.approx(0.767, abs=0.006)


def test_websearch_published_calls(websearch):
    assert _slice(websearch, "company_lookup").winner == "exa"
    assert _slice(websearch, "government_registry").winner is None
    assert _slice(websearch, "government_registry").band[:2] == ["exa", "serpapi"]
    assert _slice(websearch, "b2b_tools").thin is True


# --- extraction (reports/snapshot-v2-analysis.md) --------------------------- #

def test_extraction_published_rates(extraction):
    assert _point(extraction, "fed_register", "firecrawl") == pytest.approx(1.000, abs=0.006)
    assert _point(extraction, "fed_register", "exa_live") == pytest.approx(0.000, abs=0.006)
    assert _point(extraction, "nvd_cve", "tavily") == pytest.approx(0.960, abs=0.006)


def test_extraction_published_calls(extraction):
    assert "exa_live" not in _slice(extraction, "fed_register").band
    assert _slice(extraction, "sec_edgar").saturated is True


def test_as_of_is_threaded(websearch, extraction):
    assert websearch.as_of == "2026-06"
    assert extraction.as_of == "2026-06"
