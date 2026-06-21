"""Unit tests for chunking scoring + range algebra (pure; no model/network)."""
from __future__ import annotations

import pytest

from eval_chunking.ranges import (
    intersection,
    measure_intersection,
    sum_of_ranges,
    union_ranges,
)
from eval_chunking.scoring import RECALL_TARGET, precision_omega, score_chunking

# A query whose single gold reference is the span [100, 200).
ITEM = {"references": [{"content": "x", "start_index": 100, "end_index": 200}]}


def _chunk(start: int, end: int) -> dict:
    return {"text": "x", "start": start, "end": end}


# --- range algebra ---------------------------------------------------------- #

def test_union_merges_overlapping_and_touching():
    assert union_ranges([(0, 10), (5, 15), (15, 20)]) == [(0, 20)]
    assert union_ranges([(0, 5), (10, 15)]) == [(0, 5), (10, 15)]


def test_intersection_and_measure():
    assert intersection([(0, 10)], [(5, 20)]) == [(5, 10)]
    assert measure_intersection([(0, 10), (20, 30)], [(5, 25)]) == 10  # [5,10)+[20,25)


def test_sum_of_ranges_assumes_disjoint():
    assert sum_of_ranges([(0, 10), (20, 25)]) == 15


# --- score_chunking --------------------------------------------------------- #

def test_full_coverage_is_a_hit_recall_one():
    out = score_chunking(ITEM, [_chunk(100, 200)])
    assert out.correct is True
    assert out.score == 1.0
    assert out.metrics["recall"] == 1.0
    assert out.metrics["full_recall"] == 1.0


def test_precision_penalizes_oversized_chunk():
    # One chunk [0,200) fully covers the gold [100,200): recall 1, precision 0.5.
    out = score_chunking(ITEM, [_chunk(0, 200)])
    assert out.correct is True
    assert out.metrics["recall"] == 1.0
    assert out.metrics["precision"] == pytest.approx(0.5, abs=1e-6)
    assert out.metrics["iou"] == pytest.approx(0.5, abs=1e-6)


def test_partial_recall_below_target_is_a_charged_miss():
    # Recovers [100,150) of [100,200): recall 0.5 < target -> miss.
    out = score_chunking(ITEM, [_chunk(100, 150)])
    assert out.correct is False
    assert out.metrics["recall"] == pytest.approx(0.5, abs=1e-6)
    assert out.miss_reason == "partial_recall"


def test_recall_target_boundary_is_a_hit():
    out = score_chunking(ITEM, [_chunk(100, 100 + int(RECALL_TARGET * 100))])
    assert out.metrics["recall"] == pytest.approx(RECALL_TARGET, abs=1e-6)
    assert out.correct is True


def test_no_overlap_is_answer_not_retrieved():
    out = score_chunking(ITEM, [_chunk(0, 50)])
    assert out.correct is False
    assert out.metrics["recall"] == 0.0
    assert out.miss_reason == "answer_not_retrieved"


def test_empty_retrieved_is_uncharged():
    out = score_chunking(ITEM, [])
    assert out.correct is None and out.miss_reason == "empty_chunking"


def test_chunker_error_is_uncharged():
    out = score_chunking(ITEM, [_chunk(100, 200)], error="boom")
    assert out.correct is None and out.miss_reason == "failed"


def test_zero_measure_references_is_uncharged():
    out = score_chunking({"references": [{"start_index": 5, "end_index": 5}]}, [_chunk(0, 10)])
    assert out.correct is None and out.miss_reason == "references_unresolved"


def test_multi_span_recall_is_fraction_recovered():
    item = {"references": [
        {"start_index": 0, "end_index": 100},
        {"start_index": 200, "end_index": 300},
    ]}
    # Retrieve only the first span -> 100/200 recovered.
    out = score_chunking(item, [_chunk(0, 100)])
    assert out.metrics["recall"] == pytest.approx(0.5, abs=1e-6)
    assert out.correct is False


def test_precision_omega_ceiling_from_all_chunks():
    # All chunks tile [0,300) in 100-wide pieces; gold is [100,200) (one whole chunk).
    all_chunks = [_chunk(0, 100), _chunk(100, 200), _chunk(200, 300)]
    # Only the middle chunk touches the gold -> ceiling precision = 100/100 = 1.0.
    assert precision_omega(all_chunks, ITEM) == pytest.approx(1.0, abs=1e-6)
    out = score_chunking(ITEM, [_chunk(100, 200)], all_chunks=all_chunks)
    assert out.metrics["precision_omega"] == pytest.approx(1.0, abs=1e-6)
