"""Unit tests for reranker scoring (pure; reuses bench_stats.retrieval metrics)."""
from __future__ import annotations

import pytest

from eval_reranker.scoring import score_rerank

# One relevant candidate ("a") among three.
ITEM = {"relevance": {"a": 1, "b": 0, "c": 0}, "n_relevant": 1}


def test_perfect_reorder_scores_ndcg_one_and_hits():
    out = score_rerank(ITEM, {"reordered_ids": ["a", "b", "c"]})
    assert out.correct is True
    assert out.score == 1.0
    assert out.metrics["ndcg@10"] == 1.0
    assert out.metrics["hit@1"] == 1.0


def test_relevant_demoted_is_a_charged_miss():
    out = score_rerank(ITEM, {"reordered_ids": ["b", "c", "a"]})
    assert out.correct is False
    assert out.miss_reason == "top_relevant_demoted"
    assert out.metrics["ndcg@10"] < 1.0
    assert out.metrics["mrr"] == pytest.approx(1 / 3, abs=1e-6)


def test_no_relevant_in_candidates_is_uncharged():
    out = score_rerank({"relevance": {"a": 0, "b": 0}, "n_relevant": 0},
                       {"reordered_ids": ["a", "b"]})
    assert out.correct is None and out.miss_reason == "no_relevant_in_candidates"


def test_adapter_error_is_uncharged():
    out = score_rerank(ITEM, {"error": "boom"})
    assert out.correct is None and out.miss_reason == "failed"


def test_empty_ranking_is_uncharged():
    out = score_rerank(ITEM, {"reordered_ids": []})
    assert out.correct is None and out.miss_reason == "empty_ranking"


def test_reports_all_continuous_metrics():
    out = score_rerank(ITEM, {"reordered_ids": ["a", "b", "c"]})
    assert set(out.metrics) == {"ndcg@10", "map", "mrr", "hit@1"}


def test_n_relevant_inferred_when_absent():
    # n_relevant omitted -> inferred from the relevance map (two positives here).
    item = {"relevance": {"a": 1, "b": 1, "c": 0}}
    out = score_rerank(item, {"reordered_ids": ["c", "a", "b"]})
    assert out.correct is False  # top is the irrelevant "c"
    assert out.metrics["map"] == pytest.approx((0.5 + 2 / 3) / 2, abs=1e-6)
