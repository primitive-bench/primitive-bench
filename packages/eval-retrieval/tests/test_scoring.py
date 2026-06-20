"""Unit tests for retrieval scoring (pure; reuses bench_stats.retrieval metrics)."""
from __future__ import annotations

import pytest

from eval_retrieval.scoring import score_retrieval

# One relevant candidate ("a") among three (small pool: any order keeps it in top-10).
ITEM = {"relevance": {"a": 1, "b": 0, "c": 0}, "n_relevant": 1}


def test_perfect_ranking_scores_ndcg_one_and_success():
    out = score_retrieval(ITEM, {"retrieved_ids": ["a", "b", "c"]})
    assert out.correct is True
    assert out.score == 1.0
    assert out.metrics["ndcg@10"] == 1.0
    assert out.metrics["recall@10"] == 1.0


def test_relevant_within_top10_is_still_success():
    # 3-candidate pool: even the worst order keeps the relevant doc in the top-10.
    out = score_retrieval(ITEM, {"retrieved_ids": ["b", "c", "a"]})
    assert out.correct is True
    assert out.metrics["mrr"] == pytest.approx(1 / 3, abs=1e-6)


def test_relevant_outside_top10_is_a_charged_miss():
    # 12-candidate pool, single relevant doc ranked last (rank 12) -> outside top-10.
    rel = {f"d{i}": (1 if i == 11 else 0) for i in range(12)}
    item = {"relevance": rel, "n_relevant": 1}
    order = [f"d{i}" for i in range(12)]
    out = score_retrieval(item, {"retrieved_ids": order})
    assert out.correct is False
    assert out.miss_reason == "relevant_outside_top10"
    assert out.metrics["recall@10"] == 0.0


def test_recall_partial_when_one_relevant_beyond_k():
    # 12 candidates, relevant at rank 1 and rank 12 -> recall@10 = 1/2 but success@10 true.
    rel = {f"d{i}": (1 if i in (0, 11) else 0) for i in range(12)}
    item = {"relevance": rel, "n_relevant": 2}
    order = [f"d{i}" for i in range(12)]
    out = score_retrieval(item, {"retrieved_ids": order})
    assert out.correct is True
    assert out.metrics["recall@10"] == pytest.approx(0.5, abs=1e-6)


def test_no_relevant_in_pool_is_uncharged():
    out = score_retrieval({"relevance": {"a": 0, "b": 0}, "n_relevant": 0},
                          {"retrieved_ids": ["a", "b"]})
    assert out.correct is None and out.miss_reason == "no_relevant_in_pool"


def test_adapter_error_is_uncharged():
    out = score_retrieval(ITEM, {"error": "boom"})
    assert out.correct is None and out.miss_reason == "failed"


def test_empty_ranking_is_uncharged():
    out = score_retrieval(ITEM, {"retrieved_ids": []})
    assert out.correct is None and out.miss_reason == "empty_ranking"


def test_reports_all_continuous_metrics():
    out = score_retrieval(ITEM, {"retrieved_ids": ["a", "b", "c"]})
    assert set(out.metrics) == {"ndcg@10", "recall@10", "map", "mrr"}


def test_map_for_multi_relevant():
    # two relevant; ranked at positions 1 and 3 -> AP = (1/1 + 2/3) / 2.
    item = {"relevance": {"a": 1, "b": 1, "c": 0}, "n_relevant": 2}
    out = score_retrieval(item, {"retrieved_ids": ["a", "c", "b"]})
    assert out.metrics["map"] == pytest.approx((1.0 + 2 / 3) / 2, abs=1e-6)


def test_n_relevant_inferred_when_absent():
    item = {"relevance": {"a": 1, "b": 1, "c": 0}}  # n_relevant omitted -> inferred = 2
    out = score_retrieval(item, {"retrieved_ids": ["a", "b", "c"]})
    assert out.correct is True
    assert out.metrics["recall@10"] == 1.0
