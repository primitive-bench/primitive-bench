"""Unit tests for vectordb scoring (pure recall@10 with the hits/k decomposition)."""
from __future__ import annotations

from eval_vectordb.scoring import score_query

ITEM = {"neighbors": list(range(10)), "k": 10}


def test_perfect_recall_scores_one_and_is_correct():
    out = score_query(ITEM, {"returned_ids": list(range(10)), "latency_ms": 1.0})
    assert out.correct is True
    assert out.score == 1.0
    assert out.metrics["recall@10"] == 1.0
    assert out.metrics["hits"] == 10.0 and out.metrics["k"] == 10.0
    assert out.miss_reason is None


def test_partial_recall_is_a_charged_miss():
    returned = list(range(7)) + [100, 101, 102]  # 7 true neighbors + 3 wrong
    out = score_query(ITEM, {"returned_ids": returned})
    assert out.correct is False
    assert out.metrics["recall@10"] == 0.7
    assert out.metrics["hits"] == 7.0
    assert out.miss_reason == "missed_recall"


def test_extra_returned_beyond_k_are_ignored():
    # engine returns 20 ids; only the top-10 count toward recall@10.
    out = score_query(ITEM, {"returned_ids": list(range(10)) + list(range(50, 60))})
    assert out.metrics["recall@10"] == 1.0


def test_empty_result_is_charged():
    out = score_query(ITEM, {"returned_ids": []})
    assert out.correct is False and out.miss_reason == "empty_result"
    assert out.metrics["recall@10"] == 0.0


def test_adapter_error_is_uncharged():
    out = score_query(ITEM, {"error": "boom"})
    assert out.correct is None and out.miss_reason == "adapter_error"


def test_build_failure_reason_passes_through_uncharged():
    out = score_query(ITEM, {"error": "out of memory", "miss_reason": "oom"})
    assert out.correct is None and out.miss_reason == "oom"


def test_no_ground_truth_is_uncharged():
    out = score_query({"neighbors": []}, {"returned_ids": [1, 2, 3]})
    assert out.correct is None and out.miss_reason == "no_ground_truth"


def test_continuous_companions_pass_through():
    out = score_query(ITEM, {
        "returned_ids": list(range(10)),
        "latency_ms": 2.5,
        "cost_usd": 0.001,
        "engine_metrics": {"qps": 100.0, "build_s": 0.5, "index_mb": 12.0},
    })
    assert out.metrics["latency_ms"] == 2.5
    assert out.metrics["cost_usd"] == 0.001
    assert out.metrics["qps"] == 100.0
    assert out.metrics["build_s"] == 0.5
    assert out.metrics["index_mb"] == 12.0
