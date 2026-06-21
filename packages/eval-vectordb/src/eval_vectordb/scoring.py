"""Vectordb scoring: one (engine, query) result -> ScorerOutput.

The engine returns `returned_ids` (its top-k base indices, best first). The golden
row carries `neighbors` — the exact top-K nearest neighbors (computed by brute force,
`GroundTruthTier.VERIFIED_EXTERNAL`). Recall@K = |returned ∩ truth| / K.

THE SEPARABILITY SURFACE IS A PROPORTION. Per query, recall is a *fraction* (hits/K),
not a 0/1 — so this scorer carries the integer `hits` and `k` in `metrics`, and the
vectordb `report._collect` aggregates `k=Σhits, n=ΣK` across queries per slice to feed
the Wilson/leaderboard pipeline. **Do NOT binarize via `correct`** (that would collapse
fractional recall). `correct = (hits == K)` is set only as a coarse "perfect-recall?"
binary for the run-dir view; it is never the published count.

Continuous companions (QPS, p50/p99 latency, build time, index memory, $cost) ride in
`metrics` for the report's summary — the recall-vs-cost story ANN-Benchmarks /
VectorDBBench tell — but never enter the recall counts.

Non-attempts are uncharged (`correct=None`): a build failure, timeout, OOM, or adapter
error means the engine never had a chance at the query, so it is excluded from the
denominator (the search/extract/reranker stance).
"""
from __future__ import annotations

from typing import Any

from bench_schemas import ScorerOutput

RECALL_K = 10

# Query/build-time failure reasons that uncharge the lane (correct=None).
UNCHARGED_REASONS = {"adapter_error", "timeout", "oom", "build_failed", "no_ground_truth"}


def score_query(item: dict[str, Any], raw: dict[str, Any]) -> ScorerOutput:
    """Score one (engine, query) cell -> ScorerOutput (recall@K + companions)."""
    if raw.get("error"):
        reason = raw.get("miss_reason") or "adapter_error"
        return ScorerOutput(correct=None, miss_reason=reason, rationale=str(raw["error"])[:200])

    truth = [int(x) for x in (item.get("neighbors") or [])]
    k = len(truth) if truth else int(item.get("k") or RECALL_K)
    if not truth:
        return ScorerOutput(correct=None, miss_reason="no_ground_truth")

    returned = [int(x) for x in (raw.get("returned_ids") or [])][:k]
    hits = len(set(returned) & set(truth))
    recall = hits / k

    engine_metrics = raw.get("engine_metrics") or {}
    metrics: dict[str, float] = {
        "recall@10": round(recall, 6),
        "hits": float(hits),
        "k": float(k),
        "latency_ms": float(raw.get("latency_ms") or 0.0),
    }
    for key in ("qps", "build_s", "index_mb", "p50_ms", "p99_ms"):
        if engine_metrics.get(key) is not None:
            metrics[key] = float(engine_metrics[key])
    if raw.get("cost_usd") is not None:
        metrics["cost_usd"] = float(raw["cost_usd"])

    if hits == k:
        miss_reason = None
    elif not returned:
        miss_reason = "empty_result"
    else:
        miss_reason = "missed_recall"

    return ScorerOutput(
        correct=(hits == k),  # coarse "perfect recall?" — run-dir view only, NOT the count
        score=round(recall, 6),
        metrics=metrics,
        miss_reason=miss_reason,
    )
