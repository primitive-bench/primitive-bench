"""Retrieval scoring: a ranked candidate list -> ScorerOutput.

The adapter returns `retrieved_ids` (the candidate ids best-first, after ranking the
row's per-query candidate pool). We read each id's graded relevance from the golden
row, forming a relevance sequence in rank order, and feed it to the BEIR/MTEB-standard
metrics in `bench_stats.retrieval` (`ndcg_at_k`, `map_at_k`, `mrr_at_k`, `hit_at_k`).

`correct` is **success@10** — is *any* relevant document in the top-10? — a *paired
binary* outcome, exactly what the leaderboard's McNemar/Wilson separability gate
consumes (the same stance as OCR pass@test / websearch hit_rate). Unlike reranker
(which gates on hit@1, "is the single best candidate on top?"), first-stage retrieval
is judged on getting relevant docs *into* the top-k. The richer continuous metrics
(nDCG@10, recall@10, MAP, MRR) ride along in `ScorerOutput.metrics`; nDCG@10 is also
the headline `score`.

Non-attempts are uncharged (`correct=None`), never failures: an adapter error, an
empty ranking, or a row whose candidate pool contains no relevant document (so there
is nothing retrieval could have surfaced) are excluded from the denominator.
"""
from __future__ import annotations

from typing import Any

from bench_schemas import ScorerOutput
from bench_stats.retrieval import hit_at_k, map_at_k, mrr_at_k, ndcg_at_k

K = 10  # top-k for the success@k gate, nDCG@k and recall@k


def _relevance_sequence(retrieved_ids: list[str], relevance: dict[str, float]) -> list[float]:
    """Graded relevance of each returned id, in rank order (0 if unjudged/absent)."""
    return [float(relevance.get(str(rid), 0.0)) for rid in retrieved_ids]


def _recall_at_k(rels: list[float], n_relevant: int, k: int) -> float:
    """Fraction of the query's relevant docs that appear in the top-k ranking."""
    if n_relevant <= 0:
        return 0.0
    found = sum(1 for r in rels[:k] if r > 0)
    return found / n_relevant


def score_retrieval(item: dict[str, Any], raw: dict[str, Any]) -> ScorerOutput:
    """Score one (retriever, golden row) cell -> ScorerOutput."""
    if raw.get("error"):
        return ScorerOutput(correct=None, miss_reason="failed", rationale=str(raw["error"])[:200])

    retrieved = [str(x) for x in (raw.get("retrieved_ids") or [])]
    if not retrieved:
        return ScorerOutput(correct=None, miss_reason="empty_ranking")

    relevance = {str(k): float(v) for k, v in (item.get("relevance") or {}).items()}
    n_relevant = int(item.get("n_relevant") or sum(1 for v in relevance.values() if v > 0))
    if n_relevant <= 0:
        # No relevant doc in the candidate pool -> nothing retrieval could surface.
        return ScorerOutput(correct=None, miss_reason="no_relevant_in_pool")

    rels = _relevance_sequence(retrieved, relevance)
    ndcg = float(ndcg_at_k(rels, K))
    recall = float(_recall_at_k(rels, n_relevant, K))
    mapv = float(map_at_k(rels, len(rels), n_relevant=n_relevant))
    mrr = float(mrr_at_k(rels, len(rels)))
    success = float(hit_at_k(rels, K))
    correct = success >= 1.0

    return ScorerOutput(
        correct=bool(correct),
        score=round(ndcg, 6),
        metrics={
            "ndcg@10": round(ndcg, 6),
            "recall@10": round(recall, 6),
            "map": round(mapv, 6),
            "mrr": round(mrr, 6),
        },
        miss_reason=None if correct else "relevant_outside_top10",
    )
