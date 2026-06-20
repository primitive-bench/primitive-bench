"""Reranker scoring: a reordered candidate list -> ScorerOutput.

The adapter returns `reordered_ids` (the candidate ids best-first). We read each
id's graded relevance from the golden row, forming a relevance sequence in rank
order, and feed it to the BEIR/MTEB-standard metrics in `bench_stats.retrieval`
(`ndcg_at_k`, `map_at_k`, `mrr_at_k`, `hit_at_k`).

`correct` is **hit@1** — is the top-ranked candidate relevant? — a *paired binary*
outcome, exactly what the leaderboard's McNemar/Wilson separability gate consumes
(the same stance as OCR pass@test / websearch hit_rate). The richer continuous
metrics (nDCG@10, MAP, MRR) ride along in `ScorerOutput.metrics`; nDCG@10 is also
the headline `score`.

Non-attempts are uncharged (`correct=None`), never failures: an adapter error, an
empty ranking, or a row whose candidate pool contains no relevant document (so
there is nothing a reranker could have surfaced) are excluded from the denominator.
"""
from __future__ import annotations

from typing import Any

from bench_schemas import ScorerOutput
from bench_stats.retrieval import hit_at_k, map_at_k, mrr_at_k, ndcg_at_k

NDCG_K = 10


def _relevance_sequence(reordered_ids: list[str], relevance: dict[str, float]) -> list[float]:
    """Graded relevance of each returned id, in rank order (0 if unjudged/absent)."""
    return [float(relevance.get(str(rid), 0.0)) for rid in reordered_ids]


def score_rerank(item: dict[str, Any], raw: dict[str, Any]) -> ScorerOutput:
    """Score one (reranker, golden row) cell -> ScorerOutput."""
    if raw.get("error"):
        return ScorerOutput(correct=None, miss_reason="failed", rationale=str(raw["error"])[:200])

    reordered = [str(x) for x in (raw.get("reordered_ids") or [])]
    if not reordered:
        return ScorerOutput(correct=None, miss_reason="empty_ranking")

    relevance = {str(k): float(v) for k, v in (item.get("relevance") or {}).items()}
    n_relevant = int(item.get("n_relevant") or sum(1 for v in relevance.values() if v > 0))
    if n_relevant <= 0:
        # No relevant doc in the candidate pool -> nothing a reranker could surface.
        return ScorerOutput(correct=None, miss_reason="no_relevant_in_candidates")

    rels = _relevance_sequence(reordered, relevance)
    ndcg = float(ndcg_at_k(rels, NDCG_K))
    mapv = float(map_at_k(rels, len(rels), n_relevant=n_relevant))
    mrr = float(mrr_at_k(rels, len(rels)))
    hit1 = float(hit_at_k(rels, 1))
    correct = hit1 >= 1.0

    return ScorerOutput(
        correct=bool(correct),
        score=round(ndcg, 6),
        metrics={
            "ndcg@10": round(ndcg, 6),
            "map": round(mapv, 6),
            "mrr": round(mrr, 6),
            "hit@1": hit1,
        },
        miss_reason=None if correct else "top_relevant_demoted",
    )
