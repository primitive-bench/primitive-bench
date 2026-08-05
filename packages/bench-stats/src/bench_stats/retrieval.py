"""BEIR/MTEB-standard IR metrics for the search/retrieval/reranker primitives.

`relevances` is the graded relevance of each returned item in rank order
(0 = irrelevant). `n_relevant` is the total number of relevant items in the
corpus for the query — the denominator for recall-style metrics, NOT the number
returned. An item counts as relevant iff its graded relevance is > 0, matching
the existing rank metrics here and the pytrec_eval / TREC convention.

Rank-position metrics (need only the returned `relevances`):
  hit_at_k / mrr_at_k / ndcg_at_k / precision_at_k

Set-recall metrics (need `n_relevant`, the corpus total):
  recall_at_k / map_at_k / r_precision

The split matters for honesty: a system cannot be rewarded for relevant items it
never surfaced, so recall@k, MAP@k, and R-precision divide by the corpus total —
never by what came back.
"""

from __future__ import annotations

import math
from typing import Sequence


def hit_at_k(relevances: Sequence[float], k: int) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0."""
    return 1.0 if any(r > 0 for r in relevances[:k]) else 0.0


def mrr_at_k(relevances: Sequence[float], k: int) -> float:
    """Reciprocal rank of the first relevant item within top-k."""
    for i, r in enumerate(relevances[:k], start=1):
        if r > 0:
            return 1.0 / i
    return 0.0


def _dcg(relevances: Sequence[float], k: int) -> float:
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(relevances[:k], start=1))


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Normalized DCG@k against the ideal ranking of the same relevances."""
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg(ideal, k)
    return _dcg(relevances, k) / idcg if idcg > 0 else 0.0


def map_at_k(relevances: Sequence[float], k: int, n_relevant: int | None = None) -> float:
    """Average precision @k (binary relevance: rel>0)."""
    hits = 0
    precision_sum = 0.0
    for i, r in enumerate(relevances[:k], start=1):
        if r > 0:
            hits += 1
            precision_sum += hits / i
    denom = n_relevant if n_relevant is not None else hits
    return precision_sum / denom if denom > 0 else 0.0


def precision_at_k(relevances: Sequence[float], k: int) -> float:
    """Precision@k: fraction of the top-k results that are relevant.

    The denominator is k, not the number of items actually returned (pytrec_eval /
    TREC convention): a system that returns fewer than k items is charged for the
    empty slots, so P@k is comparable across systems that return different depths.
    Returns 0.0 for k <= 0.
    """
    if k <= 0:
        return 0.0
    relevant = sum(1 for r in relevances[:k] if r > 0)
    return relevant / k


def recall_at_k(relevances: Sequence[float], k: int, n_relevant: int) -> float:
    """Recall@k: fraction of all relevant items for the query found in the top-k.

    `n_relevant` is the corpus total of relevant items (the denominator), so a
    system is never credited for relevant items it failed to surface. Returns 0.0
    when `n_relevant <= 0` (recall is undefined with no relevant items; reported as
    0.0 per BEIR/pytrec_eval). `n_relevant` must be the true total, i.e. at least
    the number of relevant items inside `relevances`.
    """
    if n_relevant <= 0:
        return 0.0
    retrieved_relevant = sum(1 for r in relevances[:k] if r > 0)
    return retrieved_relevant / n_relevant


def r_precision(relevances: Sequence[float], n_relevant: int) -> float:
    """R-precision: precision at rank R, where R = n_relevant (TREC).

    A cutoff-free summary that adapts the cutoff to each query's number of relevant
    items, which makes it robust to queries with very different pool sizes. At rank
    R precision and recall coincide, so this equals both precision_at_k(rel, R) and
    recall_at_k(rel, R, R). Returns 0.0 when `n_relevant <= 0`.
    """
    if n_relevant <= 0:
        return 0.0
    relevant = sum(1 for r in relevances[:n_relevant] if r > 0)
    return relevant / n_relevant
