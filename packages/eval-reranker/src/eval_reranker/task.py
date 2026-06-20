"""reranker Task + Scorer (MTEB reranking, nDCG/MAP/MRR + hit@1 separability).

`Task` yields one item per golden row (a query + a fixed candidate list with graded
relevance); `Scorer` reorders nothing itself — it scores the adapter's reordering
via `scoring.score_rerank`. The leaderboard slices each item by domain, candidate
depth, and hard-negative density, so one board becomes many — the "one winner is a
lie" surface for reranking.
"""
from __future__ import annotations

from typing import Any, Iterable

from bench_core import Scorer as _Scorer, Task as _Task
from bench_schemas import ScorerOutput
from bench_schemas.models import Primitive

from eval_reranker.scoring import score_rerank
from eval_reranker.slicing import slice_keys


class Scorer(_Scorer):
    """hit@1 (paired binary) + nDCG@10/MAP/MRR (continuous, in `metrics`).

    `raw` is the bench_adapters rerank result dict (`reordered_ids` = the candidate
    ids best-first). Non-attempts (adapter error / empty ranking / no relevant in
    pool) are uncharged (`correct=None`); see scoring.score_rerank.
    """

    def score(self, item: dict[str, Any], raw: dict[str, Any]) -> ScorerOutput:
        return score_rerank(item, raw)


class Task(_Task):
    primitive = Primitive.RERANKER
    task_version = "reranker@1"
    dataset_version = "reranker-2026.06.scidocs+askubuntu.n100.seed0"

    def __init__(self, rows: Iterable[dict[str, Any]] | None = None) -> None:
        """`rows` are golden rows (query + candidates + relevance + domain/slices).

        When omitted the Task carries no rows (the public DEV split is loaded by the
        harness via `eval_reranker.loader.load_rows`). Each yielded item is
        normalized to the shape the Scorer expects.
        """
        self._rows = list(rows or [])

    def items(self) -> Iterable[dict[str, Any]]:
        for r in self._rows:
            candidates = r.get("candidates") or []
            relevance = r.get("relevance") or {}
            n_relevant = int(r.get("n_relevant") or sum(1 for v in relevance.values() if float(v) > 0))
            slices = r.get("slices") or slice_keys(
                str(r.get("domain", "unknown")), len(candidates), n_relevant
            )
            yield {
                "id": r.get("row_id") or r.get("id"),
                "query": r.get("query", ""),
                "candidates": candidates,
                "relevance": relevance,
                "n_relevant": n_relevant,
                "slices": slices,
                "ground_truth_tier": r.get("ground_truth_tier", "verified_external"),
            }

    def scorer(self) -> _Scorer:
        return Scorer()
