"""retrieval Task + Scorer (BEIR per-query pools, nDCG/recall/MAP/MRR + success@10).

`Task` yields one item per golden row (a query + a fixed candidate pool with graded
relevance); `Scorer` retrieves nothing itself — it scores the adapter's ranking of
that pool via `scoring.score_retrieval`. The leaderboard slices each item by domain
and relevant-set size, so one board becomes many — the "one winner is a lie" surface
for first-stage retrieval.
"""
from __future__ import annotations

from typing import Any, Iterable

from bench_core import Scorer as _Scorer, Task as _Task
from bench_schemas import ScorerOutput
from bench_schemas.models import Primitive

from eval_retrieval.scoring import score_retrieval
from eval_retrieval.slicing import slice_keys


class Scorer(_Scorer):
    """success@10 (paired binary) + nDCG@10/recall@10/MAP/MRR (continuous, in `metrics`).

    `raw` is the bench_adapters retrieval result dict (`retrieved_ids` = the candidate
    ids best-first). Non-attempts (adapter error / empty ranking / no relevant in
    pool) are uncharged (`correct=None`); see scoring.score_retrieval.
    """

    def score(self, item: dict[str, Any], raw: dict[str, Any]) -> ScorerOutput:
        return score_retrieval(item, raw)


class Task(_Task):
    primitive = Primitive.RETRIEVAL
    task_version = "retrieval@1"
    dataset_version = "retrieval-2026.06.scifact+nfcorpus+fiqa.n1271.seed0"

    def __init__(self, rows: Iterable[dict[str, Any]] | None = None) -> None:
        """`rows` are golden rows (query + candidate pool + relevance + domain/slices).

        When omitted the Task carries no rows (the public DEV split is loaded by the
        harness via `eval_retrieval.loader.load_rows`). Each yielded item is
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
