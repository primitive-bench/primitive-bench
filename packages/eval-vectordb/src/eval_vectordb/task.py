"""vectordb Task + Scorer (ANN recall@10 across engines, sliced by dataset/metric/dim/budget).

`Task` yields one item per golden query (its exact top-K neighbor ids); `Scorer` reads
the engine's returned ids and scores recall@10 via `scoring.score_query`. The runner
(`eval_vectordb.runner`) builds the corpora from `datasets.py` and drives the
build-once/query-many lifecycle; `Task` carries the frozen task/dataset versions and
normalizes golden rows from the committed example split. The leaderboard separability
surface is **recall@10 as a proportion** (Σhits/ΣK), so per-slice winners or TIE bands,
never one global ranking.
"""
from __future__ import annotations

from typing import Any, Iterable

from bench_core import Scorer as _Scorer, Task as _Task
from bench_schemas import ScorerOutput
from bench_schemas.models import Primitive

from eval_vectordb.scoring import RECALL_K, score_query
from eval_vectordb.slicing import slice_keys


class Scorer(_Scorer):
    """recall@10 (proportion) + companions (QPS / latency / build / index / cost in `metrics`).

    `raw` is the runner's per-query result dict (`returned_ids` = the engine's top-k base
    ids). Non-attempts (build failure / timeout / adapter error) are uncharged
    (`correct=None`); see scoring.score_query.
    """

    def score(self, item: dict[str, Any], raw: dict[str, Any]) -> ScorerOutput:
        return score_query(item, raw)


class Task(_Task):
    primitive = Primitive.VECTORDB
    task_version = "vectordb@1"
    dataset_version = "vectordb-2026.06.sift+glove+gist+cohere-v3+openai3.seed0"

    def __init__(self, rows: Iterable[dict[str, Any]] | None = None) -> None:
        """`rows` are golden rows (a query + its exact true-neighbor ids + dataset facts).

        When omitted the Task carries no rows — the runner builds the real corpora from
        `datasets.py`. Each yielded item is normalized to the shape the Scorer expects.
        """
        self._rows = list(rows or [])

    def items(self) -> Iterable[dict[str, Any]]:
        for r in self._rows:
            dataset = str(r.get("dataset", "unknown"))
            metric = str(r.get("metric", "euclidean"))
            dim = int(r.get("dim") or len(r.get("query_vector") or []) or 0)
            budget = str(r.get("budget", "accurate"))
            neighbors = r.get("neighbors") or r.get("true_neighbors") or []
            slices = r.get("slices") or slice_keys(dataset, metric, dim, budget)
            yield {
                "id": r.get("row_id") or r.get("id"),
                "neighbors": [int(x) for x in neighbors],
                "k": int(r.get("k") or len(neighbors) or RECALL_K),
                "dataset": dataset,
                "metric": metric,
                "dim": dim,
                "query_vector": r.get("query_vector"),
                "slices": slices,
                "ground_truth_tier": r.get("ground_truth_tier", "verified_external"),
            }

    def scorer(self) -> _Scorer:
        return Scorer()
