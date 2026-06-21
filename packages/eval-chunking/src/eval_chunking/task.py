"""chunking Task + Scorer (downstream retrieval over Chroma chunking corpora).

`Task` yields one item per golden row (a query + its gold reference spans + the
corpus it lives in). The chunking itself is corpus-level, so the *runner* chunks each
corpus with a chunker, embeds with the shared embedder, and retrieves top-k per
query; `Scorer` then scores the retrieved chunks against the gold spans via
`scoring.score_chunking`. The leaderboard slices each item by domain (corpus) and
reference dispersion, so one board becomes many — the "one winner is a lie" surface
for chunking.
"""
from __future__ import annotations

from typing import Any, Iterable

from bench_core import Scorer as _Scorer, Task as _Task
from bench_schemas import ScorerOutput
from bench_schemas.models import Primitive

from eval_chunking.scoring import score_chunking
from eval_chunking.slicing import slice_keys


class Scorer(_Scorer):
    """coverage@k (paired binary) + recall/precision/IoU/precision_Ω (continuous).

    `raw` is the runner's retrieval result for one (chunker, query) cell:
    ``{"retrieved": [chunk...], "all_chunks": [chunk...], "error": str|None}``.
    Non-attempts (chunker error / empty chunking / unresolved refs) are uncharged
    (`correct=None`); see scoring.score_chunking.
    """

    def score(self, item: dict[str, Any], raw: dict[str, Any]) -> ScorerOutput:
        return score_chunking(
            item,
            raw.get("retrieved") or [],
            error=raw.get("error"),
            all_chunks=raw.get("all_chunks"),
        )


class Task(_Task):
    primitive = Primitive.CHUNKING
    task_version = "chunking@1"
    dataset_version = "chunking-2026.06.chroma5.k5.seed0"

    def __init__(self, rows: Iterable[dict[str, Any]] | None = None) -> None:
        """`rows` are golden rows (query + references + corpus_id + slices).

        When omitted the Task carries no rows (the public DEV split is loaded by the
        harness via `eval_chunking.loader.load_golden`). Each yielded item is
        normalized to the shape the runner + Scorer expect.
        """
        self._rows = list(rows or [])

    def items(self) -> Iterable[dict[str, Any]]:
        for r in self._rows:
            references = r.get("references") or []
            domain = str(r.get("corpus_id") or r.get("domain") or "unknown")
            slices = r.get("slices") or slice_keys(domain, references)
            yield {
                "id": r.get("row_id") or r.get("id"),
                "query": r.get("question") or r.get("query", ""),
                "corpus_id": domain,
                "references": references,
                "slices": slices,
                "ground_truth_tier": r.get("ground_truth_tier", "verified_external"),
            }

    def scorer(self) -> _Scorer:
        return Scorer()
