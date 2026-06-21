"""Chunking scoring: retrieved chunks vs. gold reference spans -> ScorerOutput.

The downstream-task contract (Chroma "Evaluating Chunking Strategies for Retrieval",
2024): a chunking strategy is good iff, after chunk -> embed -> retrieve top-k, the
retrieved chunks **recover the gold reference spans** of the query. Everything is a
character-range overlap (see ``ranges``):

    intersection = | retrieved ∩ references |
    recall    = intersection / |references|                 # headline (Chroma primary)
    precision = intersection / |retrieved|                   # over/under-chunking penalty
    iou       = intersection / | retrieved ∪ references |    # Jaccard
    precision_Ω = |all_chunks ∩ references| /                # chunking-granularity ceiling
                  |chunks_touching_refs ∪ uncovered_refs|

``correct`` — the paired-binary separability surface the leaderboard's McNemar/Wilson
gate consumes — is **coverage@k**: did the top-k retrieved chunks recover at least
``RECALL_TARGET`` of the gold reference characters? (Same stance as the reranker's
hit@1 / OCR's pass@test: one binary for the gate, the continuous recall/precision/IoU
ride along in ``metrics``.) ``RECALL_TARGET`` is a pinned methodology constant: 0.8
means "the answer evidence is substantially present", which is both retrieval-
meaningful and discriminative between chunkers (a strategy that fragments the answer
across chunks falls below it; one that keeps it together clears it).

Non-attempts are uncharged (``correct=None``): a chunker error, an empty chunking, or
a row whose gold references have zero measure (nothing to recover) are excluded from
the denominator rather than scored a miss.
"""
from __future__ import annotations

from typing import Any, Sequence

from bench_schemas import ScorerOutput

from eval_chunking.ranges import measure_intersection, sum_of_ranges, union_ranges

# Pinned methodology constants (see module docstring).
RETRIEVE_K = 5
RECALL_TARGET = 0.8
COVERAGE_METRIC = "coverage_at_5"

Range = tuple[int, int]


def _gold_ranges(item: dict[str, Any]) -> list[Range]:
    refs = item.get("references") or []
    spans = [(int(r["start_index"]), int(r["end_index"])) for r in refs
             if int(r.get("end_index", 0)) > int(r.get("start_index", 0))]
    return union_ranges(spans)


def _chunk_ranges(chunks: Sequence[dict[str, Any]]) -> list[Range]:
    return union_ranges([(int(c["start"]), int(c["end"])) for c in chunks
                         if int(c.get("end", 0)) > int(c.get("start", 0))])


def precision_omega(all_chunks: Sequence[dict[str, Any]], item: dict[str, Any]) -> float:
    """Best precision the chunking *granularity* allows, independent of retrieval.

    Numerator: gold measure covered by ANY chunk. Denominator: the chunks that touch
    a reference, unioned with any uncovered gold. Tighter chunks around references
    (smaller chunk size) raise this ceiling; oversized chunks lower it.
    """
    gold = _gold_ranges(item)
    if not gold:
        return 0.0
    touching = [(int(c["start"]), int(c["end"])) for c in all_chunks
                if measure_intersection([(int(c["start"]), int(c["end"]))], gold) > 0]
    covered = measure_intersection(union_ranges([(c["start"], c["end"]) for c in all_chunks]), gold)
    uncovered = sum_of_ranges(gold) - covered
    denom = sum_of_ranges(union_ranges(touching)) + max(0, uncovered)
    return covered / denom if denom > 0 else 0.0


def score_chunking(
    item: dict[str, Any],
    retrieved: Sequence[dict[str, Any]],
    *,
    error: str | None = None,
    all_chunks: Sequence[dict[str, Any]] | None = None,
) -> ScorerOutput:
    """Score one (chunker, query) cell -> ScorerOutput.

    ``retrieved`` are the top-k chunks (each ``{"text","start","end"}``) the shared
    embedder surfaced for this query under this chunker. ``all_chunks`` (optional) is
    the chunker's full chunking of the corpus, used only for ``precision_omega``.
    """
    if error:
        return ScorerOutput(correct=None, miss_reason="failed", rationale=str(error)[:200])

    gold = _gold_ranges(item)
    gold_measure = sum_of_ranges(gold)
    if gold_measure <= 0:
        return ScorerOutput(correct=None, miss_reason="references_unresolved")
    if not retrieved:
        return ScorerOutput(correct=None, miss_reason="empty_chunking")

    ret = _chunk_ranges(retrieved)
    inter = measure_intersection(ret, gold)
    union_measure = sum_of_ranges(union_ranges(list(ret) + list(gold)))

    recall = inter / gold_measure
    precision = inter / sum_of_ranges(ret) if ret else 0.0
    iou = inter / union_measure if union_measure > 0 else 0.0
    correct = recall >= RECALL_TARGET

    metrics = {
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "iou": round(iou, 6),
        "full_recall": 1.0 if recall >= 0.999 else 0.0,
        "n_retrieved": float(len(retrieved)),
    }
    if all_chunks is not None:
        metrics["precision_omega"] = round(precision_omega(all_chunks, item), 6)

    return ScorerOutput(
        correct=bool(correct),
        score=round(recall, 6),  # headline continuous metric (Chroma primary)
        metrics=metrics,
        miss_reason=None if correct else ("answer_not_retrieved" if recall == 0 else "partial_recall"),
    )
