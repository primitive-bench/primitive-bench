"""Deterministic slice assignment for retrieval golden rows.

Slices are the "one winner is a lie" surface: a constraint that can *separate*
retrievers. Assigned from row facts only (no model), so the ingest tool and the Task
agree. Two axes:

  * `domain`        — the source corpus (scientific / medical / financial / ...). The
    primary separator; embedding models tuned on web/QA text vs. scientific or
    biomedical text trade places across domains, which is why we pull several sources.
  * `relevant_set`  — `single` (exactly one relevant doc for the query) vs. `multi`
    (several). A single-answer query is an all-or-nothing needle-in-haystack; a
    multi-answer query rewards broad recall. The two reward different retrievers.

(Reranker's `hard_negative_density` axis is dropped here: a BM25 top-N pool is almost
entirely negatives for every query, so that axis is degenerate for first-stage
retrieval. `relevant_set` is the non-degenerate analogue.)
"""
from __future__ import annotations


def relevant_set_bucket(n_relevant: int) -> str:
    return "single" if n_relevant <= 1 else "multi"


def slice_keys(domain: str, n_candidates: int, n_relevant: int) -> list[str]:
    """Slice keys for a row: domain, relevant_set."""
    return [
        f"domain:{domain}",
        f"relevant_set:{relevant_set_bucket(n_relevant)}",
    ]
