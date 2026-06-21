"""Deterministic slice assignment for chunking golden rows.

Slices are the "one winner is a lie" surface: a constraint that can *separate*
chunkers. Assigned from row facts only (no model), so the ingest tool and the Task
agree. Two axes:

  * ``domain``               — the source corpus (chatlogs / finance / pubmed /
    state_of_the_union / wikitexts). THE primary separator: document structure
    differs sharply across domains (turn-structured chat vs. tabular finance vs.
    long encyclopedic prose), and the chunker that respects one structure wrecks
    another. This is why the benchmark spans five corpora.
  * ``reference_dispersion`` — does answering the query need ONE localized span
    (``single``) or material scattered across SEVERAL spans (``multi``)? A
    dispersed answer is the sharp test of chunking: the strategy must keep related
    content together *and* fit every relevant span inside the top-k budget, so it
    separates structure-aware chunkers from naive fixed windows. Analogous to the
    reranker's hard-negative-density axis.
"""
from __future__ import annotations

from typing import Any, Sequence

from eval_chunking.ranges import union_ranges


def reference_dispersion(references: Sequence[dict[str, Any]]) -> str:
    """'multi' if the gold answer lives in >=2 disjoint spans, else 'single'."""
    spans = [(int(r["start_index"]), int(r["end_index"])) for r in references
             if int(r.get("end_index", 0)) > int(r.get("start_index", 0))]
    return "multi" if len(union_ranges(spans)) >= 2 else "single"


def slice_keys(domain: str, references: Sequence[dict[str, Any]]) -> list[str]:
    """Slice keys for a row: domain, reference_dispersion."""
    return [
        f"domain:{domain}",
        f"reference_dispersion:{reference_dispersion(references)}",
    ]
