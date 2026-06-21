"""Range algebra for token-survival-style chunking scores (Chroma methodology).

Every gold reference and every chunk is a half-open ``[start, end)`` span of
**character** offsets into the source corpus. Recall / precision / IoU are computed
from the measure (summed length) of unions and intersections of these spans —
identical in form to Chroma's ``chunking_evaluation`` (which works in token offsets;
the ratio metrics are equivalent under any monotone offset map, and character space
needs no tokenizer, so it stays exact and offline).

The three primitives below — ``sum_of_ranges``, ``union_ranges``,
``intersection`` — are exactly the operations Chroma composes for each metric.
"""
from __future__ import annotations

from typing import Sequence

Range = tuple[int, int]


def sum_of_ranges(ranges: Sequence[Range]) -> int:
    """Total measure (Σ end−start). Assumes the ranges are already disjoint."""
    return sum(max(0, end - start) for start, end in ranges)


def union_ranges(ranges: Sequence[Range]) -> list[Range]:
    """Merge overlapping/adjacent spans into a minimal disjoint cover (sorted)."""
    spans = sorted((s, e) for s, e in ranges if e > s)
    if not spans:
        return []
    merged: list[Range] = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # overlapping or touching -> extend
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _intersect(a: Range, b: Range) -> Range | None:
    start, end = max(a[0], b[0]), min(a[1], b[1])
    return (start, end) if start < end else None


def intersection(a: Sequence[Range], b: Sequence[Range]) -> list[Range]:
    """Disjoint cover of the overlap between two range sets (a ∩ b)."""
    ua, ub = union_ranges(a), union_ranges(b)
    out: list[Range] = []
    i = j = 0
    while i < len(ua) and j < len(ub):
        hit = _intersect(ua[i], ub[j])
        if hit:
            out.append(hit)
        # advance whichever ends first
        if ua[i][1] <= ub[j][1]:
            i += 1
        else:
            j += 1
    return out


def measure_intersection(a: Sequence[Range], b: Sequence[Range]) -> int:
    """|a ∩ b| — total overlapping measure between two range sets."""
    return sum_of_ranges(intersection(a, b))
