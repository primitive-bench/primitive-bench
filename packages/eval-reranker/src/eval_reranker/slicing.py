"""Deterministic slice assignment for reranker golden rows.

Slices are the "one winner is a lie" surface: a constraint that can *separate*
adapters. Assigned from row facts only (no model), so the ingest tool and the Task
agree. Two axes:

  * `domain`                — scientific (SciDocs) vs. tech_qa (AskUbuntu). The
    primary separator; the reason we pull two sources.
  * `hard_negative_density` — fraction of the pool that is a (hard) negative. A
    pool that is almost all negatives is a harder discrimination problem.

(`candidate_depth` was dropped: on the SciDocs+AskUbuntu subset every SciDocs query
has ~30 candidates and every AskUbuntu query has 20, so a depth slice is collinear
with `domain` and adds no independent signal. Re-introduce it if a future source
gives within-domain depth variation.)
"""
from __future__ import annotations

HARD_NEG_HIGH = 0.8


def hard_negative_bucket(density: float) -> str:
    return "high" if density >= HARD_NEG_HIGH else "low"


def slice_keys(domain: str, n_candidates: int, n_relevant: int) -> list[str]:
    """Slice keys for a row: domain, hard_negative_density."""
    density = (n_candidates - n_relevant) / n_candidates if n_candidates else 0.0
    return [
        f"domain:{domain}",
        f"hard_negative_density:{hard_negative_bucket(density)}",
    ]
