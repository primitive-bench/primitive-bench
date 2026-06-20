"""Derive per-slice leaderboard reports from raw (k, n) counts.

`bench_schemas.SliceResult` is the frozen, per-(adapter, slice) record. The
leaderboard, though, also needs *slice-level* facts the frozen schema deliberately
does not carry — who won (or whether it is a TIE), whether the slice is too thin to
call, whether it is saturated. Those live here, in a wrapper (`SliceReport`,
`PrimitiveReport`) that *contains* the frozen `SliceResult` objects rather than
extending them. This keeps D-03 intact: we import the contract, never edit it.

Everything is derived deterministically from raw counts using the existing
statistics — `wilson` for the per-adapter CI and `tied_rank_band` for the
Wilson-overlap winner/TIE decision — so the same code path serves the curated
public snapshots today and pipeline-emitted runs later.
"""

from __future__ import annotations

from typing import Literal, Optional

from bench_schemas import SliceResult, StatTest
from bench_schemas.models import Primitive
from pydantic import BaseModel, Field

from bench_stats.proportions import wilson
from bench_stats.reporting import tied_rank_band

# A single raw observation for one adapter on one slice: (adapter, successes, n).
AdapterCount = tuple[str, int, int]

SliceStatus = Literal["published", "thin_data", "saturated"]


class SliceReport(BaseModel):
    """Slice-level view: the frozen per-adapter rows plus the derived call.

    `winner` is set only when the leader's Wilson interval clears the runner-up's
    (D-10 separability gate). Otherwise `band` is the TIE group and `winner` is None.
    `thin`/`saturated` are honesty flags the recommender uses to refuse a call.
    """

    slice: str
    metric_name: str
    n: int = Field(..., description="Slice size (max n across adapters).")
    status: SliceStatus = "published"
    winner: Optional[str] = None
    band: list[str] = Field(default_factory=list, description="TIE group, leader first.")
    thin: bool = False
    saturated: bool = False
    citation: Optional[str] = None
    results: list[SliceResult] = Field(default_factory=list, description="Per-adapter, ranked.")


class PrimitiveReport(BaseModel):
    """All published slices for one primitive — the unit a seed file holds."""

    primitive: Primitive
    run_id: str
    status: Literal["published", "no_published_results"] = "published"
    slices: list[SliceReport] = Field(default_factory=list)


def build_slice_report(
    primitive: Primitive,
    run_id: str,
    slice_key: str,
    counts: list[AdapterCount],
    *,
    metric_name: str,
    citation: Optional[str] = None,
    thin_threshold: int = 10,
    saturated_at: float = 0.9,
) -> SliceReport:
    """Derive one `SliceReport` (winner/TIE + ranked `SliceResult`s) from raw counts."""
    # Per-adapter Wilson interval; keep the StatTest to attach to the SliceResult.
    rows: list[tuple[str, int, StatTest]] = [(adapter, n, wilson(k, n)) for adapter, k, n in counts]

    # Wilson-overlap winner / TIE band (reuses the reporting helper).
    band = tied_rank_band(
        [(a, ci.statistic or 0.0, ci.ci_low or 0.0, ci.ci_high or 1.0) for a, _n, ci in rows]
    )

    # Rank by point estimate, descending; rank 1 = leader.
    ordered = sorted(rows, key=lambda r: -(r[2].statistic or 0.0))
    slice_n = max((n for _a, n, _ci in rows), default=0)

    results: list[SliceResult] = []
    for rank, (adapter, n, ci) in enumerate(ordered, start=1):
        # Only the leader carries a separability verdict vs. the runner-up.
        separable = (band.winner is not None) if rank == 1 else None
        results.append(SliceResult(
            run_id=run_id,
            primitive=primitive,
            slice=slice_key,
            adapter=adapter,
            n=n,
            point_estimate=ci.statistic or 0.0,
            metric_name=metric_name,
            ci=ci,
            separable=separable,
            rank=rank,
        ))

    thin = slice_n < thin_threshold
    all_tied = len(band.band) == len(rows) and len(rows) > 1
    min_point = min((ci.statistic or 0.0 for _a, _n, ci in rows), default=0.0)
    saturated = all_tied and min_point >= saturated_at and not thin

    status: SliceStatus = "thin_data" if thin else "saturated" if saturated else "published"

    return SliceReport(
        slice=slice_key,
        metric_name=metric_name,
        n=slice_n,
        status=status,
        winner=band.winner,
        band=band.band,
        thin=thin,
        saturated=saturated,
        citation=citation,
        results=results,
    )


def build_primitive_report(
    primitive: Primitive,
    run_id: str,
    slices_raw: dict[str, list[AdapterCount]],
    *,
    metric_name: str,
    citation: Optional[str] = None,
    thin_threshold: int = 10,
    saturated_at: float = 0.9,
) -> PrimitiveReport:
    """Build a full `PrimitiveReport` from a {slice_key: [(adapter, k, n), ...]} map."""
    slices = [
        build_slice_report(
            primitive, run_id, slice_key, counts,
            metric_name=metric_name, citation=citation,
            thin_threshold=thin_threshold, saturated_at=saturated_at,
        )
        for slice_key, counts in slices_raw.items()
    ]
    return PrimitiveReport(primitive=primitive, run_id=run_id, status="published", slices=slices)
