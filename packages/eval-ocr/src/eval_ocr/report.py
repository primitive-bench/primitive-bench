"""OCR leaderboard: per-slice pass@test with separability badges.

Mirrors the extraction/websearch reporting stance: a slice names a winner only
when its Wilson interval clears the runner-up's AND the pair is McNemar-separable
(`bench_stats.tied_rank_band` + `bench_stats.separable`) — otherwise a TIE band.
This is the "one winner is a lie" gate: per (slice, adapter) results with CIs,
never a single global ranking.

`run()` returns the frozen `SliceResult` records (written to <run>/slices.jsonl)
and a human-readable board, and prints the board. Uncharged non-attempts
(`correct is None` — missing key / refusal / truncation) are excluded from the
denominator, never counted as failures.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from bench_schemas import ItemResult, Primitive, SliceResult
from bench_stats import cusum, separable, tied_rank_band, wilson

_ALL = "all"


def _collect(items: Iterable[ItemResult]) -> tuple[
    str, Primitive, dict[str, dict[str, dict[str, int]]]
]:
    """Index charged outcomes as slice -> adapter -> {item_id: 0/1}."""
    run_id, primitive = "", Primitive.OCR
    by_slice: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for r in items:
        run_id, primitive = r.run_id, r.primitive
        correct = r.output.correct
        if correct is None:  # uncharged non-attempt
            continue
        hit = int(bool(correct))
        for sl in [_ALL, *r.slices]:
            by_slice[sl][r.adapter][r.item_id] = hit
    return run_id, primitive, by_slice


def _mcnemar_counts(a: dict[str, int], b: dict[str, int]) -> tuple[int, int]:
    """Discordant counts on shared items: (a-wrong/b-right, a-right/b-wrong)."""
    n01 = n10 = 0
    for item_id, av in a.items():
        bv = b.get(item_id)
        if bv is None:
            continue
        if av == 0 and bv == 1:
            n01 += 1
        elif av == 1 and bv == 0:
            n10 += 1
    return n01, n10


def run(items: Iterable[ItemResult], *, anchor_target: float | None = None) -> tuple[
    list[SliceResult], list[dict[str, Any]]
]:
    """Build SliceResults + a printable board from scored OCR ItemResults."""
    items = list(items)
    run_id, primitive, by_slice = _collect(items)

    slice_results: list[SliceResult] = []
    board: list[dict[str, Any]] = []

    for sl in sorted(by_slice):
        per_adapter = by_slice[sl]
        # point estimates + Wilson CIs
        rates: list[tuple[str, float, float, float]] = []
        cis = {}
        for adapter, outcomes in per_adapter.items():
            n = len(outcomes)
            succ = sum(outcomes.values())
            ci = wilson(succ, n)
            cis[adapter] = (ci, n, succ)
            rates.append((adapter, ci.statistic or 0.0, ci.ci_low or 0.0, ci.ci_high or 1.0))

        band = tied_rank_band(rates)
        ordered = sorted(rates, key=lambda x: -x[1])

        # The leader is separable only if it is the sole band member AND the
        # leader-vs-runner-up pair is McNemar-separable on shared items.
        leader_separable = False
        if band.winner and len(ordered) >= 2:
            leader, runnerup = ordered[0][0], ordered[1][0]
            n01, n10 = _mcnemar_counts(per_adapter[leader], per_adapter[runnerup])
            leader_separable = separable(n01, n10).separable
        elif band.winner and len(ordered) == 1:
            leader_separable = False  # nothing to separate from

        for rank, (adapter, point, _lo, _hi) in enumerate(ordered, start=1):
            ci, n, _succ = cis[adapter]
            is_winner = adapter == (band.winner or "")
            slice_results.append(SliceResult(
                run_id=run_id,
                primitive=primitive,
                slice=sl,
                adapter=adapter,
                n=n,
                point_estimate=round(point, 4),
                metric_name="pass_at_test",
                ci=ci,
                separable=(is_winner and leader_separable),
                rank=rank,
            ))
        board.append({
            "slice": sl,
            "winner": band.winner if (band.winner and leader_separable) else None,
            "band": band.band,
            "ranking": [(a, round(p, 4), n) for (a, p, _l, _h), (_, _, n)
                        in ((r, cis[r[0]]) for r in ordered)],
        })

    _print_board(board)
    if anchor_target is not None:
        _print_anchor_cusum(by_slice, anchor_target)
    return slice_results, board


def _print_board(board: list[dict[str, Any]]) -> None:
    print("[ocr-report] pass@test by slice (winner = Wilson-clear AND McNemar-separable):")
    for row in board:
        verdict = f"WINNER={row['winner']}" if row["winner"] else f"TIE={row['band']}"
        rank = "  ".join(f"{a}={p:.3f}(n={n})" for a, p, n in row["ranking"])
        print(f"  {row['slice']:>22}  {verdict}\n{'':>24}{rank}")


def _print_anchor_cusum(by_slice: dict[str, dict[str, dict[str, int]]], target: float) -> None:
    """Drift alarm on the tesseract control over the `anchor` slice (if present)."""
    anchor = by_slice.get("anchor", {}).get("tesseract")
    if not anchor:
        return
    series = [float(v) for _, v in sorted(anchor.items())]
    drift = cusum(series, target)
    print(f"[ocr-report] tesseract anchor CUSUM (target={target}): max={max(drift):.3f}")
