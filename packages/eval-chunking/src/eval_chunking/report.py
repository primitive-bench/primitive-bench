"""Chunking leaderboard: per-slice coverage@k with separability badges.

Mirrors the reranker/extraction reporting stance: a slice names a winner only when
its Wilson interval clears the runner-up's AND the pair is McNemar-separable —
otherwise a TIE band ("one winner is a lie"). coverage@k (recall@k ≥ target) is the
paired-binary separability surface; the continuous recall/precision/IoU ride in each
ItemResult's ``metrics`` and are printed as an auxiliary per-adapter summary (mean +
seeded bootstrap CI). Precision is reported alongside recall because it is where
chunkers diverge most: an over-chunking strategy can buy recall at a steep precision
cost, and the cost-aware reader needs to see it.

``run()`` returns the frozen ``SliceResult`` records (written to <run>/slices.jsonl)
plus a board, and prints both. ``write_counts_toml()`` emits the curated
``snapshots/chunking.counts.toml`` the ``bench results emit`` pipeline turns into the
public leaderboard JSON.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from bench_schemas import ItemResult, Primitive, SliceResult
from bench_stats import bootstrap_ci, separable, tied_rank_band, wilson

from eval_chunking.scoring import COVERAGE_METRIC

_ALL = "all"
METRIC_NAME = COVERAGE_METRIC
BOOTSTRAP_SEED = 0


def _collect(items: Iterable[ItemResult]) -> tuple[
    str, dict[str, dict[str, dict[str, int]]], dict[str, dict[str, list[float]]]
]:
    """Index charged coverage outcomes as slice -> adapter -> {item_id: 0/1}, plus
    per-adapter continuous metric lists (recall, precision) for the summary."""
    run_id = ""
    by_slice: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    cont: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in items:
        run_id = r.run_id
        correct = r.output.correct
        if correct is None:  # uncharged non-attempt
            continue
        hit = int(bool(correct))
        for sl in [_ALL, *r.slices]:
            by_slice[sl][r.adapter][r.item_id] = hit
        for metric in ("recall", "precision"):
            v = (r.output.metrics or {}).get(metric)
            if v is not None:
                cont[metric][r.adapter].append(float(v))
    return run_id, by_slice, cont


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


def run(items: Iterable[ItemResult]) -> tuple[list[SliceResult], list[dict[str, Any]]]:
    """Build SliceResults + a printable board from scored chunking ItemResults."""
    items = list(items)
    run_id, by_slice, cont = _collect(items)

    slice_results: list[SliceResult] = []
    board: list[dict[str, Any]] = []

    for sl in sorted(by_slice):
        per_adapter = by_slice[sl]
        rates: list[tuple[str, float, float, float]] = []
        cis = {}
        for adapter, outcomes in per_adapter.items():
            n = len(outcomes)
            succ = sum(outcomes.values())
            ci = wilson(succ, n)
            cis[adapter] = (ci, n)
            rates.append((adapter, ci.statistic or 0.0, ci.ci_low or 0.0, ci.ci_high or 1.0))

        band = tied_rank_band(rates)
        ordered = sorted(rates, key=lambda x: -x[1])

        leader_separable = False
        if band.winner and len(ordered) >= 2:
            leader, runnerup = ordered[0][0], ordered[1][0]
            n01, n10 = _mcnemar_counts(per_adapter[leader], per_adapter[runnerup])
            leader_separable = separable(n01, n10).separable

        for rank, (adapter, point, _lo, _hi) in enumerate(ordered, start=1):
            ci, n = cis[adapter]
            is_winner = adapter == (band.winner or "")
            slice_results.append(SliceResult(
                run_id=run_id,
                primitive=Primitive.CHUNKING,
                slice=sl,
                adapter=adapter,
                n=n,
                point_estimate=round(point, 4),
                metric_name=METRIC_NAME,
                ci=ci,
                separable=(is_winner and leader_separable),
                rank=rank,
            ))
        board.append({
            "slice": sl,
            "winner": band.winner if (band.winner and leader_separable) else None,
            "band": band.band,
            "ranking": [(a, round(p, 4), cis[a][1]) for a, p, _l, _h in ordered],
        })

    _print_board(board)
    _print_continuous_summary(cont)
    return slice_results, board


def _print_board(board: list[dict[str, Any]]) -> None:
    print("[chunking-report] coverage@k by slice (winner = Wilson-clear AND McNemar-separable):")
    for row in board:
        verdict = f"WINNER={row['winner']}" if row["winner"] else f"TIE={row['band']}"
        rank = "  ".join(f"{a}={p:.3f}(n={n})" for a, p, n in row["ranking"])
        print(f"  {row['slice']:>34}  {verdict}\n{'':>36}{rank}")


def _print_continuous_summary(cont: dict[str, dict[str, list[float]]]) -> None:
    for metric in ("recall", "precision"):
        per = cont.get(metric)
        if not per:
            continue
        print(f"[chunking-report] {metric} per adapter (mean + bootstrap 95% CI):")
        for adapter in sorted(per, key=lambda a: -(sum(per[a]) / max(1, len(per[a])))):
            ci = bootstrap_ci(per[adapter], seed=BOOTSTRAP_SEED)
            print(f"  {adapter:>18}  {metric}={ci.statistic:.4f}  "
                  f"[{ci.ci_low:.4f}, {ci.ci_high:.4f}]  (n={ci.n})")


# --------------------------------------------------------------------------- #
# Curated snapshot (k, n) counts -> snapshots/chunking.counts.toml
# --------------------------------------------------------------------------- #
def collect_counts(items: Iterable[ItemResult]) -> dict[str, list[tuple[str, int, int]]]:
    """{slice_key: [(adapter, hits, n), ...]} on coverage@k — the leaderboard input."""
    _run, by_slice, _cont = _collect(items)
    out: dict[str, list[tuple[str, int, int]]] = {}
    for sl, per in by_slice.items():
        out[sl] = sorted(
            (adapter, sum(o.values()), len(o)) for adapter, o in per.items()
        )
    return out


def _toml_str(s: str) -> str:
    """A TOML basic string: quote and escape backslash then double-quote."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_counts_toml(
    items: Iterable[ItemResult],
    path: str | Path,
    *,
    run_id: str = "chunking-public-snapshot",
    as_of: str = "2026-06",
    citation: str | None = None,
) -> Path:
    """Write the curated ``(adapter, hits, n)`` counts the ``bench results emit``
    pipeline reads (tomllib). Slice keys contain ':' so they are quoted."""
    counts = collect_counts(list(items))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pipeline-emitted (adapter, hits, n) counts for the chunking primitive.",
        "# hits = coverage@k successes (recall@k >= target); the paired-binary separability surface.",
        "# Generated by eval_chunking.report.write_counts_toml from a run's items.jsonl.",
        "",
        'primitive = "chunking"',
        f'run_id = "{run_id}"',
        f'as_of = "{as_of}"',
        f'metric_name = "{METRIC_NAME}"',
    ]
    if citation:
        lines.append(f'citation = "{citation}"')
    lines += ["", "[slices]"]
    for slice_key in sorted(counts):
        rows = ", ".join(f'[{_toml_str(a)}, {k}, {n}]' for a, k, n in counts[slice_key])
        lines.append(f'{_toml_str(slice_key)} = [{rows}]')
    text = "\n".join(lines) + "\n"
    p.write_text(text, encoding="utf-8")
    return p
