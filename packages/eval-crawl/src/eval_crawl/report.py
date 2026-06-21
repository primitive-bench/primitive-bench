"""Crawl leaderboard: per-slice page_coverage with separability badges.

Mirrors the reranker/extraction reporting stance: a slice names a winner only when
its Wilson interval clears the runner-up's AND the pair is McNemar-separable —
otherwise a TIE band ("one winner is a lie"). page_coverage (reached AND fresh
content delivered) is the paired-binary separability surface.

Crawl quality has two axes (Olston & Najork 2010), so the board also surfaces them
apart from the headline: a per-adapter **coverage vs freshness** summary
(`url_discovered` mean = pure URL-recall; `content_fresh` mean) and the **miss
decomposition** (not_reached = coverage gap; stale = freshness gap; plus
blocked/token_absent/empty), so a low score is legible as a discovery failure vs a
staleness failure rather than one opaque number.

`run()` returns the frozen `SliceResult` records (written to <run>/slices.jsonl)
plus a board, and prints both. `write_counts_toml()` emits the curated
`snapshots/crawl.counts.toml` the `bench results emit` pipeline turns into the
public leaderboard JSON.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from bench_schemas import ItemResult, Primitive, SliceResult
from bench_stats import separable, tied_rank_band, wilson

_ALL = "all"
METRIC_NAME = "page_coverage"


def _collect(items: Iterable[ItemResult]) -> tuple[
    str,
    dict[str, dict[str, dict[str, int]]],
    dict[str, list[float]],
    dict[str, dict[str, int]],
]:
    """Index charged coverage outcomes as slice -> adapter -> {item_id: 0/1}, plus
    per-adapter url_discovered (pure coverage) lists and miss-reason counts."""
    run_id = ""
    by_slice: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    discovered: dict[str, list[float]] = defaultdict(list)
    miss_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in items:
        run_id = r.run_id
        correct = r.output.correct
        if correct is None:  # uncharged non-attempt (crawl error)
            continue
        hit = int(bool(correct))
        for sl in [_ALL, *r.slices]:
            by_slice[sl][r.adapter][r.item_id] = hit
        ud = (r.output.metrics or {}).get("url_discovered")
        if ud is not None:
            discovered[r.adapter].append(float(ud))
        if not correct and r.output.miss_reason:
            miss_counts[r.adapter][r.output.miss_reason] += 1
    return run_id, by_slice, discovered, miss_counts


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
    """Build SliceResults + a printable board from scored crawl ItemResults."""
    items = list(items)
    run_id, by_slice, discovered, miss_counts = _collect(items)

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
                primitive=Primitive.CRAWL,
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
    _print_axes_summary(discovered, miss_counts)
    return slice_results, board


def _print_board(board: list[dict[str, Any]]) -> None:
    print("[crawl-report] page_coverage by slice (winner = Wilson-clear AND McNemar-separable):")
    for row in board:
        verdict = f"WINNER={row['winner']}" if row["winner"] else f"TIE={row['band']}"
        rank = "  ".join(f"{a}={p:.3f}(n={n})" for a, p, n in row["ranking"])
        print(f"  {row['slice']:>26}  {verdict}\n{'':>28}{rank}")


def _print_axes_summary(discovered: dict[str, list[float]],
                        miss_counts: dict[str, dict[str, int]]) -> None:
    """Coverage (url_discovered mean) + the miss decomposition per adapter."""
    if not discovered:
        return
    print("[crawl-report] coverage (URL-recall) + miss decomposition per adapter:")
    for adapter in sorted(discovered, key=lambda a: -(sum(discovered[a]) / max(1, len(discovered[a])))):
        ud = discovered[adapter]
        cov = sum(ud) / len(ud) if ud else 0.0
        misses = dict(miss_counts.get(adapter, {}))
        nr = misses.get("not_reached", 0)
        st = misses.get("stale", 0)
        extra = "  ".join(f"{k}={v}" for k, v in sorted(misses.items()))
        print(f"  {adapter:>14}  url_recall={cov:.3f}  not_reached={nr} stale={st}"
              + (f"  [{extra}]" if extra else ""))


# --------------------------------------------------------------------------- #
# Curated snapshot (k, n) counts -> snapshots/crawl.counts.toml
# --------------------------------------------------------------------------- #
def collect_counts(items: Iterable[ItemResult]) -> dict[str, list[tuple[str, int, int]]]:
    """{slice_key: [(adapter, hits, n), ...]} on page_coverage — the leaderboard input."""
    _run, by_slice, _disc, _miss = _collect(items)
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
    run_id: str = "crawl-public-snapshot",
    as_of: str = "2026-06",
    citation: str | None = None,
) -> Path:
    """Write the curated `(adapter, hits, n)` counts the `bench results emit`
    pipeline reads (tomllib). Slice keys contain ':' so they are quoted."""
    counts = collect_counts(list(items))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pipeline-emitted (adapter, hits, n) counts for the crawl primitive.",
        "# hits = page_coverage successes (target reached AND current content delivered);",
        "# metric is the paired-binary separability surface.",
        "# Generated by eval_crawl.report.write_counts_toml from a run's items.jsonl.",
        "",
        'primitive = "crawl"',
        f'run_id = "{run_id}"',
        f'as_of = "{as_of}"',
        'metric_name = "page_coverage"',
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
