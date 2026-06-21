"""Vectordb leaderboard: per-slice recall@10 with separability badges.

recall@10 is a *proportion* — per slice we pool `k=Σhits, n=ΣK` over queries and run
it through the SAME Wilson + `tied_rank_band` path the public `bench results emit`
pipeline uses (`bench_stats.leaderboard.build_slice_report`), so the run-dir
`slices.jsonl` and the emitted leaderboard agree exactly. A slice names a winner only
when the leader's Wilson interval clears the runner-up's; otherwise a TIE band.

KEY DIVERGENCE FROM THE RERANKER REPORT: recall is fractional per query, so `_collect`
sums `hits`/`k` from each ItemResult's `metrics` — it does NOT binarize `correct`
(which is only the coarse "perfect-recall?" view). The continuous companions (QPS,
p50/p99 latency, build time, index memory, $cost) ride in `metrics` and are printed as
an auxiliary per-engine summary — the recall-vs-QPS-vs-cost surface, never the gate.

`run()` returns the frozen `SliceResult` records (written to <run>/slices.jsonl) plus a
board, and prints both. `write_counts_toml()` emits the curated
`snapshots/vectordb.counts.toml` the `bench results emit` pipeline turns into the
public leaderboard JSON.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from bench_schemas import ItemResult, Primitive, SliceResult
from bench_stats import tied_rank_band, wilson

_ALL = "all"
METRIC_NAME = "recall_at_10"
_CONT_KEYS = ("recall@10", "latency_ms", "qps", "build_s", "index_mb", "cost_usd")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[idx]


def _collect(items: Iterable[ItemResult]) -> tuple[
    str, dict[str, dict[str, list[int]]], dict[str, dict[str, list[float]]]
]:
    """slice -> adapter -> [Σhits, Σk] (charged recall), plus per-adapter continuous lists."""
    run_id = ""
    by_slice: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    cont: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in items:
        run_id = r.run_id
        if r.output.correct is None:  # uncharged non-attempt (build fail / timeout / error)
            continue
        m = r.output.metrics or {}
        hits, k = int(m.get("hits", 0)), int(m.get("k", 0))
        if k <= 0:
            continue
        for sl in [_ALL, *r.slices]:
            agg = by_slice[sl][r.adapter]
            agg[0] += hits
            agg[1] += k
        for key in _CONT_KEYS:
            if m.get(key) is not None:
                cont[r.adapter][key].append(float(m[key]))
    return run_id, by_slice, cont


def run(items: Iterable[ItemResult]) -> tuple[list[SliceResult], list[dict[str, Any]]]:
    """Build SliceResults + a printable board from scored vectordb ItemResults."""
    items = list(items)
    run_id, by_slice, cont = _collect(items)

    slice_results: list[SliceResult] = []
    board: list[dict[str, Any]] = []

    for sl in sorted(by_slice):
        per = by_slice[sl]
        rates: list[tuple[str, float, float, float]] = []
        cis: dict[str, tuple[Any, int]] = {}
        for adapter, (hits, k) in per.items():
            ci = wilson(hits, k)
            cis[adapter] = (ci, k)
            rates.append((adapter, ci.statistic or 0.0, ci.ci_low or 0.0, ci.ci_high or 1.0))

        band = tied_rank_band(rates)
        ordered = sorted(rates, key=lambda x: -x[1])

        for rank, (adapter, point, _lo, _hi) in enumerate(ordered, start=1):
            ci, n = cis[adapter]
            slice_results.append(SliceResult(
                run_id=run_id,
                primitive=Primitive.VECTORDB,
                slice=sl,
                adapter=adapter,
                n=n,
                point_estimate=round(point, 4),
                metric_name=METRIC_NAME,
                ci=ci,
                separable=(band.winner is not None) if rank == 1 else None,
                rank=rank,
            ))
        board.append({
            "slice": sl,
            "winner": band.winner,
            "band": band.band,
            "ranking": [(a, round(p, 4), cis[a][1]) for a, p, _l, _h in ordered],
        })

    _print_board(board)
    _print_continuous(cont)
    return slice_results, board


def _print_board(board: list[dict[str, Any]]) -> None:
    print("[vectordb-report] recall@10 by slice (winner = Wilson interval clears runner-up):")
    for row in board:
        verdict = f"WINNER={row['winner']}" if row["winner"] else f"TIE={row['band']}"
        rank = "  ".join(f"{a}={p:.3f}(n={n})" for a, p, n in row["ranking"])
        print(f"  {row['slice']:>26}  {verdict}\n{'':>28}{rank}")


def _print_continuous(cont: dict[str, dict[str, list[float]]]) -> None:
    if not cont:
        return
    print("[vectordb-report] recall / QPS / latency / build / index / cost per engine "
          "(pooled across slices):")
    rows = []
    for adapter, m in cont.items():
        lat = m.get("latency_ms", [])
        total_s = sum(lat) / 1000.0
        rows.append({
            "adapter": adapter,
            "recall": mean(m.get("recall@10", [0.0])),
            "qps": (len(lat) / total_s) if total_s > 0 else 0.0,
            "p50": _percentile(lat, 50),
            "p99": _percentile(lat, 99),
            "build_s": mean(m.get("build_s", [0.0])),
            "index_mb": mean(m.get("index_mb", [0.0])),
            "cost_usd": sum(m.get("cost_usd", [0.0])),
        })
    for r in sorted(rows, key=lambda x: -x["recall"]):
        print(f"  {r['adapter']:>16}  recall={r['recall']:.4f}  qps={r['qps']:8.1f}  "
              f"p50={r['p50']:7.2f}ms  p99={r['p99']:7.2f}ms  build={r['build_s']:6.2f}s  "
              f"index={r['index_mb']:7.1f}MB  cost=${r['cost_usd']:.4f}")


# --------------------------------------------------------------------------- #
# Curated snapshot (adapter, Σhits, ΣK) counts -> snapshots/vectordb.counts.toml
# --------------------------------------------------------------------------- #
def collect_counts(items: Iterable[ItemResult]) -> dict[str, list[tuple[str, int, int]]]:
    """{slice_key: [(adapter, Σhits, ΣK), ...]} on recall@10 — the leaderboard input."""
    _run, by_slice, _cont = _collect(items)
    out: dict[str, list[tuple[str, int, int]]] = {}
    for sl, per in by_slice.items():
        out[sl] = sorted((adapter, hk[0], hk[1]) for adapter, hk in per.items())
    return out


def _toml_str(s: str) -> str:
    """A TOML basic string: quote and escape backslash then double-quote."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_counts_toml(
    items: Iterable[ItemResult],
    path: str | Path,
    *,
    run_id: str = "vectordb-public-snapshot",
    as_of: str = "2026-06",
    citation: str | None = None,
) -> Path:
    """Write the curated `(adapter, Σhits, ΣK)` recall counts the `bench results emit`
    pipeline reads (tomllib). Slice keys contain ':' so they are quoted."""
    counts = collect_counts(list(items))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pipeline-emitted (adapter, hits, n) recall@10 counts for the vectordb primitive.",
        "# hits = true neighbors retrieved in top-10; n = total true neighbors (queries * 10).",
        "# recall@10 = hits / n is the proportion separability surface (Wilson + tied_rank_band).",
        "# Generated by eval_vectordb.report.write_counts_toml from a run's items.jsonl.",
        "",
        'primitive = "vectordb"',
        f'run_id = "{run_id}"',
        f'as_of = "{as_of}"',
        'metric_name = "recall_at_10"',
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
