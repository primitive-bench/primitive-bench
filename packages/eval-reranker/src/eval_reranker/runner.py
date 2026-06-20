"""Reranker probe runner: (adapter, golden row) -> ItemResult, streamed to a run dir.

Simpler than the OCR runner — a reranker call is self-contained (one query +
candidate list), so there is no page grouping. For each registered adapter we
invoke it on every item, score the reordering, and stream `ItemResult`s to
`items.jsonl`. A `VendorUnavailable` (missing key / dep / model) skips the whole
lane uncharged; a per-item exception is recorded as an uncharged failure. After the
run we write the manifest, derive per-slice `SliceResult`s (report.run), and emit
the curated `snapshots/reranker.counts.toml` the leaderboard pipeline consumes.
"""
from __future__ import annotations

import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any, Iterable

from bench_adapters import get, registry
from bench_adapters.rerank import VendorUnavailable
from bench_core.runlayout import RunDir
from bench_schemas import AdapterSpec, ItemResult, RunManifest
from bench_schemas.models import GroundTruthTier, Primitive

from eval_reranker import report
from eval_reranker._paths import CANARY, SNAPSHOTS_DIR
from eval_reranker.task import Task

MAX_RAW_OUTPUT_CHARS = 2000

# The free local sentinel + the hosted APIs (skipped if no key). `bge-reranker` is
# registered but opt-in (2.3GB weights, slow on CPU) — pass it explicitly to include it.
DEFAULT_VENDORS = (
    "ce-minilm", "voyage-rerank", "jina-rerank", "cohere-rerank",
)


def run_cost(records: Iterable[ItemResult]) -> float:
    return sum(r.cost_usd or 0.0 for r in records)


def _spec(name: str) -> AdapterSpec:
    cls = get(name)
    return AdapterSpec(
        name=name,
        primitive=Primitive.RERANKER,
        vendor=getattr(cls, "vendor", name),
        version=getattr(cls, "model_version", "unknown"),
        is_sentinel=getattr(cls, "is_sentinel", False),
        params={},
    )


def _env_versions(specs: list[AdapterSpec]) -> dict[str, str]:
    """Allow-listed package versions only — never raw os.environ (no key leakage)."""
    env: dict[str, str] = {"python": platform.python_version()}
    for pkg in ("eval-reranker", "bench-adapters", "bench-stats", "sentence-transformers"):
        try:
            env[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            pass
    for s in specs:
        env[f"model:{s.name}"] = s.version
    return env


def _item_result(run_id: str, adapter: str, item: dict[str, Any],
                 out: Any, raw: dict[str, Any]) -> ItemResult:
    tier = item.get("ground_truth_tier")
    raw_out = raw.get("raw_output")
    return ItemResult(
        run_id=run_id,
        adapter=adapter,
        item_id=str(item["id"]),
        primitive=Primitive.RERANKER,
        slices=item.get("slices", []),
        ground_truth_tier=GroundTruthTier(tier) if tier else None,
        output=out,
        raw_output=str(raw_out)[:MAX_RAW_OUTPUT_CHARS] if raw_out else None,
        latency_ms=raw.get("latency_ms"),
        cost_usd=raw.get("cost_usd") or 0.0,
        error=raw.get("error"),
    )


def _manifest(run_id: str, seed: int, specs: list[AdapterSpec], notes: str,
              n_items: int, spent: float) -> RunManifest:
    progress = f"items/adapter={n_items}; adapters={len(specs)}; spend=${spent:.4f}"
    return RunManifest(
        run_id=run_id,
        primitive=Primitive.RERANKER,
        created_at=datetime.now(timezone.utc),
        seed=seed,
        adapters=specs,
        task_version=Task.task_version,
        dataset_version=Task.dataset_version,
        split="public_dev",
        canary_guid=CANARY,
        env=_env_versions(specs),
        notes=(notes + " | " if notes else "") + progress,
    )


def run_sync(
    rows: Iterable[dict[str, Any]],
    run_id: str,
    *,
    vendors: Iterable[str] = DEFAULT_VENDORS,
    seed: int = 0,
    limit: int | None = None,
    out_root: str = "runs",
    notes: str = "",
    write_snapshot: bool = True,
) -> list[ItemResult]:
    """Run the reranker probe over `rows`; write the run dir; return all ItemResults."""
    task = Task(rows)
    items = list(task.items())
    if limit:
        items = items[:limit]
    scorer = task.scorer()

    names = [n for n in vendors if n in registry]
    skipped = [n for n in vendors if n not in registry]
    rundir = RunDir(out_root, run_id)

    all_records: list[ItemResult] = []
    specs: list[AdapterSpec] = []
    for name in names:
        spec = _spec(name)
        adapter = get(name)(spec)
        lane_used = False
        for it in items:
            try:
                raw = adapter.invoke(it)
            except VendorUnavailable as exc:
                print(f"[reranker] {name} unavailable, skipping lane (uncharged): {exc}")
                break
            except Exception as exc:  # terminal per-item error -> recorded, uncharged
                raw = {"error": repr(exc)[:200]}
            out = scorer.score(it, raw)
            rec = _item_result(run_id, name, it, out, raw)
            rundir.append_item(rec)
            all_records.append(rec)
            lane_used = True
        if lane_used:
            specs.append(spec)

    spent = run_cost(all_records)
    rundir.write_manifest(_manifest(run_id, seed, specs, notes, len(items), spent))
    slice_results, _board = report.run(all_records)
    rundir.write_slices(slice_results)
    if write_snapshot and all_records:
        snap = report.write_counts_toml(
            all_records, SNAPSHOTS_DIR / "reranker.counts.toml",
            run_id=run_id, citation="packages/eval-reranker/README.md",
        )
        print(f"[reranker] wrote snapshot -> {snap}")

    print(f"[reranker] {len(all_records)} (adapter,item) cells across {len(specs)} adapters, "
          f"{len(items)} items; spend=${spent:.4f}"
          + (f"; not registered: {skipped}" if skipped else ""))
    return all_records
