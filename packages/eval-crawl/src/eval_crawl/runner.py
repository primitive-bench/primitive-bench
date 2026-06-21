"""Crawl probe runner: (adapter, seed) -> one crawl, scored per target -> ItemResult.

Unlike the reranker runner, a crawl invocation is grouped: a crawler is invoked
ONCE per seed site (an expensive multi-page crawl), and that single result is then
scored against every golden TARGET page under that seed. So for each registered
adapter we crawl each seed once, cache the result, and stream one `ItemResult` per
(adapter, target) to `items.jsonl`. A `VendorUnavailable` (missing key/dep/beta)
skips the whole lane uncharged; a per-seed crawl exception is recorded as an
uncharged failure for every target under that seed. After the run we write the
manifest, derive per-slice `SliceResult`s (report.run), and emit the curated
`snapshots/crawl.counts.toml` the leaderboard pipeline consumes.
"""
from __future__ import annotations

import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any, Iterable

from bench_adapters import get, registry
from bench_adapters.crawl import VendorUnavailable
from bench_core.runlayout import RunDir
from bench_schemas import AdapterSpec, ItemResult, RunManifest
from bench_schemas.models import GroundTruthTier, Primitive

from eval_crawl import report
from eval_crawl._paths import CANARY, SNAPSHOTS_DIR
from eval_crawl.task import Task

MAX_RAW_OUTPUT_CHARS = 2000

# The keyless local crawl strategies are the DEFAULT (reproducible offline run that
# regenerates the public snapshot). The hosted vendors (firecrawl-crawl,
# tavily-crawl, spider-crawl, apify-crawl) are added explicitly for the live run;
# each is skipped uncharged when its key is unset. `bfs-deep` doubles as the
# keyless live baseline.
DEFAULT_VENDORS = ("bfs-shallow", "bfs-deep", "sitemap-crawl", "render-crawl")
HOSTED_VENDORS = ("firecrawl-crawl", "tavily-crawl", "spider-crawl", "apify-crawl")


def run_cost(records: Iterable[ItemResult]) -> float:
    return sum(r.cost_usd or 0.0 for r in records)


def _spec(name: str) -> AdapterSpec:
    cls = get(name)
    return AdapterSpec(
        name=name,
        primitive=Primitive.CRAWL,
        vendor=getattr(cls, "vendor", name),
        version=getattr(cls, "model_version", "unknown"),
        is_sentinel=getattr(cls, "is_sentinel", False),
        params={},
    )


def _env_versions(specs: list[AdapterSpec]) -> dict[str, str]:
    """Allow-listed package versions only — never raw os.environ (no key leakage)."""
    env: dict[str, str] = {"python": platform.python_version()}
    for pkg in ("eval-crawl", "bench-adapters", "bench-stats", "bench-core", "httpx", "selectolax"):
        try:
            env[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            pass
    for s in specs:
        env[f"model:{s.name}"] = s.version
    return env


def _seed_groups(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group target items by seed (preserving first-seen order)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        groups.setdefault(str(it.get("seed_id") or it.get("seed_url")), []).append(it)
    return groups


def _invocation(seed_targets: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the crawl invocation (seed_url + budget + optional inline graph) for a
    seed group. The inline `site` graph may ride on ANY row of the group (the
    controlled split carries it once per seed, not on every target)."""
    head = seed_targets[0]
    site = next((t.get("site") for t in seed_targets if t.get("site")), None)
    return {
        "seed_url": head.get("seed_url", ""),
        "site": site,
        "max_pages": int(head.get("max_pages", 60)),
        "max_depth": int(head.get("max_depth", 4)),
    }


def _item_result(run_id: str, adapter: str, item: dict[str, Any],
                 out: Any, raw: dict[str, Any]) -> ItemResult:
    tier = item.get("ground_truth_tier")
    raw_out = raw.get("raw_output")
    return ItemResult(
        run_id=run_id,
        adapter=adapter,
        item_id=str(item["id"]),
        primitive=Primitive.CRAWL,
        slices=item.get("slices", []),
        ground_truth_tier=GroundTruthTier(tier) if tier else None,
        output=out,
        raw_output=str(raw_out)[:MAX_RAW_OUTPUT_CHARS] if raw_out else None,
        latency_ms=raw.get("latency_ms"),
        cost_usd=raw.get("cost_usd") or 0.0,
        error=raw.get("error"),
    )


def _manifest(run_id: str, seed: int, specs: list[AdapterSpec], notes: str,
              n_items: int, n_seeds: int, spent: float) -> RunManifest:
    progress = f"targets={n_items}; seeds={n_seeds}; adapters={len(specs)}; spend=${spent:.4f}"
    return RunManifest(
        run_id=run_id,
        primitive=Primitive.CRAWL,
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
    """Run the crawl probe over `rows`; write the run dir; return all ItemResults.

    Each adapter crawls each seed once (cached) and is scored against every target
    under that seed. `limit` caps the number of TARGET items (after grouping seeds).
    """
    task = Task(rows)
    items = list(task.items())
    if limit:
        items = items[:limit]
    scorer = task.scorer()
    groups = _seed_groups(items)

    names = [n for n in vendors if n in registry]
    skipped = [n for n in vendors if n not in registry]
    rundir = RunDir(out_root, run_id)

    all_records: list[ItemResult] = []
    specs: list[AdapterSpec] = []
    for name in names:
        spec = _spec(name)
        adapter = get(name)(spec)
        lane_used = False
        unavailable = False
        for _seed_id, seed_targets in groups.items():
            if unavailable:
                break
            invocation = _invocation(seed_targets)
            try:
                raw = adapter.invoke(invocation)
            except VendorUnavailable as exc:
                print(f"[crawl] {name} unavailable, skipping lane (uncharged): {exc}")
                unavailable = True
                break
            except Exception as exc:  # crawl error -> every target under this seed uncharged
                raw = {"error": repr(exc)[:200]}
            for it in seed_targets:
                out = scorer.score(it, raw)
                rec = _item_result(run_id, name, it, out, raw)
                rundir.append_item(rec)
                all_records.append(rec)
                lane_used = True
        if lane_used:
            specs.append(spec)

    spent = run_cost(all_records)
    rundir.write_manifest(_manifest(run_id, seed, specs, notes, len(items), len(groups), spent))
    slice_results, _board = report.run(all_records)
    rundir.write_slices(slice_results)
    if write_snapshot and all_records:
        snap = report.write_counts_toml(
            all_records, SNAPSHOTS_DIR / "crawl.counts.toml",
            run_id=run_id, citation="packages/eval-crawl/README.md",
        )
        print(f"[crawl] wrote snapshot -> {snap}")

    print(f"[crawl] {len(all_records)} (adapter,target) cells across {len(specs)} adapters, "
          f"{len(items)} targets / {len(groups)} seeds; spend=${spent:.4f}"
          + (f"; not registered: {skipped}" if skipped else ""))
    return all_records
