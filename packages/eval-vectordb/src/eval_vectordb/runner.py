"""Vectordb runner: (engine, config, dataset, query) -> ItemResult, streamed to a run dir.

A vector engine is stateful: build() an index once over a dataset's base vectors, then
query() it for every test vector. So the loop is nested — for each dataset, for each
engine, for each operating-point config (budget: fast vs accurate), we build once
(timing it + capturing index memory), query all test vectors (timing each for QPS /
p50 / p99), score recall@10 against the exact neighbors, and stream `ItemResult`s.

A `VendorUnavailable` (missing dep / key / unreachable service) skips that engine for
the whole run, uncharged (its dep won't appear mid-run). A build failure / OOM records
a single uncharged marker for that (engine, config, dataset). A per-query exception is
recorded as an uncharged failure on that query. After the run we write the manifest,
derive per-slice `SliceResult`s (report.run), and emit the curated
`snapshots/vectordb.counts.toml` the leaderboard pipeline consumes.
"""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any, Iterable

from bench_adapters import get, registry
from bench_adapters.vectordb import VendorUnavailable
from bench_adapters.vectordb import pricing
from bench_core.runlayout import RunDir
from bench_schemas import AdapterSpec, ItemResult, RunManifest
from bench_schemas.models import GroundTruthTier, Primitive

from eval_vectordb import datasets as datasets_mod
from eval_vectordb import report
from eval_vectordb._paths import CANARY, SNAPSHOTS_DIR
from eval_vectordb.scoring import RECALL_K
from eval_vectordb.slicing import slice_keys
from eval_vectordb.task import Task

MAX_RAW_OUTPUT_CHARS = 2000

# OSS in-process engines run keyless out of the box; Docker/hosted engines skip
# cleanly (VendorUnavailable) unless their service/key is present. Pass an explicit
# --engines list to add pgvector / milvus / weaviate / elasticsearch / pinecone / ...
DEFAULT_ENGINES = (
    "bruteforce-numpy", "faiss-flat", "faiss-hnsw", "faiss-ivf",
    "hnswlib", "annoy", "qdrant-local", "lancedb",
)

DEFAULT_DATASETS = ("synthetic",)


@dataclass
class Config:
    """One operating point for an engine: a budget label + its index/query params."""

    budget: str
    params: dict[str, Any] = field(default_factory=dict)


# fast = low ef/nprobe/trees (high QPS, lower recall); accurate = the opposite.
_GRAPH_CONFIGS = [
    Config("fast", {"ef_search": 16, "M": 16, "ef_construction": 200}),
    Config("accurate", {"ef_search": 200, "M": 16, "ef_construction": 200}),
]
_IVF_CONFIGS = [
    Config("fast", {"nprobe": 1}),
    Config("accurate", {"nprobe": 32}),
]
_ANNOY_CONFIGS = [
    Config("fast", {"n_trees": 10, "search_k": -1}),
    Config("accurate", {"n_trees": 100, "search_k": 20000}),
]
_EXACT_CONFIGS = [Config("accurate", {})]  # exact engines have a single operating point

_CONFIGS: dict[str, list[Config]] = {
    "bruteforce-numpy": _EXACT_CONFIGS,
    "faiss-flat": _EXACT_CONFIGS,
    "faiss-hnsw": _GRAPH_CONFIGS,
    "hnswlib": _GRAPH_CONFIGS,
    "qdrant-local": _GRAPH_CONFIGS,
    "weaviate": _GRAPH_CONFIGS,
    "weaviate-cloud": _GRAPH_CONFIGS,
    "pgvector": _GRAPH_CONFIGS,
    "milvus": _GRAPH_CONFIGS,
    "zilliz-cloud": _GRAPH_CONFIGS,
    "elasticsearch": _GRAPH_CONFIGS,
    "pinecone": [Config("accurate", {})],  # managed; serverless has no ef knob exposed
    "faiss-ivf": _IVF_CONFIGS,
    "lancedb": _IVF_CONFIGS,
    "annoy": _ANNOY_CONFIGS,
}


def _configs_for(name: str) -> list[Config]:
    return _CONFIGS.get(name, _GRAPH_CONFIGS)


def run_cost(records: Iterable[ItemResult]) -> float:
    return sum(r.cost_usd or 0.0 for r in records)


def _spec(name: str, cfg: Config) -> AdapterSpec:
    cls = get(name)
    return AdapterSpec(
        name=name,
        primitive=Primitive.VECTORDB,
        vendor=getattr(cls, "vendor", name),
        version=getattr(cls, "engine_version", "unknown"),
        is_sentinel=getattr(cls, "is_sentinel", False),
        params={"budget": cfg.budget, **cfg.params},
        publish_restricted=getattr(cls, "publish_restricted", False),
    )


def _env_versions(specs: list[AdapterSpec]) -> dict[str, str]:
    """Allow-listed package versions only — never raw os.environ (no key leakage)."""
    env: dict[str, str] = {"python": platform.python_version()}
    for pkg in ("eval-vectordb", "bench-adapters", "bench-stats", "numpy",
                "faiss-cpu", "hnswlib", "annoy", "qdrant-client", "lancedb", "h5py"):
        try:
            env[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            pass
    for s in specs:
        env[f"engine:{s.name}:{s.params.get('budget')}"] = s.version
    return env


def _item(ds: datasets_mod.Dataset, ds_name: str, budget: str, qi: int, neighbors) -> dict[str, Any]:
    return {
        "id": f"{ds_name}:{budget}:{qi}",
        "neighbors": [int(x) for x in neighbors],
        "k": ds.k,
        "dataset": ds_name,
        "metric": ds.metric,
        "dim": ds.dim,
        "slices": slice_keys(ds_name, ds.metric, ds.dim, budget),
        "ground_truth_tier": "verified_external",
    }


def _item_result(run_id: str, adapter: str, item: dict[str, Any],
                 out: Any, raw: dict[str, Any]) -> ItemResult:
    tier = item.get("ground_truth_tier")
    raw_out = raw.get("raw_output")
    return ItemResult(
        run_id=run_id,
        adapter=adapter,
        item_id=str(item["id"]),
        primitive=Primitive.VECTORDB,
        slices=item.get("slices", []),
        ground_truth_tier=GroundTruthTier(tier) if tier else None,
        output=out,
        raw_output=str(raw_out)[:MAX_RAW_OUTPUT_CHARS] if raw_out else None,
        latency_ms=raw.get("latency_ms"),
        cost_usd=raw.get("cost_usd") or 0.0,
        error=raw.get("error"),
    )


def _manifest(run_id: str, seed: int, specs: list[AdapterSpec], notes: str,
              n_items: int, spent: float, datasets: list[str]) -> RunManifest:
    progress = f"datasets={datasets}; engine-configs={len(specs)}; items={n_items}; spend=${spent:.4f}"
    return RunManifest(
        run_id=run_id,
        primitive=Primitive.VECTORDB,
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
    datasets: Iterable[str] = DEFAULT_DATASETS,
    run_id: str = "vectordb-dev",
    *,
    engines: Iterable[str] = DEFAULT_ENGINES,
    seed: int = 0,
    limit: int | None = None,
    base_limit: int | None = datasets_mod.DEFAULT_BASE_LIMIT,
    query_limit: int | None = datasets_mod.DEFAULT_QUERY_LIMIT,
    k: int = RECALL_K,
    out_root: str = "runs",
    notes: str = "",
    write_snapshot: bool = True,
) -> list[ItemResult]:
    """Build each engine over each dataset, query, score recall@10; write the run dir."""
    dataset_names = list(datasets)
    engine_names = [n for n in engines if n in registry]
    skipped_registry = [n for n in engines if n not in registry]
    rundir = RunDir(out_root, run_id)
    scorer = Task().scorer()

    all_records: list[ItemResult] = []
    specs: list[AdapterSpec] = []
    unavailable: set[str] = set()

    for ds_name in dataset_names:
        ds = datasets_mod.load(ds_name, base_limit=base_limit, query_limit=query_limit, k=k, seed=seed)
        queries = ds.test if limit is None else ds.test[:limit]
        truth = ds.neighbors if limit is None else ds.neighbors[:limit]

        for name in engine_names:
            if name in unavailable:
                continue
            for cfg in _configs_for(name):
                spec = _spec(name, cfg)
                engine = get(name)(spec)
                # ---- build ------------------------------------------------ #
                try:
                    t0 = time.monotonic()
                    engine.build(ds.train, ds.metric, cfg.params)
                    build_s = time.monotonic() - t0
                except VendorUnavailable as exc:
                    print(f"[vectordb] {name} unavailable, skipping (uncharged): {exc}")
                    unavailable.add(name)
                    break
                except MemoryError:
                    all_records.append(_build_failure(run_id, ds, ds_name, cfg, name, "oom"))
                    continue
                except Exception as exc:  # build_failed -> uncharged marker
                    rec = _build_failure(run_id, ds, ds_name, cfg, name, "build_failed", repr(exc))
                    rundir.append_item(rec)
                    all_records.append(rec)
                    continue
                spec.version = getattr(engine, "engine_version", "unknown")
                index_mb = (engine.index_memory_bytes() or 0) / 1e6

                # ---- query ------------------------------------------------ #
                results: list[tuple[list[int], str | None, float]] = []
                for q in queries:
                    qt = time.monotonic()
                    try:
                        ids = engine.query(q, ds.k)
                        err = None
                    except Exception as exc:
                        ids, err = [], repr(exc)[:200]
                    results.append((ids, err, (time.monotonic() - qt) * 1000.0))

                total_s = sum(lat for _i, _e, lat in results) / 1000.0
                qps = (len(results) / total_s) if total_s > 0 else 0.0
                cost_per_query = pricing.query_cost(spec.vendor, 1.0)
                engine_metrics = {"qps": qps, "build_s": build_s, "index_mb": index_mb}

                lane_used = False
                for qi, (ids, err, lat) in enumerate(results):
                    item = _item(ds, ds_name, cfg.budget, qi, truth[qi])
                    raw = {
                        "returned_ids": ids,
                        "latency_ms": lat,
                        "cost_usd": cost_per_query,
                        "engine_metrics": engine_metrics,
                        "error": err,
                        "miss_reason": "adapter_error" if err else None,
                        "raw_output": ",".join(str(i) for i in ids[:RECALL_K]),
                    }
                    out = scorer.score(item, raw)
                    rec = _item_result(run_id, name, item, out, raw)
                    rundir.append_item(rec)
                    all_records.append(rec)
                    lane_used = True

                engine.free()
                if lane_used:
                    specs.append(spec)
                print(f"[vectordb] {name}/{cfg.budget} on {ds_name}: "
                      f"build={build_s:.2f}s qps={qps:.0f} index={index_mb:.1f}MB")

    spent = run_cost(all_records)
    rundir.write_manifest(_manifest(run_id, seed, specs, notes, len(all_records), spent, dataset_names))
    slice_results, _board = report.run(all_records)
    rundir.write_slices(slice_results)
    if write_snapshot and all_records:
        snap = report.write_counts_toml(
            all_records, SNAPSHOTS_DIR / "vectordb.counts.toml",
            run_id=run_id, citation="packages/eval-vectordb/README.md",
        )
        print(f"[vectordb] wrote snapshot -> {snap}")

    print(f"[vectordb] {len(all_records)} (engine,query) cells across {len(specs)} engine-configs, "
          f"{len(dataset_names)} datasets; spend=${spent:.4f}"
          + (f"; not registered: {skipped_registry}" if skipped_registry else ""))
    return all_records


def _build_failure(run_id: str, ds, ds_name: str, cfg: Config, name: str,
                   reason: str, detail: str = "") -> ItemResult:
    """One uncharged marker for a (engine, config, dataset) build that never ran."""
    from bench_schemas import ScorerOutput

    item = _item(ds, ds_name, cfg.budget, -1, [])
    return ItemResult(
        run_id=run_id,
        adapter=name,
        item_id=f"{ds_name}:{cfg.budget}:build",
        primitive=Primitive.VECTORDB,
        slices=item["slices"],
        ground_truth_tier=GroundTruthTier.VERIFIED_EXTERNAL,
        output=ScorerOutput(correct=None, miss_reason=reason, rationale=detail[:200] or None),
        error=detail[:200] or reason,
    )
