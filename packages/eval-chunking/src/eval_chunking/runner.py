"""Chunking probe runner: (chunker, query) -> ItemResult, streamed to a run dir.

Chunking is corpus-level, so the loop is one layer deeper than the reranker's. For
each corpus we fit the **shared embedder once** and reuse it for every chunker — that
is what holds the embedder constant so the comparison isolates the chunker. Then, per
chunker: chunk the corpus, embed the chunks, and for every query in that corpus
retrieve top-k and score the recovery of the gold spans. ``ItemResult``s stream to
``items.jsonl``. A chunker that cannot run (e.g. a semantic chunker with no embedder)
skips its lane uncharged; a per-corpus chunking exception charges that corpus's
queries as uncharged failures. After the run we write the manifest, derive per-slice
``SliceResult``s (report.run), and emit the curated ``snapshots/chunking.counts.toml``
the leaderboard pipeline consumes.
"""
from __future__ import annotations

import platform
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any, Iterable

from bench_adapters import get, registry
from bench_adapters.chunk import VendorUnavailable
from bench_core.runlayout import RunDir
from bench_schemas import AdapterSpec, ItemResult, RunManifest
from bench_schemas.models import GroundTruthTier, Primitive

from eval_chunking import report
from eval_chunking._paths import CANARY, SNAPSHOTS_DIR
from eval_chunking.embedders import EmbedderUnavailable, make_embedder
from eval_chunking.retrieval import embed_chunks, retrieve
from eval_chunking.scoring import RETRIEVE_K
from eval_chunking.task import Task

MAX_RAW_OUTPUT_CHARS = 2000

# The naive sentinel + the canonical strategies. All free + local, so the default
# run is keyless (it only needs the offline tfidf-local embedder).
DEFAULT_CHUNKERS = ("fixed-token", "recursive", "sentence", "semantic", "cluster-semantic")


def run_cost(records: Iterable[ItemResult]) -> float:
    return sum(r.cost_usd or 0.0 for r in records)


def _spec(name: str) -> AdapterSpec:
    cls = get(name)
    return AdapterSpec(
        name=name,
        primitive=Primitive.CHUNKING,
        vendor=getattr(cls, "vendor", "local"),
        version=getattr(cls, "model_version", "unknown"),
        is_sentinel=getattr(cls, "is_sentinel", False),
        params={
            "chunk_size": getattr(cls, "chunk_size", None),
            "chunk_overlap": getattr(cls, "chunk_overlap", None),
        },
    )


def _env_versions(specs: list[AdapterSpec], embedder_name: str) -> dict[str, str]:
    """Allow-listed package versions only — never raw os.environ (no key leakage)."""
    env: dict[str, str] = {"python": platform.python_version(), "embedder": embedder_name}
    for pkg in ("eval-chunking", "bench-adapters", "bench-stats", "scikit-learn",
                "sentence-transformers"):
        try:
            env[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            pass
    return env


def _item_result(run_id: str, adapter: str, item: dict[str, Any], out: Any) -> ItemResult:
    tier = item.get("ground_truth_tier")
    return ItemResult(
        run_id=run_id,
        adapter=adapter,
        item_id=str(item["id"]),
        primitive=Primitive.CHUNKING,
        slices=item.get("slices", []),
        ground_truth_tier=GroundTruthTier(tier) if tier else None,
        output=out,
        latency_ms=None,
        cost_usd=0.0,
        # `error` marks a genuine chunker/transport failure only — uncharged data
        # reasons (empty_chunking, references_unresolved) live on output.miss_reason.
        error=out.rationale if out.miss_reason == "failed" else None,
    )


def _manifest(run_id: str, seed: int, specs: list[AdapterSpec], notes: str,
              embedder_name: str, n_items: int, k: int) -> RunManifest:
    progress = f"items/chunker={n_items}; chunkers={len(specs)}; k={k}; embedder={embedder_name}"
    return RunManifest(
        run_id=run_id,
        primitive=Primitive.CHUNKING,
        created_at=datetime.now(timezone.utc),
        seed=seed,
        adapters=specs,
        task_version=Task.task_version,
        dataset_version=Task.dataset_version,
        split="public_dev",
        canary_guid=CANARY,
        env=_env_versions(specs, embedder_name),
        notes=(notes + " | " if notes else "") + progress,
    )


def run_sync(
    rows: Iterable[dict[str, Any]],
    corpora: dict[str, str],
    run_id: str,
    *,
    chunkers: Iterable[str] = DEFAULT_CHUNKERS,
    embedder: str = "tfidf-local",
    k: int = RETRIEVE_K,
    seed: int = 0,
    limit: int | None = None,
    out_root: str = "runs",
    notes: str = "",
    write_snapshot: bool = True,
) -> list[ItemResult]:
    """Run the chunking probe over `rows` against `corpora`; write the run dir."""
    task = Task(rows)
    items = list(task.items())
    if limit:
        items = items[:limit]
    scorer = task.scorer()

    names = [n for n in chunkers if n in registry]
    skipped = [n for n in chunkers if n not in registry]
    rundir = RunDir(out_root, run_id)

    # Queries grouped by corpus; chunk + embed per corpus, reused across chunkers.
    by_corpus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_corpus[str(it["corpus_id"])].append(it)

    emb = make_embedder(embedder)
    all_records: list[ItemResult] = []
    used_chunkers: set[str] = set()

    for cid, queries in by_corpus.items():
        corpus_text = corpora.get(cid)
        if not corpus_text:
            print(f"[chunking] corpus '{cid}' missing; skipping {len(queries)} queries")
            continue
        try:
            emb.fit(corpus_text)  # one fit per corpus, shared across chunkers
        except EmbedderUnavailable as exc:
            print(f"[chunking] embedder '{embedder}' unavailable: {exc} -> aborting run")
            break

        for name in names:
            adapter = get(name)(_spec(name))
            try:
                chunks = adapter.invoke({"document": corpus_text, "embed": emb.embed})["chunks"]
                chunk_embs = embed_chunks(emb, chunks)
            except VendorUnavailable as exc:
                print(f"[chunking] {name} unavailable on '{cid}', skipping lane: {exc}")
                continue
            except EmbedderUnavailable as exc:
                print(f"[chunking] embedder unavailable for {name} on '{cid}': {exc}")
                break
            except Exception as exc:  # chunking failed for this corpus -> uncharged misses
                for q in queries:
                    rec = _item_result(run_id, name, q,
                                       scorer.score(q, {"error": repr(exc)[:200]}))
                    rundir.append_item(rec)
                    all_records.append(rec)
                used_chunkers.add(name)
                continue

            for q in queries:
                retrieved = retrieve(emb, chunk_embs, chunks, q["query"], k)
                out = scorer.score(q, {"retrieved": retrieved, "all_chunks": chunks})
                rec = _item_result(run_id, name, q, out)
                rundir.append_item(rec)
                all_records.append(rec)
            used_chunkers.add(name)

    specs = [_spec(n) for n in names if n in used_chunkers]
    n_per = len(items)
    rundir.write_manifest(_manifest(run_id, seed, specs, notes, emb.name, n_per, k))
    slice_results, _board = report.run(all_records)
    rundir.write_slices(slice_results)
    if write_snapshot and all_records:
        snap = report.write_counts_toml(
            all_records, SNAPSHOTS_DIR / "chunking.counts.toml",
            run_id=run_id, citation="packages/eval-chunking/README.md",
        )
        print(f"[chunking] wrote snapshot -> {snap}")

    spent = run_cost(all_records)
    print(f"[chunking] {len(all_records)} (chunker,query) cells across {len(specs)} chunkers, "
          f"{n_per} queries; embedder={emb.name}; embed_cost=${emb.cost_usd:.4f}; "
          f"item_cost=${spent:.4f}"
          + (f"; not registered: {skipped}" if skipped else ""))
    return all_records
