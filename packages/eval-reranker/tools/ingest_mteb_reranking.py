"""Generate the reranker dev split locally from MTEB reranking sets (NOT committed).

Two sources give the `domain` slice that actually separates rerankers:
  * mteb/scidocs-reranking            -> domain=scientific  (paper recommendation)
  * mteb/askubuntudupquestions-reranking -> domain=tech_qa  (duplicate-question retrieval)

Both ship rows already in `{query, positive[], negative[]}` form — a ready-made
candidate list with human-judged relevance — so there is NO corpus, index, or BM25
step. We pull them to the user's machine and emit a git-ignored `dev.jsonl`; only
the tiny self-authored `dev.example.jsonl` is committed. AskUbuntu derives from
StackExchange (CC-BY-SA): like OCR's source PDFs it is ingested locally and never
redistributed from this repo.

Determinism (D-14): a seeded, density-stratified sample of `--per-source` rows per
source; candidate order is shuffled per-row with a row-stable seed so the input
order never leaks relevance. The chosen row ids + seed are written to
`selection_manifest.json` (committed) so the exact subset is auditable, and the
recipe is pinned in `Task.dataset_version`.

Run (downloads a few hundred MB the first time):
    uv run python packages/eval-reranker/tools/ingest_mteb_reranking.py --per-source 50 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_reranker._paths import CANARY, GOLDEN_RERANK_DIR
from eval_reranker.slicing import hard_negative_bucket, slice_keys

# (HuggingFace dataset id, domain label, split).
SOURCES = [
    ("mteb/scidocs-reranking", "scientific", "test"),
    ("mteb/askubuntudupquestions-reranking", "tech_qa", "test"),
]

MANIFEST = Path(__file__).resolve().parents[1] / "selection_manifest.json"


def _as_list(value: Any) -> list[str]:
    """MTEB positive/negative is a list of doc strings; tolerate a bare string."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _eligible_rows(ds: Any, domain: str) -> list[dict[str, Any]]:
    """Rows usable for reranking: non-empty query, >=1 positive and >=1 negative."""
    rows: list[dict[str, Any]] = []
    for i, ex in enumerate(ds):
        query = str(ex.get("query") or "").strip()
        pos = [t for t in _as_list(ex.get("positive")) if t.strip()]
        neg = [t for t in _as_list(ex.get("negative")) if t.strip()]
        if not query or not pos or not neg:
            continue
        n_cand = len(pos) + len(neg)
        density = len(neg) / n_cand
        rows.append({
            "_idx": i,
            "_sortkey": (query, i),         # stable order before the seeded shuffle
            "query": query,
            "positive": pos,
            "negative": neg,
            "n_candidates": n_cand,
            "density_bucket": hard_negative_bucket(density),
        })
    return rows


def _stratified_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Seeded, density-balanced sample of n rows (round-robin across strata).

    Representative (seeded shuffle, not head-N) and deterministic. Balancing across
    the hard-negative-density buckets keeps both the high/low slices populated."""
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["density_bucket"]].append(r)
    for g in groups.values():
        g.sort(key=lambda x: x["_sortkey"])
        rng.shuffle(g)
    keys = sorted(groups)
    idx = {k: 0 for k in keys}
    chosen: list[dict[str, Any]] = []
    while len(chosen) < n and any(idx[k] < len(groups[k]) for k in keys):
        for k in keys:
            if idx[k] < len(groups[k]):
                chosen.append(groups[k][idx[k]])
                idx[k] += 1
                if len(chosen) >= n:
                    break
    return chosen


def _build_row(src_row: dict[str, Any], domain: str, source: str, seed: int) -> dict[str, Any]:
    """Pool positives+negatives, shuffle (row-stable seed) so order is uninformative,
    assign stable ids, and tag slices. Relevance is binary (positive=1)."""
    row_id = f"{domain}_{src_row['_idx']:06d}"
    pool = [(t, 1) for t in src_row["positive"]] + [(t, 0) for t in src_row["negative"]]
    random.Random(f"{seed}:{row_id}").shuffle(pool)

    candidates = [{"id": f"d{i}", "text": text} for i, (text, _rel) in enumerate(pool)]
    relevance = {f"d{i}": rel for i, (_text, rel) in enumerate(pool)}
    n_relevant = sum(relevance.values())
    n_candidates = len(candidates)
    return {
        "row_id": row_id,
        "domain": domain,
        "query": src_row["query"],
        "candidates": candidates,
        "relevance": relevance,
        "n_relevant": n_relevant,
        "slices": slice_keys(domain, n_candidates, n_relevant),
        "ground_truth_tier": "verified_external",
        "source": source,
        "canary": CANARY,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-source", type=int, default=50,
                    help="rows sampled per source (default 50 -> ~100 total)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(GOLDEN_RERANK_DIR / "dev.jsonl"))
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"`datasets` required: {exc}") from exc

    out_rows: list[dict[str, Any]] = []
    manifest_sources: list[dict[str, Any]] = []
    for repo, domain, split in SOURCES:
        print(f"loading {repo} [{split}] ...")
        ds = load_dataset(repo, split=split)
        eligible = _eligible_rows(ds, domain)
        chosen = _stratified_sample(eligible, args.per_source, args.seed)
        for src_row in chosen:
            out_rows.append(_build_row(src_row, domain, repo, args.seed))
        manifest_sources.append({
            "repo": repo, "domain": domain, "split": split,
            "eligible": len(eligible), "chosen": len(chosen),
            "row_ids": [f"{domain}_{r['_idx']:06d}" for r in chosen],
        })
        print(f"  {repo}: {len(chosen)}/{len(eligible)} eligible rows sampled")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# {CANARY}  (do not train on this file; see ../CANARY)\n")
        f.write("# Generated locally from MTEB reranking sets; not redistributed (see README).\n")
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    MANIFEST.write_text(json.dumps({
        "dataset_version": f"reranker-2026.06.scidocs+askubuntu.n{len(out_rows)}.seed{args.seed}",
        "per_source": args.per_source,
        "seed": args.seed,
        "total_rows": len(out_rows),
        "sources": manifest_sources,
    }, indent=2) + "\n", encoding="utf-8")

    by_domain = defaultdict(int)
    for r in out_rows:
        by_domain[r["domain"]] += 1
    print(f"wrote {len(out_rows)} rows ({dict(by_domain)}) -> {out_path}")
    print(f"wrote selection manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
