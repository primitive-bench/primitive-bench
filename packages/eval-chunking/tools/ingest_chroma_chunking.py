"""Generate the chunking dev split locally from the Chroma evaluation set (NOT committed).

Source: Chroma's ``chunking_evaluation`` general-evaluation data — the canonical,
citable chunking benchmark ("Evaluating Chunking Strategies for Retrieval", 2024).
Five corpora give the ``domain`` slice that actually separates chunkers:

    chatlogs · finance · pubmed · state_of_the_union · wikitexts

Each question ships with human-curated **reference excerpts** carrying exact character
offsets into its corpus (``{content, start_index, end_index}``) — a ready-made gold
for token-range recall/precision, so there is NO LLM-graded step. We pull the data
from GitHub at a PINNED commit (reachable even where the HF hub is not), write the
full corpora under ``golden-sets-public/chunking/corpora/`` (git-ignored, not
redistributed) and a git-ignored ``dev.jsonl``; only the tiny self-authored
``dev.example.jsonl`` is committed.

Determinism (D-14): a seeded, dispersion-stratified sample of ``--per-source``
questions per corpus; the chosen row ids + seed + the pinned source commit are written
to ``selection_manifest.json`` (committed) so the exact subset is auditable, and the
recipe is pinned in ``Task.dataset_version``.

Run (downloads a few MB from GitHub):
    uv run python packages/eval-chunking/tools/ingest_chroma_chunking.py --per-source 40 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_chunking._paths import CANARY, GOLDEN_CHUNKING_DIR
from eval_chunking.slicing import reference_dispersion, slice_keys

# Pinned Chroma chunking_evaluation commit (general_evaluation_data). Refresh
# deliberately, never silently — the SHA is part of the dataset provenance.
SOURCE_REPO = "brandonstarxel/chunking_evaluation"
SOURCE_COMMIT = "e708410d1c61cb76a85cd9d433630ef89b9c6b85"
_BASE = (f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}"
         "/chunking_evaluation/evaluation_framework/general_evaluation_data")

CORPORA = ["chatlogs", "finance", "pubmed", "state_of_the_union", "wikitexts"]
MANIFEST = Path(__file__).resolve().parents[1] / "selection_manifest.json"


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (pinned https)
        return r.read()


def _load_questions() -> list[dict[str, Any]]:
    text = _fetch(f"{_BASE}/questions_df.csv").decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        refs = json.loads(r["references"])
        if not refs:
            continue
        out.append({
            "_idx": i,
            "corpus_id": r["corpus_id"],
            "question": r["question"],
            "references": refs,
            "dispersion": reference_dispersion(refs),
        })
    return out


def _stratified_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Seeded, dispersion-balanced sample of n rows (round-robin across strata).

    Representative (seeded shuffle, not head-N) and deterministic; balancing across
    single/multi keeps both reference_dispersion slices populated."""
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["dispersion"]].append(r)
    for g in groups.values():
        g.sort(key=lambda x: x["_idx"])
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


def _build_row(q: dict[str, Any], seed: int) -> dict[str, Any]:
    cid = q["corpus_id"]
    row_id = f"{cid}_{q['_idx']:06d}"
    return {
        "row_id": row_id,
        "corpus_id": cid,
        "question": q["question"],
        "references": q["references"],
        "slices": slice_keys(cid, q["references"]),
        "ground_truth_tier": "verified_external",
        "source": f"{SOURCE_REPO}@{SOURCE_COMMIT[:10]}",
        "canary": CANARY,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-source", type=int, default=40,
                    help="questions sampled per corpus (default 40 -> ~200 total)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(GOLDEN_CHUNKING_DIR / "dev.jsonl"))
    args = ap.parse_args()

    # 1) Corpora (full text; references index into these). Written but not redistributed.
    corpora_dir = GOLDEN_CHUNKING_DIR / "corpora"
    corpora_dir.mkdir(parents=True, exist_ok=True)
    for cid in CORPORA:
        text = _fetch(f"{_BASE}/corpora/{cid}.md").decode("utf-8")
        (corpora_dir / f"{cid}.md").write_text(text, encoding="utf-8")
        print(f"  corpus {cid}: {len(text)} chars")

    # 2) Questions -> seeded, dispersion-stratified per-corpus sample.
    questions = _load_questions()
    by_corpus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        by_corpus[q["corpus_id"]].append(q)

    out_rows: list[dict[str, Any]] = []
    manifest_sources: list[dict[str, Any]] = []
    for cid in CORPORA:
        eligible = by_corpus.get(cid, [])
        chosen = _stratified_sample(eligible, args.per_source, args.seed)
        out_rows.extend(_build_row(q, args.seed) for q in chosen)
        manifest_sources.append({
            "corpus_id": cid, "eligible": len(eligible), "chosen": len(chosen),
            "row_ids": [f"{cid}_{q['_idx']:06d}" for q in chosen],
        })
        print(f"  {cid}: {len(chosen)}/{len(eligible)} eligible questions sampled")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# {CANARY}  (do not train on this file; see ../CANARY)\n")
        f.write(f"# Generated locally from {SOURCE_REPO}@{SOURCE_COMMIT[:10]}; not redistributed (see README).\n")
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    MANIFEST.write_text(json.dumps({
        "dataset_version": f"chunking-2026.06.chroma5.n{len(out_rows)}.seed{args.seed}",
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "per_source": args.per_source,
        "seed": args.seed,
        "total_rows": len(out_rows),
        "sources": manifest_sources,
    }, indent=2) + "\n", encoding="utf-8")

    by_domain = defaultdict(int)
    for r in out_rows:
        by_domain[r["corpus_id"]] += 1
    print(f"wrote {len(out_rows)} rows ({dict(by_domain)}) -> {out_path}")
    print(f"wrote selection manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
