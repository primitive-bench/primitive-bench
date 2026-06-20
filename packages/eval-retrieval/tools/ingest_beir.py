"""Generate the retrieval dev split locally from BEIR sets (NOT committed).

Three BEIR retrieval sources give the `domain` slice that actually separates
embedding retrievers:
  * BeIR/scifact   -> domain=scientific  (scientific claim verification)
  * BeIR/nfcorpus  -> domain=medical     (biomedical / nutrition IR)
  * BeIR/fiqa      -> domain=financial   (financial-domain question answering)

BEIR ships each set as a `corpus` (id, title, text), a `queries` (id, text), and a
qrels table (query-id, corpus-id, score). True first-stage retrieval is over the whole
corpus; to keep the eval keyless, deterministic, and bounded we instead ship a
**per-query candidate pool** — the query's relevant docs plus BM25 top-N hard
negatives mined offline from that corpus (the standard BEIR "rerank the BM25 pool"
setup). The bi-encoder retrievers then rank that pool. We pull the sources to the
user's machine and emit a git-ignored `dev.jsonl`; only the tiny self-authored
`dev.example.jsonl` is committed.

Determinism (D-14): a seeded, relevant-set-stratified sample of `--per-source` queries
per source; each pool is shuffled with a row-stable seed so candidate order never leaks
relevance, and candidate ids are reassigned `d0..dN` so corpus ids don't leak either.
The chosen query ids + seed are written to `selection_manifest.json` (committed) so the
exact subset is auditable, and the recipe is pinned in `Task.dataset_version`.

Run (downloads a few hundred MB the first time):
    uv run python packages/eval-retrieval/tools/ingest_beir.py --per-source 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_retrieval._paths import CANARY, GOLDEN_RETRIEVAL_DIR
from eval_retrieval.slicing import relevant_set_bucket, slice_keys

# (BEIR HuggingFace name, domain label, qrels split).
SOURCES = [
    ("scifact", "scientific", "test"),
    ("nfcorpus", "medical", "test"),
    ("fiqa", "financial", "test"),
]

MANIFEST = Path(__file__).resolve().parents[1] / "selection_manifest.json"
DOC_CHARS = 600           # truncate each candidate doc to bound embedding tokens
_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _doc_text(rec: dict[str, Any]) -> str:
    """Combine BEIR corpus title + text, truncated."""
    title = str(rec.get("title") or "").strip()
    body = str(rec.get("text") or "").strip()
    joined = f"{title}. {body}".strip(". ").strip() if title else body
    return joined[:DOC_CHARS]


def _load_corpus(name: str) -> dict[str, str]:
    from datasets import load_dataset
    ds = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    return {str(r["_id"]): _doc_text(r) for r in ds}


def _load_queries(name: str) -> dict[str, str]:
    from datasets import load_dataset
    ds = load_dataset(f"BeIR/{name}", "queries", split="queries")
    return {str(r["_id"]): str(r.get("text") or "").strip() for r in ds}


def _load_qrels(name: str, split: str) -> dict[str, dict[str, int]]:
    from datasets import load_dataset
    ds = load_dataset(f"BeIR/{name}-qrels", split=split)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for r in ds:
        score = int(r["score"])
        if score > 0:
            qrels[str(r["query-id"])][str(r["corpus-id"])] = score
    return qrels


def _eligible_rows(
    corpus: dict[str, str], queries: dict[str, str], qrels: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    """Queries usable for retrieval: non-empty text and >=1 relevant doc present in corpus."""
    rows: list[dict[str, Any]] = []
    for qid, rels in qrels.items():
        query = queries.get(qid, "").strip()
        pos = {d: s for d, s in rels.items() if d in corpus}
        if not query or not pos:
            continue
        n_rel = len(pos)
        rows.append({
            "qid": qid,
            "query": query,
            "positives": pos,                     # {doc_id: score}
            "relevant_bucket": relevant_set_bucket(n_rel),
        })
    return rows


def _stratified_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Seeded sample of n rows balanced across the relevant-set buckets (single/multi)."""
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["relevant_bucket"]].append(r)
    for g in groups.values():
        g.sort(key=lambda x: x["qid"])
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


class _BM25:
    """Tiny deterministic BM25Okapi wrapper over a fixed corpus (rank_bm25)."""

    def __init__(self, corpus: dict[str, str]) -> None:
        from rank_bm25 import BM25Okapi
        self.ids = list(corpus)
        self.bm25 = BM25Okapi([_tok(corpus[i]) for i in self.ids])

    def top_negatives(self, query: str, exclude: set[str], k: int) -> list[str]:
        scores = self.bm25.get_scores(_tok(query))
        order = sorted(range(len(self.ids)), key=lambda i: -scores[i])
        out: list[str] = []
        for i in order:
            doc_id = self.ids[i]
            if doc_id in exclude:
                continue
            out.append(doc_id)
            if len(out) >= k:
                break
        return out


def _build_row(src_row: dict[str, Any], corpus: dict[str, str], bm25: _BM25,
               domain: str, source: str, pool: int, seed: int) -> dict[str, Any]:
    """Pool positives + BM25 hard negatives, shuffle (row-stable seed) so order is
    uninformative, reassign ids d0..dN, and tag slices. Relevance carries the qrels grade."""
    row_id = f"{domain}_{src_row['qid']}"
    positives = src_row["positives"]                       # {doc_id: score}
    n_neg = max(0, pool - len(positives))
    negatives = bm25.top_negatives(src_row["query"], set(positives), n_neg)

    items = [(d, corpus[d], int(s)) for d, s in positives.items()]
    items += [(d, corpus[d], 0) for d in negatives]
    random.Random(f"{seed}:{row_id}").shuffle(items)

    candidates = [{"id": f"d{i}", "text": text} for i, (_d, text, _s) in enumerate(items)]
    relevance = {f"d{i}": s for i, (_d, _text, s) in enumerate(items)}
    n_relevant = sum(1 for s in relevance.values() if s > 0)
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
        "source": f"BeIR/{source}",
        "canary": CANARY,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-source", type=int, default=200,
                    help="queries sampled per source (default 200 -> ~600 total)")
    ap.add_argument("--pool", type=int, default=100,
                    help="candidate pool size per query (positives + BM25 hard negatives)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(GOLDEN_RETRIEVAL_DIR / "dev.jsonl"))
    args = ap.parse_args()

    out_rows: list[dict[str, Any]] = []
    manifest_sources: list[dict[str, Any]] = []
    for name, domain, split in SOURCES:
        print(f"loading BeIR/{name} [{split}] ...")
        corpus = _load_corpus(name)
        queries = _load_queries(name)
        qrels = _load_qrels(name, split)
        eligible = _eligible_rows(corpus, queries, qrels)
        chosen = _stratified_sample(eligible, args.per_source, args.seed)
        print(f"  building BM25 over {len(corpus)} docs ...")
        bm25 = _BM25(corpus)
        for src_row in chosen:
            out_rows.append(_build_row(src_row, corpus, bm25, domain, name, args.pool, args.seed))
        manifest_sources.append({
            "repo": f"BeIR/{name}", "domain": domain, "split": split,
            "corpus_docs": len(corpus), "eligible": len(eligible), "chosen": len(chosen),
            "query_ids": [r["qid"] for r in chosen],
        })
        print(f"  BeIR/{name}: {len(chosen)}/{len(eligible)} eligible queries sampled")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# {CANARY}  (do not train on this file; see ../CANARY)\n")
        f.write("# Generated locally from BEIR sets (BM25-pooled); not redistributed (see README).\n")
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    MANIFEST.write_text(json.dumps({
        "dataset_version": "retrieval-2026.06.scifact+nfcorpus+fiqa.n600.seed0",
        "per_source": args.per_source,
        "pool": args.pool,
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
