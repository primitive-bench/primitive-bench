# golden-sets-public / retrieval

PUBLIC dev split for the retrieval primitive. Canary-marked; reproducible by anyone.

**Held-out test answers NEVER live here** — they sit behind the private eval
server (primitivebench-platform/apps/eval-server) with HMAC-keyed split integrity.

## What ships here vs. generated locally

Mirroring the OCR / reranker split model:

- `dev.example.jsonl` — a tiny, **self-authored** illustrative set (CC-BY-4.0),
  committed. Enough to run the loop end-to-end with the keyless `bge-small` adapter.
- `dev.jsonl` — the **full** organized subset, generated locally and **git-ignored**
  (never redistributed). Build it with:
  ```bash
  uv run python packages/eval-retrieval/tools/ingest_beir.py --per-source 200 --seed 0
  ```
  This writes ~600 rows (200 per domain) plus a committed
  `packages/eval-retrieval/selection_manifest.json` (the chosen query ids + seed) so
  the exact subset is auditable without re-downloading.
- `CANARY` — the canary GUID for contamination detection (BIG-bench convention),
  embedded as the header comment of each split file (see `../CANARY`).

## Row schema

```jsonc
{ "row_id": "scientific_1234",
  "domain": "scientific|medical|financial",
  "query": "…",
  "candidates": [ {"id": "d0", "text": "…"}, … ],   // per-query pool; order is uninformative
  "relevance": { "d0": 1, "d1": 0, … },              // BEIR qrels grade (rel>0 = relevant)
  "n_relevant": 2,
  "slices": ["domain:scientific", "relevant_set:multi"],
  "ground_truth_tier": "verified_external",
  "source": "BeIR/scifact",
  "canary": "PRIMITIVEBENCH-CANARY-GUID:…" }
```

A bi-encoder retriever ranks `candidates`; the scorer reads `relevance` in rank order
and computes **success@10** (the leaderboard separability gate) plus
nDCG@10 / recall@10 / MAP / MRR. Slices are assigned deterministically from the pool
(see `packages/eval-retrieval/src/eval_retrieval/slicing.py`).

## Per-query pools (why not the whole corpus)

True first-stage retrieval is over an entire corpus. To keep the eval keyless,
deterministic, and bounded, each row ships a **per-query candidate pool**: the query's
relevant docs plus BM25 top-N hard negatives mined offline from that corpus (the
standard BEIR "rerank the BM25 pool" setup). This isolates *embedding quality* — the
thing that separates retrievers — from corpus-indexing engineering (which the vectordb
primitive owns), and lets a free local bi-encoder run the whole split on CPU.

## Organized subset (why a sample, not the whole set)

The ingest tool takes a **deterministic, seeded, relevant-set-stratified sample** of
`--per-source` queries per source (round-robin across the single/multi buckets), not a
biased head-N — representative and powered for separability. The recipe is pinned in
`Task.dataset_version` (`retrieval-2026.06.scifact+nfcorpus+fiqa.n600.seed0`);
re-running with the same `--seed` reproduces the subset byte-for-byte.

## Provenance & license

Generated from three BEIR retrieval sets:

- [`BeIR/scifact`](https://huggingface.co/datasets/BeIR/scifact) — scientific claim
  verification; `domain=scientific`.
- [`BeIR/nfcorpus`](https://huggingface.co/datasets/BeIR/nfcorpus) — biomedical /
  nutrition IR; `domain=medical`.
- [`BeIR/fiqa`](https://huggingface.co/datasets/BeIR/fiqa) — financial-domain question
  answering; `domain=financial`.

BEIR corpora carry their own upstream licenses; like OCR's source PDFs they are
ingested locally and **not redistributed** from this repo. `dev.example.jsonl` is
original, authored by the Primitive Bench team and released under CC-BY-4.0.
