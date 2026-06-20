# golden-sets-public / reranker

PUBLIC dev split for the reranker primitive. Canary-marked; reproducible by anyone.

**Held-out test answers NEVER live here** — they sit behind the private eval
server (primitivebench-platform/apps/eval-server) with HMAC-keyed split integrity.

## What ships here vs. generated locally

Mirroring the OCR split model:

- `dev.example.jsonl` — a tiny, **self-authored** illustrative set (CC-BY-4.0),
  committed. Enough to run the loop end-to-end with the keyless `ce-minilm` adapter.
- `dev.jsonl` — the **full** organized subset, generated locally and **git-ignored**
  (never redistributed). Build it with:
  ```bash
  uv run python packages/eval-reranker/tools/ingest_mteb_reranking.py --per-source 50 --seed 0
  ```
  This writes ~100 rows (50 per domain) plus a committed
  `packages/eval-reranker/selection_manifest.json` (the chosen `row_id`s + seed) so
  the exact subset is auditable without re-downloading.
- `CANARY` — the canary GUID for contamination detection (BIG-bench convention),
  embedded as the header comment of each split file.

## Row schema

```jsonc
{ "row_id": "scientific_001234",
  "domain": "scientific|tech_qa",
  "query": "…",
  "candidates": [ {"id": "d0", "text": "…"}, … ],   // shuffled; order is uninformative
  "relevance": { "d0": 1, "d1": 0, … },              // binary (positive=1)
  "n_relevant": 3,
  "slices": ["domain:scientific", "hard_negative_density:low"],
  "ground_truth_tier": "verified_external",
  "source": "mteb/scidocs-reranking",
  "canary": "PRIMITIVEBENCH-CANARY-GUID:…" }
```

The adapter reorders `candidates`; the scorer reads `relevance` in rank order and
computes **hit@1** (the leaderboard separability gate) plus nDCG@10 / MAP / MRR.
Slices are assigned deterministically from the candidate pool (see
`packages/eval-reranker/src/eval_reranker/slicing.py`).

## Organized subset (why a sample, not the whole set)

The ingest tool takes a **deterministic, seeded, density-stratified sample** of
`--per-source` rows per source (round-robin across the hard-negative-density
buckets), not a biased head-N — representative and powered for separability. The
recipe is pinned in `Task.dataset_version`
(`reranker-2026.06.scidocs+askubuntu.n100.seed0`); re-running with the same
`--seed` reproduces the subset byte-for-byte.

## Provenance & license

Generated from two MTEB reranking sets:

- [`mteb/scidocs-reranking`](https://huggingface.co/datasets/mteb/scidocs-reranking)
  — SciDocs (AllenAI) scientific paper recommendation; `domain=scientific`.
- [`mteb/askubuntudupquestions-reranking`](https://huggingface.co/datasets/mteb/askubuntudupquestions-reranking)
  — AskUbuntu duplicate-question retrieval; `domain=tech_qa`. Derived from
  StackExchange (**CC-BY-SA**) — like OCR's source PDFs, it is ingested locally and
  **not redistributed** from this repo.

`dev.example.jsonl` is original, authored by the Primitive Bench team and released
under CC-BY-4.0.
