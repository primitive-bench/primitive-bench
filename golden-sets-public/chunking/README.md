# golden-sets-public / chunking

PUBLIC dev split for the chunking primitive. Canary-marked; reproducible by anyone.

**Held-out test answers NEVER live here** — they sit behind the private eval
server (primitivebench-platform/apps/eval-server) with HMAC-keyed split integrity.

## What ships here vs. generated locally

Mirroring the reranker split model:

- `dev.example.jsonl` + `corpora/example_*.md` — a tiny, **self-authored**
  illustrative set (CC-BY-4.0), committed. Enough to run the loop end-to-end with the
  keyless, offline `tfidf-local` embedder.
- `dev.jsonl` + `corpora/{chatlogs,finance,pubmed,state_of_the_union,wikitexts}.md` —
  the **full** organized subset, generated locally and **git-ignored** (never
  redistributed). Build it with:
  ```bash
  uv run python packages/eval-chunking/tools/ingest_chroma_chunking.py --per-source 40 --seed 0
  ```
  This writes ~200 question rows (40 per corpus) + the five full corpora, plus a
  committed `packages/eval-chunking/selection_manifest.json` (the chosen `row_id`s,
  seed, and pinned source commit) so the exact subset is auditable without
  re-downloading.
- `CANARY` — the canary GUID for contamination detection (BIG-bench convention),
  embedded as the header comment of each split file.

## Row schema

A row is a **query + gold reference spans + a `corpus_id`**; the corpus text itself is
a sibling file under `corpora/` (inlining a 0.5 MB corpus into every row would be
absurd), and the reference offsets index into that corpus.

```jsonc
{ "row_id": "finance_001234",
  "corpus_id": "finance",                          // -> corpora/finance.md
  "question": "…",
  "references": [                                   // gold spans (character offsets)
    { "content": "…", "start_index": 27346, "end_index": 27425 }, … ],
  "slices": ["domain:finance", "reference_dispersion:multi"],
  "ground_truth_tier": "verified_external",
  "source": "brandonstarxel/chunking_evaluation@e708410d1c",
  "canary": "PRIMITIVEBENCH-CANARY-GUID:…" }
```

The chunker chunks `corpora/<corpus_id>.md`; the eval embeds the chunks with a fixed
embedder, retrieves top-k for the `question`, and scores recovery of `references` —
**coverage@k** (the leaderboard separability gate) plus recall / precision / IoU /
precision_Ω. Slices are assigned deterministically from the corpus + the reference
spans (see `packages/eval-chunking/src/eval_chunking/slicing.py`).

## Organized subset (why a sample, not the whole set)

The ingest tool takes a **deterministic, seeded, dispersion-stratified sample** of
`--per-source` questions per corpus (round-robin across the single/multi buckets),
not a biased head-N — representative and balanced across the `reference_dispersion`
slices. The recipe is pinned in `Task.dataset_version`
(`chunking-2026.06.chroma5.n200.seed0`); re-running with the same `--seed` reproduces
the subset byte-for-byte. The five full corpora are always written in full (the
reference offsets index into them).

## Provenance & license

Generated from Chroma's
[`chunking_evaluation`](https://github.com/brandonstarxel/chunking_evaluation)
general-evaluation set (**MIT**) — the canonical, citable chunking benchmark from
*"Evaluating Chunking Strategies for Retrieval"* (2024). Five corpora —
`chatlogs`, `finance`, `pubmed`, `state_of_the_union`, `wikitexts` — each with
human-curated reference excerpts carrying exact character offsets. Pulled at a pinned
commit and ingested locally; **not redistributed** from this repo.

`dev.example.jsonl` and `corpora/example_*.md` are original, authored by the Primitive
Bench team and released under CC-BY-4.0.
