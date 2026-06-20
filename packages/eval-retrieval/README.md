# eval-retrieval

Primitive Bench eval vertical for **retrieval** (embedding bi-encoders). One query +
a fixed per-query candidate pool with graded relevance → the adapter ranks the pool by
cosine similarity → the scorer reports per-slice **success@10** with separability
badges, plus nDCG@10 / recall@10 / MAP / MRR.

## Why success@10 (and why a candidate pool)

- **success@10** (any relevant doc in the top-10) is a *paired-binary* outcome, which
  is what the leaderboard's McNemar/Wilson separability gate consumes — the same stance
  as OCR pass@test and websearch hit_rate. First-stage retrieval is judged on getting
  relevant docs *into* the top-k, so success@10 is the gate; nDCG@10/recall@10/MAP/MRR
  ride along in `ScorerOutput.metrics`. (Contrast the reranker vertical, which gates on
  hit@1 — "is the single best candidate on top?".)
- **Per-query pools** (qrels positives + BM25 top-N hard negatives, mined offline in
  `tools/ingest_beir.py`) keep the eval keyless and deterministic and isolate *embedding
  quality* from corpus-indexing engineering (which the vectordb primitive owns). The
  metrics themselves are the BEIR/MTEB-standard ones in `bench_stats.retrieval`.

## Slices ("one winner is a lie")

Assigned deterministically from row facts (`slicing.py`), never a model:

- `domain` — `scientific` (SciFact) / `medical` (NFCorpus) / `financial` (FiQA). The
  primary separator: embedding models trade places across domains.
- `relevant_set` — `single` (one relevant doc, all-or-nothing) vs. `multi` (rewards
  broad recall).

A slice publishes a winner only when adapters are Wilson-clear **and** McNemar-separable
on success@10; otherwise it reports a TIE band.

## Adapters

Registered in `bench_adapters.retrieval` — bi-encoders that embed query + each candidate
and rank by cosine:

| name | model | key | cost |
|---|---|---|---|
| `bge-small` | BAAI/bge-small-en-v1.5 (local) | — | $0 (sentinel) |
| `e5-small` | intfloat/e5-small-v2 (local) | — | $0 |
| `openai-embed` | text-embedding-3-large | `OPENAI_API_KEY` | list price (pricing.py) |
| `openai-embed-small` | text-embedding-3-small | `OPENAI_API_KEY` | list price |
| `cohere-embed` | embed-v4.0 | `COHERE_API_KEY` | list price |
| `voyage-embed` | voyage-4-large | `VOYAGE_API_KEY` | list price |

Hosted lanes are skipped uncharged when their key is unset (`VendorUnavailable`).

## Run

```bash
# keyless smoke test (free local bi-encoder; downloads the model the first time):
uv run python -m eval_retrieval \
    --golden golden-sets-public/retrieval/dev.example.jsonl \
    --adapters bge-small --run-id retrieval-smoke

# full subset (build dev.jsonl first; hosted lanes need keys in .env):
uv run python packages/eval-retrieval/tools/ingest_beir.py --per-source 200 --seed 0
uv run python -m eval_retrieval --golden golden-sets-public/retrieval/dev.jsonl --run-id retrieval-dev
```

Each run streams `ItemResult`s to `runs/<run-id>/items.jsonl`, writes a `RunManifest`
and per-slice `SliceResult`s, and emits the curated `snapshots/retrieval.counts.toml`
the `bench results emit` pipeline turns into the public leaderboard JSON.
