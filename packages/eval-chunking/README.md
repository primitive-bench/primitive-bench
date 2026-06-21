# eval-chunking

Primitive Bench vertical for **chunking** — *downstream retrieval quality by chunk
strategy*. A chunker is not judged on how pretty its chunks look; it is judged on what
happens **downstream**: chunk a corpus, embed the chunks with a **fixed** embedder,
retrieve top-k for each query, and measure how well the retrieved chunks recover the
query's gold reference spans. This is the methodology of Chroma's *"Evaluating
Chunking Strategies for Retrieval"* (2024), expressed against the frozen
`bench-schemas` contract.

> **One winner is a lie.** Results are per-slice (domain × reference dispersion) with
> Wilson intervals and a McNemar separability gate — never one global ranking.

## What it measures

For each (chunker, query) cell, with `retrieved` = the top-k chunks and `references` =
the gold character spans (Chroma curated), everything is a character-range overlap
(see `ranges.py`):

| metric | definition | role |
|---|---|---|
| **recall** | \|retrieved ∩ refs\| / \|refs\| | headline continuous score (Chroma primary) |
| **precision** | \|retrieved ∩ refs\| / \|retrieved\| | over/under-chunking penalty (reported) |
| **IoU** | \|retrieved ∩ refs\| / \|retrieved ∪ refs\| | Jaccard (reported) |
| **precision_Ω** | ceiling precision the chunk *granularity* allows | reported |
| **coverage@k** | `recall@k ≥ 0.8` (a paired binary) | **the leaderboard separability surface** |

`coverage@k` is the paired-binary outcome the McNemar/Wilson gate consumes (same
stance as the reranker's hit@1 / OCR's pass@test): one binary for the gate, the
continuous recall/precision/IoU ride along in `ScorerOutput.metrics`. `RECALL_TARGET =
0.8` is a pinned methodology constant — "the answer evidence is substantially
present" — discriminative because a strategy that fragments the answer across chunks
falls below it while one that keeps it together clears it. Precision is always
reported alongside recall because that is where chunkers diverge most: an
over-chunking strategy buys recall at a steep precision cost.

Non-attempts are uncharged (`correct=None`), never failures: a chunker error, an empty
chunking, or a row whose references have zero measure are excluded from the
denominator.

## Systems under test (the chunkers)

The chunkers are `bench_adapters` chunk adapters (`document -> character-span chunks`).
**Chunk size is held constant at 200 tokens** across strategies so the comparison
isolates *boundary placement*, not size.

| adapter | strategy | notes |
|---|---|---|
| `fixed-token` | fixed token windows | naive baseline / regression **sentinel** |
| `recursive` | recursive separator split | LangChain-style; the production default |
| `sentence` | greedy sentence packing | never cuts a sentence |
| `semantic` | embedding breakpoint (Kamradt) | uses the shared embedder |
| `cluster-semantic` | Chroma ClusterSemanticChunker | DP over piece similarities; uses the shared embedder |

## The embedder is held constant

For the comparison to isolate the chunker, the embedder is identical across all
chunkers (and is even injected into the semantic chunkers' boundary detection):

- **`tfidf-local` (default)** — a corpus-fit TF-IDF retriever (scikit-learn). No
  download, no key, deterministic → the whole loop runs offline and in CI. Held
  constant, it ranks chunkers fairly.
- **Production embedders** — the best available dense models for the published
  snapshot: local sentence-transformers (`st:BAAI/bge-large-en-v1.5`,
  `arctic-l`, …) and hosted SOTA APIs (OpenAI `text-embedding-3-large`, Voyage
  `voyage-3-large`, Cohere `embed-v4.0`, Jina `jina-embeddings-v3`). Pin whichever
  you publish in `Task.dataset_version`.

## Slices (`slices.yaml`)

- **`domain`** — the source corpus (`chatlogs`, `finance`, `pubmed`,
  `state_of_the_union`, `wikitexts`). THE primary separator: document structure
  differs sharply, so the chunker that respects one structure wrecks another.
- **`reference_dispersion`** — `single` (answer in one localized span) vs `multi`
  (scattered across disjoint spans). A dispersed answer is the sharp test of
  chunking. Assigned deterministically from the gold spans (`slicing.py`) — no model.

## Run it

```bash
# keyless, fully-offline smoke test (committed example corpora + tfidf-local):
uv run python -m eval_chunking \
    --golden golden-sets-public/chunking/dev.example.jsonl --run-id chunking-smoke

# 1) generate the full Chroma subset locally (downloads a few MB from GitHub):
uv run python packages/eval-chunking/tools/ingest_chroma_chunking.py --per-source 40 --seed 0
# 2) score it (offline, reproducible baseline):
uv run python -m eval_chunking \
    --golden golden-sets-public/chunking/dev.jsonl --run-id chunking-public-snapshot

# publish-grade run with a SOTA dense embedder (needs the key / a download):
uv run python -m eval_chunking --golden golden-sets-public/chunking/dev.jsonl \
    --embedder voyage --run-id chunking-voyage
```

A run writes `runs/<run_id>/{manifest.json,items.jsonl,slices.jsonl}` and refreshes
`snapshots/chunking.counts.toml`, which `bench results emit` turns into the public
leaderboard JSON the MCP server serves.

## Published snapshot

`snapshots/chunking.counts.toml` is a **real, measured** run over the 200-row Chroma
subset (40 questions × 5 corpora) with the offline `tfidf-local` embedder — fully
reproducible by anyone with the steps above. Because that lexical embedder is held
constant it ranks chunkers fairly, but its absolute scores are conservative, so every
slice currently reports a **TIE band** at this n (the methodologically correct
outcome — no strategy is statistically separable here). The headline production
experiment is the same run with a SOTA dense embedder (`--embedder voyage` /
`text-embedding-3-large`); re-running `bench results emit` republishes the board.

## Provenance & license

Dataset: Chroma [`chunking_evaluation`](https://github.com/brandonstarxel/chunking_evaluation)
general-evaluation set (MIT), pulled at a pinned commit by the ingest tool — five
corpora (chatlogs, finance, pubmed, state_of_the_union, wikitexts) with
human-curated reference excerpts. Pulled locally and **not redistributed** from this
repo; only the self-authored `dev.example.jsonl` (CC-BY-4.0) is committed. Code is
Apache-2.0.
