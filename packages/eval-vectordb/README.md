# eval-vectordb

Primitive Bench vertical for **vector databases / ANN search**: build each engine's
index over a corpus, query it, and score **recall@10 against the exact nearest
neighbors** — reported per slice with QPS, latency, build time, index memory, and cost
alongside. Methodology follows [ANN-Benchmarks](https://ann-benchmarks.com) (recall vs
QPS, exact top-K ground truth, single-query timing) and
[VectorDBBench](https://github.com/zilliztech/VectorDBBench) (the QP$ cost dimension).

## What it measures

For each query the corpus has **K exact nearest neighbors** (ground truth, computed by
brute force → `GroundTruthTier.VERIFIED_EXTERNAL`). An engine returns its top-K base
ids; `recall@10 = |returned ∩ true| / K`.

**recall@10 is the separability surface, as a proportion.** Per slice we pool
`k = Σhits, n = ΣK` over queries → counts of the same shape as every other primitive, so
it flows through the frozen Wilson + `tied_rank_band` leaderboard pipeline unchanged
(`bench results emit`). A slice names a winner only when the leader's Wilson interval
clears the runner-up's; otherwise a TIE band — *one winner is a lie*.

The **continuous companions** — QPS, p50/p99 latency, build time, index memory, and
`$cost` (hosted) — ride in each item's `metrics` and are printed per engine, but never
gate the leaderboard (the same stance as reranker reporting nDCG@10 next to hit@1). The
`budget:fast` vs `budget:accurate` slices make the **recall-vs-QPS tradeoff** explicit:
the same engine separates differently at low vs high ef/nprobe.

## Slices

`dataset:<name>` · `metric:euclidean|angular` · `dim:low|mid|high` ·
`budget:fast|accurate` (see `slices.yaml`). Charged miss reasons: `missed_recall`,
`empty_result`. Uncharged non-attempts: `build_failed`, `oom`, `timeout`,
`adapter_error`.

## Engines (`bench_adapters.vectordb`)

Three tiers, registered in the shared adapter registry; a missing dep/key/service raises
`VendorUnavailable` and the lane skips uncharged:

- **OSS in-process** (keyless, the default runnable set): `bruteforce-numpy` (exact
  oracle / regression sentinel), `faiss-flat`, `faiss-hnsw`, `faiss-ivf`, `hnswlib`,
  `annoy`, `qdrant-local`, `lancedb`.
- **Dockerized servers** (`docker/docker-compose.vectordb.yml`): `pgvector` (0.8 HNSW),
  `milvus` (2.5), `weaviate`, `elasticsearch`.
- **Hosted clouds** (key from env, billed per query; `publish_restricted`, D-12):
  `pinecone`, `zilliz-cloud`, `weaviate-cloud`.

## Datasets (`eval_vectordb.datasets`)

- **ANN-Benchmarks HDF5** (exact precomputed neighbors): `sift-128-euclidean`,
  `gist-960-euclidean`, `glove-{25,100,200}-angular`, `nytimes-256-angular`,
  `fashion-mnist-784-euclidean`, `deep-image-96-angular`, …
- **Modern RAG text embeddings** (best/latest models): `cohere-wiki-v3-1024`
  (Cohere `embed-multilingual-v3.0`), `cohere-wiki-768` (`-v2.0`), `openai3-1536`
  (OpenAI `text-embedding-3`).
- **`synthetic` / `synthetic-angular`**: deterministic, offline, for tests/smoke.

Corpora are subsampled to a **medium** scale (default 100k base / 1k queries / k=10); when
the base is subsampled the precomputed neighbors are invalidated, so exact neighbors are
**recomputed by brute force** (faiss-accelerated) and cached. Set `--base-limit 0`-style
flags or pass the full ANN-Benchmarks sizes for a publication-scale run.

## Run it

```bash
# offline smoke (synthetic, OSS in-process engines, no keys/downloads):
uv run python -m eval_vectordb --datasets synthetic,synthetic-angular --run-id vectordb-smoke

# medium real run (engines auto-skip if their dep/service/key is absent):
uv pip install -e "packages/eval-vectordb[datasets]" "packages/bench-adapters[vectordb-local,vectordb-faiss]"
uv run python -m eval_vectordb \
    --datasets sift-128-euclidean,glove-100-angular,gist-960-euclidean,cohere-wiki-v3-1024,openai3-1536 \
    --engines faiss-hnsw,faiss-ivf,hnswlib,annoy,qdrant-local,lancedb \
    --run-id vectordb-dev
uv run bench results emit        # -> data/results/public-snapshot/vectordb.json + apps/mcp/data/vectordb.json
```

The run writes `runs/<run_id>/{items,slices}.jsonl + manifest.json` and the curated
`snapshots/vectordb.counts.toml` the leaderboard pipeline consumes.

## Status of the published snapshot

The committed `snapshots/vectordb.counts.toml` is a **provisional** snapshot generated
from a real offline `synthetic` + `synthetic-angular` run (honest, reproducible — no
fabricated numbers) so the end-to-end pipeline (run → emit → MCP) is live. The full
multi-engine run over the real ANN-Benchmarks + modern-embedding corpora (and the
Docker/hosted engines) is the documented next step; re-running `bench results emit`
refreshes the public JSON.

## Provenance

Methodology and dataset choices follow ANN-Benchmarks (Aumüller, Bernhardsson, Faithfull;
MIT) and VectorDBBench (Zilliz; Apache-2.0). Datasets are downloaded from their authors,
never vendored. Code Apache-2.0; example data CC-BY-4.0.
