# bench-adapters

Provider/primitive adapter SDK for Primitive Bench (lm-eval registry pattern).
Adapters wrap a system-under-test behind a uniform `invoke(item) -> dict` and are
registered by name so configs/CLI can reference them as strings.

```python
import bench_adapters  # importing auto-registers all adapters below
from bench_adapters import get, registry

cls = get("brave")
adapter = cls(spec)            # spec: bench_schemas.AdapterSpec
out = adapter.invoke({"query": "who is the CEO of Acme", "k": 10})
# search:     out -> {raw_output, latency_ms, cost_usd, returned_urls: [...]}
# extraction: out -> {raw_output, latency_ms, cost_usd, main_text: "..."}
```

API keys are read from the environment — never hardcoded. Each adapter raises
`VendorUnavailable` if its key is unset, so the harness can skip it cleanly
rather than scoring a miss it never had a chance at. Each key is read from its
bare `<VENDOR>_API_KEY` environment variable (e.g. `EXA_API_KEY`).

## Provenance

Ported from [arlenk2021/GoldenEvalsWebSearch](https://github.com/arlenk2021/GoldenEvalsWebSearch)
(`src/probe/vendors/{base,adapters}.py` for search; `src/probe/extract/adapters.py`
for extraction). Originally Apache-2.0 code / CC-BY-4.0 data; redistributed under
Apache-2.0 within primitive-bench by the author. Vendor request/response parsing logic and
comments are preserved; the async `search(query, k)` / `extract(url)` methods are
adapted to the synchronous bench-adapters `Adapter.invoke(item)` contract.

## Registered SEARCH adapters (query -> ranked URLs)

`invoke(item)` reads `item["query"]` (or `q`) and optional `item["k"]`
(or `count`, default 10). Returns `returned_urls: list[str]`.

| name | vendor | env var(s) required |
|------|--------|-----------------------------------------|
| `exa` | Exa | `EXA_API_KEY` |
| `brave` | Brave Search | `BRAVE_SEARCH_API_KEY` |
| `tavily` | Tavily | `TAVILY_API_KEY` |
| `google_cse` | Google Custom Search | `GOOGLE_CSE_KEY` **and** `GOOGLE_CSE_ENGINE_ID` |
| `bing` | Bing Web Search | `BING_SEARCH_KEY` |
| `serpapi` | SerpAPI | `SERPAPI_KEY` |
| `perplexity` | Perplexity | `PERPLEXITY_API_KEY` |
| `you` | You.com | `YOU_API_KEY` |

## Registered EXTRACTION adapters (URL -> clean content)

`invoke(item)` reads `item["url"]`. Returns `main_text: str`.

| name | vendor | env var(s) required |
|------|--------|-----------------------------------------|
| `firecrawl` | Firecrawl | `FIRECRAWL_API_KEY` |
| `jina` | Jina Reader | `JINA_API_KEY` |
| `exa_live` | Exa /contents (livecrawl=always) | `EXA_API_KEY` |
| `exa_cached` | Exa /contents (livecrawl=never) | `EXA_API_KEY` |
| `tavily_extract` | Tavily /extract | `TAVILY_API_KEY` |
| `apify` | Apify content crawler | `APIFY_API_KEY` (optional actor: `APIFY_ACTOR`, default `apify/website-content-crawler`) |

`BrightData` is ported but **not registered** (Web Unlocker zone is
account-specific and the endpoint 400s without it); kept in the module for
future use.

## Registered RERANK adapters (query + candidate list -> reordered ids)

`invoke(item)` reads `item["query"]` and `item["candidates"]` (a list of
`{"id", "text"}`). Returns `reordered_ids: list[str]` (candidate ids, best first).
A reranker is a pure function over a fixed candidate list — no corpus or index.

The two local cross-encoders are keyless (free, run on CPU) and need the
`local-rerank` extra (`sentence-transformers`); install with
`uv sync --extra local-rerank` or via `eval-reranker`, which depends on it. The
hosted APIs read a bare `<VENDOR>_API_KEY`. `cost_usd` records the call's list
price even inside a vendor's free tier.

| name | vendor | model | env var / requirement |
|------|--------|-------|-----------------------|
| `ce-minilm` | local (sentinel) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | none — `local-rerank` extra |
| `bge-reranker` | local | `BAAI/bge-reranker-v2-m3` | none — `local-rerank` extra |
| `voyage-rerank` | Voyage | `rerank-2` | `VOYAGE_API_KEY` (first 200M tok free) |
| `jina-rerank` | Jina | `jina-reranker-v2-base-multilingual` | `JINA_API_KEY` (first 10M tok free) |
| `cohere-rerank` | Cohere | `rerank-v3.5` | `COHERE_API_KEY` (pay-as-you-go) |

`bge-reranker` is **opt-in** — its weights are ~2.3 GB and slow on CPU, so it is not in
`eval-reranker`'s default vendor set (nor the published board); pass it explicitly to include it.

## Registered VECTORDB engines (build an index, then query it)

A vector engine is **stateful** — it does not fit the stateless `invoke(item)` shape.
Instead `from bench_adapters.vectordb import VectorEngine, VendorUnavailable` defines a
two-phase lifecycle the `eval_vectordb.runner` drives:

```python
from bench_adapters import get
eng = get("faiss-hnsw")(spec)          # spec.params carries M / ef_search / nprobe ...
eng.build(base_vectors, "angular", {"M": 16, "ef_search": 200})  # index once
ids = eng.query(query_vector, k=10)     # -> top-k base indices (best first); many times
eng.free()                              # release index memory between configs
```

Metrics use the ann-benchmarks vocabulary: `euclidean` (L2) and `angular` (cosine; the
vectors are L2-normalized so an L2/IP index ranks as cosine). Every heavy import is lazy,
so `import bench_adapters` never needs faiss/hnswlib/etc.; a missing dep, key, or service
raises `VendorUnavailable` and the runner skips the lane uncharged. Self-hosted engines
cost `$0`; hosted clouds record per-query list price (see `vectordb/pricing.py`).

Install extras per tier: `bench-adapters[vectordb-local]` (numpy/hnswlib/annoy/qdrant/lancedb),
`bench-adapters[vectordb-faiss]` (faiss-cpu), `bench-adapters[vectordb-servers]`,
`bench-adapters[vectordb-hosted]`.

| name | vendor | tier | requirement |
|------|--------|------|-------------|
| `bruteforce-numpy` | numpy (sentinel) | OSS in-process | none — exact recall=1.0 oracle |
| `faiss-flat` | FAISS | OSS in-process | `vectordb-faiss` — exact reference |
| `faiss-hnsw` | FAISS | OSS in-process | `vectordb-faiss` — graph, `M`/`ef_search` |
| `faiss-ivf` | FAISS | OSS in-process | `vectordb-faiss` — IVF, `nlist`/`nprobe` |
| `hnswlib` | hnswlib | OSS in-process | `vectordb-local` |
| `annoy` | Spotify | OSS in-process | `vectordb-local` — `n_trees`/`search_k` |
| `qdrant-local` | Qdrant | OSS in-process | `vectordb-local` — in-memory `:memory:` |
| `lancedb` | LanceDB | OSS in-process | `vectordb-local` — embedded IVF-PQ |
| `pgvector` | PostgreSQL | Docker server | `vectordb-servers`; `PGVECTOR_DSN` / `DATABASE_URL` |
| `milvus` | Milvus | Docker server | `vectordb-servers`; `MILVUS_URI` |
| `weaviate` | Weaviate | Docker server | `vectordb-servers`; `WEAVIATE_HOST`/`WEAVIATE_PORT` |
| `elasticsearch` | Elasticsearch | Docker server | `vectordb-servers`; `ES_URL` |
| `pinecone` | Pinecone | hosted (restricted) | `vectordb-hosted`; `PINECONE_API_KEY` |
| `zilliz-cloud` | Zilliz Cloud | hosted (restricted) | `vectordb-hosted`; `ZILLIZ_URI`/`ZILLIZ_TOKEN` |
| `weaviate-cloud` | Weaviate Cloud | hosted (restricted) | `vectordb-hosted`; `WEAVIATE_URL`/`WEAVIATE_API_KEY` |

Hosted engines are marked `publish_restricted=True` (DeWitt clause, DECISIONS D-12) as a
conservative default until each EULA's publishing terms are cleared; the leaderboard hides
restricted adapters. Bring up the Docker servers with `docker/docker-compose.vectordb.yml`.
