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

## Registered CRAWL adapters (seed URL -> fetched pages)

`invoke(item)` reads `item["seed_url"]` (+ crawl budget `max_pages`/`max_depth`, and
an optional inline `site` link graph for the offline controlled split). Returns
`pages: [{"url", "content"}]` and `returned_urls: [str]`. A crawler must *discover*
the pages under the seed — the capability extraction (handed the exact URL) never tests.

The four keyless `_LocalCrawler` strategies are a real breadth-first crawler in four
policy configs (depth / sitemap / JS) — they run with **no key** (offline-graph,
deterministic, $0) and produce the public snapshot; `bfs-deep` doubles as the keyless
live baseline. The four hosted APIs are the live-run systems-under-test. `cost_usd`
records the per-page list price (see `crawl/pricing.py`).

| name | vendor | mode | env var / requirement |
|------|--------|------|-----------------------|
| `bfs-shallow` | local (sentinel) | BFS depth 1, static only | none |
| `bfs-deep` | local | BFS depth 4, static only | none |
| `sitemap-crawl` | local | BFS depth 4 + `sitemap.xml` | none |
| `render-crawl` | local | BFS depth 4 + sitemap + JS links | none |
| `firecrawl-crawl` | Firecrawl | `/v2/crawl` (async job) | `FIRECRAWL_API_KEY` |
| `tavily-crawl` | Tavily | `/crawl` (invite-only beta) | `TAVILY_API_KEY` |
| `spider-crawl` | Spider Cloud | `/crawl` (smart HTTP/headless) | `SPIDER_API_KEY` |
| `apify-crawl` | Apify | website-content-crawler | `APIFY_API_KEY` (optional actor: `APIFY_CRAWL_ACTOR`) |

`follow_js` is a no-op on a live crawl (a pure-httpx crawler cannot execute JS — that
gap is exactly what the hosted headless vendors are measured against); it separates the
strategies in the offline controlled graph where JS links are explicit.
