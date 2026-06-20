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
