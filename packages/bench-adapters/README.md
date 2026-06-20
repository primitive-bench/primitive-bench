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

## Registered OCR adapters (page image -> transcription)

`invoke(item)` reads `item["image"]` (an absolute path to a page image) and
returns `{raw_output, text, latency_ms, cost_usd, mode}`. **Prompted VLMs**
(`claude-sonnet-ocr`, `gpt-ocr`, `gemini-ocr`) share one pinned transcription
prompt; **native OCR** systems (`tesseract`, `mistral-ocr`, `deepseek-ocr`)
transcribe with no prompt. Model IDs are env-overridable so model drift never
needs a code change; the resolved ID is recorded in the run manifest.

| name | vendor | key / requirement (first match wins) | model env override (default) |
|------|--------|--------------------------------------|------------------------------|
| `tesseract` | Tesseract (local, sentinel) | system `tesseract` binary (`brew install tesseract`) | — |
| `claude-sonnet-ocr` | Anthropic (SDK) | `ANTHROPIC_API_KEY` | `CLAUDE_OCR_MODEL` (`claude-sonnet-4-6`) |
| `gpt-ocr` | OpenAI | `OPENAI_API_KEY` | `GPT_OCR_MODEL` (`gpt-4o`) |
| `gemini-ocr` | Google | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `GEMINI_OCR_MODEL` (`gemini-2.5-pro`) |
| `mistral-ocr` | Mistral (dedicated OCR API) | `MISTRAL_API_KEY` | `MISTRAL_OCR_MODEL` (`mistral-ocr-latest`) |
| `deepseek-ocr` | DeepSeek-OCR (self-host) | `DEEPSEEK_OCR_BASE_URL` (vLLM/Ollama OpenAI-compatible endpoint) | `DEEPSEEK_OCR_MODEL` (`deepseek-ocr`) |

DeepSeek-OCR has **no hosted vision API** (as of 2026-06) — it is weights-only, so
the adapter targets a self-hosted OpenAI-compatible endpoint and is skipped
(`VendorUnavailable`, uncharged) unless `DEEPSEEK_OCR_BASE_URL` is set.
