"""Concrete web-SEARCH vendor adapters (query -> ranked URLs).

Ported from arlenk2021/GoldenEvalsWebSearch
(`src/probe/vendors/adapters.py` + `base.py`). One small class per vendor, all
behind the bench-adapters `Adapter.invoke(item) -> dict` interface. Endpoint
shapes follow each vendor's documented response as of 2026; adjust field paths
here if a vendor revises its schema.

In the source these were async `search(query, k) -> [url]` methods that pulled
keys from a pydantic Settings object. Here we keep the exact request/response
parsing logic but: (1) read keys straight from the environment (never hardcode),
(2) expose a synchronous `invoke(item)` that returns the bench-adapters result
dict, and (3) measure latency. Adapters raise VendorUnavailable when their key is
unset so the harness can skip them cleanly (and record the skip) rather than
charging a vendor a miss it never had a chance at.
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

from bench_adapters.registry import Adapter, register

# Per-request timeout (seconds). The source centralised this in Settings
# (http_timeout_seconds, default 30s); we inline a sensible default here.
SEARCH_TIMEOUT = 30.0

# Codes the source's get_with_retry treated as transient. Kept for parity.
_TRANSIENT = (429, 500, 502, 503, 504)


class VendorUnavailable(Exception):
    """Raised when a vendor cannot be queried (missing key, etc.)."""


def _need(value: str | None, name: str) -> str:
    if not value:
        raise VendorUnavailable(f"{name} key unset")
    return value


def _env(*names: str) -> str:
    """First non-empty value among the given env var names (vendor key lookup)."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return ""


class _SearchAdapter(Adapter):
    """Base for search adapters: turns a query into a ranked list of URLs.

    `invoke(item)` reads `item['query']` (the search query) and an optional
    `item['k']` (number of results, default 10), runs the vendor `search()`, and
    returns the bench-adapters result dict. Cost is left at 0.0 — vendor billing
    is per-plan and not exposed on the response, so the harness fills it in from
    pricing config if needed.
    """

    name: str = ""

    def search(self, query: str, k: int) -> list[str]:
        """Return up to k result URLs in rank order."""
        raise NotImplementedError

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        query = item.get("query") or item.get("q") or ""
        k = int(item.get("k", item.get("count", 10)))
        t0 = time.monotonic()
        urls = self.search(query, k)
        latency_ms = (time.monotonic() - t0) * 1000.0
        return {
            "raw_output": "\n".join(urls),
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
            "returned_urls": urls,
        }


@register("exa")
class Exa(_SearchAdapter):
    name = "exa"

    def search(self, query: str, k: int) -> list[str]:
        key = _need(_env("EXA_API_KEY"), self.name)
        with httpx.Client(timeout=SEARCH_TIMEOUT, follow_redirects=True,
                          headers={"x-api-key": key}) as c:
            r = c.post(
                "https://api.exa.ai/search",
                json={"query": query, "numResults": k, "type": "auto"},
            )
            r.raise_for_status()
            return [h["url"] for h in r.json().get("results", []) if h.get("url")][:k]


@register("brave")
class Brave(_SearchAdapter):
    name = "brave"

    def search(self, query: str, k: int) -> list[str]:
        key = _need(_env("BRAVE_SEARCH_API_KEY"), self.name)
        with httpx.Client(timeout=SEARCH_TIMEOUT, follow_redirects=True,
                          headers={"X-Subscription-Token": key,
                                   "Accept": "application/json"}) as c:
            r = c.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": k},
            )
            r.raise_for_status()
            results = (r.json().get("web", {}) or {}).get("results", [])
            return [x["url"] for x in results if x.get("url")][:k]


@register("tavily")
class Tavily(_SearchAdapter):
    name = "tavily"

    def search(self, query: str, k: int) -> list[str]:
        key = _need(_env("TAVILY_API_KEY"), self.name)
        with httpx.Client(timeout=SEARCH_TIMEOUT, follow_redirects=True) as c:
            r = c.post(
                "https://api.tavily.com/search",
                json={"api_key": key, "query": query, "max_results": k},
            )
            r.raise_for_status()
            return [x["url"] for x in r.json().get("results", []) if x.get("url")][:k]


@register("google_cse")
class GoogleCSE(_SearchAdapter):
    name = "google_cse"

    def search(self, query: str, k: int) -> list[str]:
        key = _need(_env("GOOGLE_CSE_KEY"), self.name)
        cx = _need(_env("GOOGLE_CSE_ENGINE_ID"), self.name)
        with httpx.Client(timeout=SEARCH_TIMEOUT, follow_redirects=True) as c:
            r = c.get(
                "https://www.googleapis.com/customsearch/v1",
                params={"key": key, "cx": cx, "q": query, "num": min(k, 10)},
            )
            r.raise_for_status()
            return [x["link"] for x in r.json().get("items", []) if x.get("link")][:k]


@register("bing")
class Bing(_SearchAdapter):
    name = "bing"

    def search(self, query: str, k: int) -> list[str]:
        key = _need(_env("BING_SEARCH_KEY"), self.name)
        with httpx.Client(timeout=SEARCH_TIMEOUT, follow_redirects=True,
                          headers={"Ocp-Apim-Subscription-Key": key}) as c:
            r = c.get(
                "https://api.bing.microsoft.com/v7.0/search",
                params={"q": query, "count": k},
            )
            r.raise_for_status()
            vals = (r.json().get("webPages", {}) or {}).get("value", [])
            return [x["url"] for x in vals if x.get("url")][:k]


@register("serpapi")
class SerpAPI(_SearchAdapter):
    name = "serpapi"

    def search(self, query: str, k: int) -> list[str]:
        key = _need(_env("SERPAPI_KEY"), self.name)
        with httpx.Client(timeout=SEARCH_TIMEOUT, follow_redirects=True) as c:
            r = c.get(
                "https://serpapi.com/search",
                params={"engine": "google", "q": query, "num": k, "api_key": key},
            )
            r.raise_for_status()
            return [x["link"] for x in r.json().get("organic_results", []) if x.get("link")][:k]


@register("perplexity")
class Perplexity(_SearchAdapter):
    name = "perplexity"

    def search(self, query: str, k: int) -> list[str]:
        key = _need(_env("PERPLEXITY_API_KEY"), self.name)
        with httpx.Client(timeout=SEARCH_TIMEOUT, follow_redirects=True,
                          headers={"Authorization": f"Bearer {key}"}) as c:
            r = c.post(
                "https://api.perplexity.ai/chat/completions",
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": query}],
                    "return_citations": True,
                },
            )
            r.raise_for_status()
            data = r.json()
            cites = data.get("citations") or data.get("search_results") or []
            urls = [c if isinstance(c, str) else c.get("url") for c in cites]
            return [u for u in urls if u][:k]


@register("you")
class You(_SearchAdapter):
    name = "you"

    def search(self, query: str, k: int) -> list[str]:
        key = _need(_env("YOU_API_KEY"), self.name)
        with httpx.Client(timeout=SEARCH_TIMEOUT, follow_redirects=True,
                          headers={"X-API-Key": key}) as c:
            r = c.get("https://api.ydc-index.io/search", params={"query": query})
            r.raise_for_status()
            hits = r.json().get("hits", [])
            return [h["url"] for h in hits if h.get("url")][:k]


ALL = [Exa, Brave, Tavily, GoogleCSE, Bing, SerpAPI, Perplexity, You]
