"""Concrete web-EXTRACTION vendor adapters (URL -> clean content).

Ported from arlenk2021/GoldenEvalsWebSearch (`src/probe/extract/adapters.py`).
Each adapter takes a URL and returns the extracted page text (markdown or plain).
The web_extraction benchmark then asks: does the row's truth token survive the
vendor's extraction? That is the natural extension of the golden-row truth-token
contract to the extraction primitive.

In the source these were async `extract(url) -> str` methods that pulled keys
from a pydantic Settings object. Here we keep the exact request/response parsing
logic but: (1) read keys straight from the environment (never hardcode), (2)
expose a synchronous `invoke(item)` returning the bench-adapters result dict, and
(3) measure latency. Adapters raise VendorUnavailable when their key is unset so
the harness skips them cleanly rather than scoring a miss they never had a chance
at.
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

from bench_adapters.registry import Adapter, register

# Extraction (esp. Apify's crawler) can be slow; give these calls more room.
EXTRACT_TIMEOUT = 120.0


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


class _ExtractAdapter(Adapter):
    """Base for extraction adapters: turns a URL into clean page text.

    `invoke(item)` reads `item['url']`, runs the vendor `extract()`, and returns
    the bench-adapters result dict. Cost is left at 0.0 — vendor billing is
    per-plan and not exposed on the response, so the harness fills it in from
    pricing config if needed.
    """

    name: str = ""

    def extract(self, url: str) -> str:
        raise NotImplementedError

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        url = item.get("url") or ""
        t0 = time.monotonic()
        text = self.extract(url)
        latency_ms = (time.monotonic() - t0) * 1000.0
        return {
            "raw_output": text,
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
            "main_text": text,
        }


@register("firecrawl")
class Firecrawl(_ExtractAdapter):
    name = "firecrawl"

    def extract(self, url: str) -> str:
        key = _need(_env("FIRECRAWL_API_KEY"), self.name)
        with httpx.Client(timeout=EXTRACT_TIMEOUT, follow_redirects=True,
                          headers={"Authorization": f"Bearer {key}"}) as c:
            r = c.post(
                "https://api.firecrawl.dev/v1/scrape",
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            )
            r.raise_for_status()
            data = r.json().get("data", {}) or {}
            return data.get("markdown") or data.get("content") or ""


@register("jina")
class Jina(_ExtractAdapter):
    name = "jina"

    def extract(self, url: str) -> str:
        key = _need(_env("JINA_API_KEY"), self.name)
        # Jina Reader proxies the URL and returns cleaned markdown in the body.
        with httpx.Client(timeout=EXTRACT_TIMEOUT, follow_redirects=True,
                          headers={"Authorization": f"Bearer {key}"}) as c:
            r = c.get(f"https://r.jina.ai/{url}")
            r.raise_for_status()
            return r.text


class _ExaContents(_ExtractAdapter):
    """Exa /contents. `livecrawl` decides cached-index vs forced live fetch.

    Running both modes disentangles "Exa's index lags fresh docs" (cached fails,
    live succeeds) from "Exa is broken / param confound" (both fail).
    """

    livecrawl = "always"

    def extract(self, url: str) -> str:
        key = _need(_env("EXA_API_KEY"), self.name)
        with httpx.Client(timeout=EXTRACT_TIMEOUT, follow_redirects=True,
                          headers={"x-api-key": key}) as c:
            r = c.post(
                "https://api.exa.ai/contents",
                json={"urls": [url], "text": True, "livecrawl": self.livecrawl},
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            return (results[0].get("text") or "") if results else ""


@register("exa_live")
class ExaLive(_ExaContents):
    name = "exa_live"
    livecrawl = "always"        # force a live fetch


@register("exa_cached")
class ExaCached(_ExaContents):
    name = "exa_cached"
    livecrawl = "never"         # pure index, no live fetch


@register("tavily_extract")
class TavilyExtract(_ExtractAdapter):
    # Source registered this under the bare name "tavily"; in bench-adapters the
    # search and extraction registries are unified, so it is registered as
    # "tavily_extract" to avoid colliding with the Tavily search adapter.
    name = "tavily_extract"

    def extract(self, url: str) -> str:
        key = _need(_env("TAVILY_API_KEY"), self.name)
        with httpx.Client(timeout=EXTRACT_TIMEOUT, follow_redirects=True) as c:
            r = c.post(
                "https://api.tavily.com/extract",
                json={"api_key": key, "urls": [url]},
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            return (results[0].get("raw_content") or "") if results else ""


@register("apify")
class Apify(_ExtractAdapter):
    name = "apify"

    def extract(self, url: str) -> str:
        key = _need(_env("APIFY_API_KEY"), self.name)
        actor = _env("APIFY_ACTOR") or "apify/website-content-crawler"
        actor_path = actor.replace("/", "~")
        endpoint = (
            f"https://api.apify.com/v2/acts/{actor_path}/run-sync-get-dataset-items"
            f"?token={key}"
        )
        with httpx.Client(timeout=EXTRACT_TIMEOUT, follow_redirects=True) as c:
            r = c.post(
                endpoint,
                json={"startUrls": [{"url": url}], "maxCrawlPages": 1, "maxCrawlDepth": 0},
            )
            r.raise_for_status()
            items = r.json()
            if not items:
                return ""
            item = items[0]
            return item.get("text") or item.get("markdown") or ""


class BrightData(_ExtractAdapter):
    name = "brightdata"

    def extract(self, url: str) -> str:
        key = _need(_env("BRIGHTDATA_API_KEY"), self.name)
        # Web Unlocker request API. Zone name is account-specific; default "web_unlocker".
        with httpx.Client(timeout=EXTRACT_TIMEOUT, follow_redirects=True,
                          headers={"Authorization": f"Bearer {key}"}) as c:
            r = c.post(
                "https://api.brightdata.com/request",
                json={"zone": "web_unlocker", "url": url, "format": "raw"},
            )
            r.raise_for_status()
            return r.text


# BrightData excluded: Web Unlocker zone is account-specific and the endpoint
# 400s without it. The adapter class is kept above for future use but is not
# registered, so it will not be attempted.
ALL = [Firecrawl, Jina, ExaLive, ExaCached, TavilyExtract, Apify]
