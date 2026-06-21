"""Concrete CRAWL adapters: (seed URL) -> a set of fetched pages.

A crawler is handed a single **seed URL** and must DISCOVER and FETCH the pages
reachable under it — the capability extraction does not test (extraction is handed
the exact URL). `invoke(item)` reads `item['seed_url']` (and crawl budget
`max_pages` / `max_depth`), runs the vendor crawl, and returns the bench-adapters
result dict with:

  * ``pages``         — ``[{"url": <fetched url>, "content": <markdown/text>}]``
  * ``returned_urls`` — just the URLs (the pure URL-discovery surface)

The crawl scorer then asks, per golden TARGET page: did the crawler reach it
(coverage) AND return its current content (freshness)? See eval_crawl.scoring.

Two execution modes share one base:

  * **Offline graph** — when ``item['site']`` carries an inline link graph
    (``{pages: {url: {content, links, js_links}}, sitemap: [...]}``), the keyless
    `_LocalCrawler` crawls it deterministically with NO network. This is how the
    controlled public example set and the committed snapshot are produced — exact
    coverage ground truth, reproducible by anyone, CI-safe.
  * **Live HTTP** — with no inline graph, `_LocalCrawler` does a real polite
    breadth-first crawl over the live web (httpx + selectolax), the keyless
    baseline / regression sentinel for the live run.

Three hosted crawl APIs (Firecrawl, Spider, Apify) plus Tavily (invite-only beta)
are the systems-under-test for the live run. As in the search/extract/rerank
adapters, keys are read straight from the environment and `VendorUnavailable` is
raised when a vendor cannot run (missing key/dep, beta access) so the harness skips
that lane cleanly instead of charging it a miss it never had a chance at.

`cost_usd` records the call's **list price** (see pricing.py) — $/page — even
inside a free tier, so the leaderboard's cost dimension stays honest.
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from collections import deque
from typing import Any, Iterable

import httpx
from selectolax.parser import HTMLParser

from bench_adapters.crawl import pricing
from bench_adapters.registry import Adapter, register

# A crawl is heavier than a single scrape; give the hosted jobs room.
CRAWL_TIMEOUT = 120.0
# Default crawl budget when the item does not pin one (kept modest: the golden
# targets sit within a few hops of the seed, and a benchmark should not hammer a
# site). Hosted vendors receive the same budget for a fair comparison.
DEFAULT_MAX_PAGES = 60
DEFAULT_MAX_DEPTH = 4

_RETRY_STATUS = {429, 500, 502, 503, 504}
CRAWL_MAX_ATTEMPTS = 6
_MAX_BACKOFF = 30.0

# Browser-like UA: many sites serve an anti-bot challenge to non-browser agents,
# which would read as false "not reached" coverage misses for the live baseline.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_BOILERPLATE = ("script", "style", "nav", "header", "footer", "aside", "noscript")


class VendorUnavailable(Exception):
    """Raised when a crawl adapter cannot run (missing key / dep / beta access)."""


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


# --------------------------------------------------------------------------- #
# URL normalization for the in-crawler `seen` set.
#
# We reuse bench_core.urls.normalize_url so the crawler's de-dup matches the
# scorer's equivalence-class membership exactly (a page fetched under a tracking
# param is the same page the scorer is looking for). bench-core is always present
# (it is a hard dependency of every eval package that drives these adapters).
# --------------------------------------------------------------------------- #
from bench_core.urls import normalize_url, registrable_domain  # noqa: E402


def _norm(url: str) -> str:
    try:
        return normalize_url(url)
    except Exception:
        return url.strip()


class _CrawlAdapter(Adapter):
    """Base: turn a seed URL into a list of fetched ``{url, content}`` pages.

    Subclasses implement ``crawl(seed_url, *, max_pages, max_depth, site)
    -> (pages, cost_usd)``. ``invoke(item)`` wraps that with latency measurement
    and the bench-adapters result-dict shape.
    """

    name: str = ""
    vendor: str = ""
    model_version: str = "unknown"
    is_sentinel: bool = False

    def crawl(self, seed_url: str, *, max_pages: int, max_depth: int,
              site: dict[str, Any] | None) -> tuple[list[dict[str, str]], float]:
        raise NotImplementedError

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        seed_url = str(item.get("seed_url") or "")
        site = item.get("site")
        max_pages = int(item.get("max_pages") or DEFAULT_MAX_PAGES)
        max_depth = int(item.get("max_depth") or DEFAULT_MAX_DEPTH)
        t0 = time.monotonic()
        pages, cost = self.crawl(seed_url, max_pages=max_pages, max_depth=max_depth, site=site)
        latency_ms = (time.monotonic() - t0) * 1000.0
        # De-dup by normalized URL, keep the first (shallowest) content for each.
        seen: set[str] = set()
        deduped: list[dict[str, str]] = []
        for p in pages:
            key = _norm(str(p.get("url", "")))
            if key and key not in seen:
                seen.add(key)
                deduped.append({"url": str(p.get("url", "")), "content": str(p.get("content", ""))})
        return {
            "pages": deduped,
            "returned_urls": [p["url"] for p in deduped],
            "latency_ms": latency_ms,
            "cost_usd": cost,
            "raw_output": f"{len(deduped)} pages crawled from {seed_url}",
        }


# --------------------------------------------------------------------------- #
# Keyless local crawler: offline-graph (deterministic) OR live BFS. No key, $0.
# --------------------------------------------------------------------------- #
class _LocalCrawler(_CrawlAdapter):
    """Real breadth-first crawler, registered in four strategy variants.

    The variants differ only in three crawl-policy knobs, and those knobs are
    exactly the axes the leaderboard slices on, so the variants genuinely separate:

      * ``depth_cfg``    — how many hops from the seed to follow (depth slice).
      * ``use_sitemap``  — seed the frontier from ``sitemap.xml`` (the site's own
        authoritative page registry), so sitemap-listed pages are reached directly
        regardless of link depth.
      * ``follow_js``    — also follow links that only appear after JS execution
        (``js_links`` in the offline graph; the headless-render capability).

    Offline (inline ``site`` graph) the crawl is fully deterministic with exact
    coverage ground truth. Live, it is a polite same-registrable-domain BFS over
    httpx; ``follow_js`` is a no-op live (a pure-httpx crawler cannot execute JS —
    that gap is precisely what the hosted headless vendors are measured against).
    """

    vendor = "local"
    depth_cfg = DEFAULT_MAX_DEPTH
    use_sitemap = False
    follow_js = False

    def crawl(self, seed_url, *, max_pages, max_depth, site):
        budget = min(self.depth_cfg, max_depth) if max_depth else self.depth_cfg
        if site is not None:
            return self._crawl_graph(seed_url, site, max_pages, budget), 0.0
        return self._crawl_live(seed_url, max_pages, budget), 0.0

    # ---- offline: deterministic BFS over an inline link graph ---------------- #
    def _crawl_graph(self, seed_url, site, max_pages, budget):
        pages_map: dict[str, dict[str, Any]] = site.get("pages", {}) or {}
        lookup = {_norm(u): (u, p) for u, p in pages_map.items()}

        frontier: deque[tuple[str, int]] = deque()
        seen: set[str] = set()

        def _push(url: str, depth: int) -> None:
            key = _norm(url)
            if key and key not in seen:
                seen.add(key)
                frontier.append((url, depth))

        _push(seed_url, 0)
        # A sitemap-aware crawler fetches every sitemap-listed URL directly (depth 0).
        if self.use_sitemap:
            for u in site.get("sitemap", []) or []:
                _push(u, 0)

        out: list[dict[str, str]] = []
        while frontier and len(out) < max_pages:
            url, depth = frontier.popleft()
            hit = lookup.get(_norm(url))
            if hit is None:
                continue  # a dangling link / 404 — fetched nothing
            real_url, page = hit
            out.append({"url": real_url, "content": str(page.get("content", ""))})
            if depth >= budget:
                continue
            links = list(page.get("links", []) or [])
            if self.follow_js:
                links += list(page.get("js_links", []) or [])
            for link in links:
                _push(link, depth + 1)
        return out

    # ---- live: polite same-domain BFS over httpx ----------------------------- #
    def _crawl_live(self, seed_url, max_pages, budget):
        if not seed_url:
            return []
        try:
            return asyncio.run(self._crawl_live_async(seed_url, max_pages, budget))
        except RuntimeError:
            # Already inside an event loop (rare for the sync runner) — fall back
            # to a fresh loop in a thread-free manner.
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._crawl_live_async(seed_url, max_pages, budget))
            finally:
                loop.close()

    async def _crawl_live_async(self, seed_url, max_pages, budget):
        domain = registrable_domain(seed_url)
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        frontier: deque[tuple[str, int]] = deque()

        def _push(url: str, depth: int) -> None:
            key = _norm(url)
            if key and key not in seen and url.startswith(("http://", "https://")):
                seen.add(key)
                frontier.append((url, depth))

        _push(seed_url, 0)
        async with httpx.AsyncClient(
            timeout=CRAWL_TIMEOUT, follow_redirects=True, http2=True,
            headers={"User-Agent": BROWSER_UA},
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        ) as client:
            if self.use_sitemap:
                for u in await self._sitemap_urls(client, seed_url, domain):
                    _push(u, 0)
            while frontier and len(out) < max_pages:
                url, depth = frontier.popleft()
                html, text = await self._fetch(client, url)
                if text is None:
                    continue
                out.append({"url": url, "content": text})
                if depth >= budget or not html:
                    continue
                for link in self._extract_links(html, url):
                    if registrable_domain(link) == domain:
                        _push(link, depth + 1)
                await asyncio.sleep(0.2)  # politeness
        return out

    async def _fetch(self, client, url):
        for attempt in range(3):
            try:
                r = await client.get(url)
                if r.status_code in _RETRY_STATUS and attempt < 2:
                    await asyncio.sleep(1.5 ** (attempt + 1))
                    continue
                ctype = r.headers.get("content-type", "")
                if "html" not in ctype:
                    return None, r.text
                return r.text, _main_text(r.text)
            except (httpx.TransportError, httpx.TimeoutException):
                await asyncio.sleep(1.5 ** (attempt + 1))
        return None, None

    async def _sitemap_urls(self, client, seed_url, domain) -> list[str]:
        from urllib.parse import urljoin
        out: list[str] = []
        try:
            r = await client.get(urljoin(seed_url, "/sitemap.xml"))
            if r.status_code == 200:
                tree = HTMLParser(r.text)
                for loc in tree.css("loc"):
                    u = loc.text(strip=True)
                    if u and registrable_domain(u) == domain:
                        out.append(u)
        except (httpx.TransportError, httpx.TimeoutException):
            pass
        return out[:DEFAULT_MAX_PAGES]

    @staticmethod
    def _extract_links(html: str, base_url: str) -> list[str]:
        from urllib.parse import urldefrag, urljoin
        tree = HTMLParser(html)
        out: list[str] = []
        for a in tree.css("a[href]"):
            href = a.attributes.get("href")
            if not href or href.startswith(("mailto:", "javascript:", "#", "tel:")):
                continue
            out.append(urldefrag(urljoin(base_url, href))[0])
        return out


def _main_text(html: str) -> str:
    tree = HTMLParser(html)
    for tag in _BOILERPLATE:
        for node in tree.css(tag):
            node.decompose()
    body = tree.body or tree.root
    return body.text(separator=" ", strip=True) if body else ""


@register("bfs-shallow")
class BfsShallow(_LocalCrawler):
    """Naive breadth-first, 1 hop, no sitemap, no JS — the regression sentinel.

    Expected to be stable and to LOSE on the deep / js_rendered slices, not to win.
    """

    name = "bfs-shallow"
    model_version = "local-bfs/d1"
    is_sentinel = True
    depth_cfg = 1


@register("bfs-deep")
class BfsDeep(_LocalCrawler):
    """Breadth-first to 4 hops, static links only. The keyless live baseline."""

    name = "bfs-deep"
    model_version = "local-bfs/d4"
    depth_cfg = 4


@register("sitemap-crawl")
class SitemapCrawl(_LocalCrawler):
    """Sitemap-seeded BFS: read the site's own sitemap.xml, then crawl to 4 hops."""

    name = "sitemap-crawl"
    model_version = "local-bfs/d4+sitemap"
    depth_cfg = 4
    use_sitemap = True


@register("render-crawl")
class RenderCrawl(_LocalCrawler):
    """Sitemap-seeded BFS that also follows JS-rendered links (headless-equivalent).

    The only keyless variant that reaches js_rendered targets not listed in the
    sitemap — the offline stand-in for a headless-browser crawl.
    """

    name = "render-crawl"
    model_version = "local-bfs/d4+sitemap+js"
    depth_cfg = 4
    use_sitemap = True
    follow_js = True


# --------------------------------------------------------------------------- #
# Hosted crawl APIs. Key from env; cost = list price (see pricing.py).
# --------------------------------------------------------------------------- #
def _post_json_with_retry(client: httpx.Client, url: str, headers: dict[str, str],
                          payload: dict[str, Any]) -> dict[str, Any]:
    """POST with exponential backoff on 429/5xx, honoring Retry-After when present."""
    for attempt in range(CRAWL_MAX_ATTEMPTS):
        r = client.post(url, headers=headers, json=payload)
        if r.status_code in _RETRY_STATUS and attempt < CRAWL_MAX_ATTEMPTS - 1:
            ra = r.headers.get("retry-after")
            try:
                wait = float(ra) if ra else 0.0
            except ValueError:
                wait = 0.0
            if wait <= 0:
                wait = min(_MAX_BACKOFF, (2 ** attempt) + random.random())
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("unreachable: retry loop exhausted")  # pragma: no cover


@register("firecrawl-crawl")
class FirecrawlCrawl(_CrawlAdapter):
    """Firecrawl /v2/crawl — async job: submit, poll, collect markdown per page.

    Returns content keyed by ``metadata.sourceURL`` so the scorer can match each
    fetched page to a golden target. The async job model is the reason crawl gets
    a longer timeout than the single-page scrape used by the extraction adapter.
    """

    name = "firecrawl-crawl"
    vendor = "firecrawl"
    model_version = "firecrawl-v2"
    BASE = "https://api.firecrawl.dev/v2"
    POLL_INTERVAL = 3.0
    POLL_MAX = 90  # ~4.5 min ceiling

    def crawl(self, seed_url, *, max_pages, max_depth, site):
        key = _need(_env("FIRECRAWL_API_KEY"), self.name)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "url": seed_url,
            "limit": max_pages,
            "maxDiscoveryDepth": max_depth,
            "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
        }
        with httpx.Client(timeout=CRAWL_TIMEOUT, follow_redirects=True) as client:
            started = _post_json_with_retry(client, f"{self.BASE}/crawl", headers, payload)
            job_id = started.get("id") or started.get("jobId")
            if not job_id:
                return [], 0.0
            data, credits = self._poll(client, headers, job_id)
        pages = [
            {"url": (d.get("metadata") or {}).get("sourceURL") or (d.get("metadata") or {}).get("url") or "",
             "content": d.get("markdown") or d.get("content") or ""}
            for d in data
        ]
        pages = [p for p in pages if p["url"]]
        cost = (credits * pricing.PAGE_PRICES["firecrawl"]) if credits else pricing.page_cost("firecrawl", len(pages))
        return pages, cost

    def _poll(self, client, headers, job_id):
        data: list[dict[str, Any]] = []
        credits = 0
        url = f"{self.BASE}/crawl/{job_id}"
        for _ in range(self.POLL_MAX):
            r = client.get(url, headers=headers)
            r.raise_for_status()
            body = r.json()
            credits = int(body.get("creditsUsed") or credits)
            page = body.get("data") or []
            if page:
                data = page  # Firecrawl returns the cumulative set on each poll
            status = body.get("status")
            if status in ("completed", "failed", "cancelled"):
                break
            time.sleep(self.POLL_INTERVAL)
        return data, credits


@register("tavily-crawl")
class TavilyCrawl(_CrawlAdapter):
    """Tavily /crawl — agent-oriented site crawl returning ``raw_content`` per URL.

    Crawl is invite-only beta; a 401/403/404 (no access) raises VendorUnavailable
    so the lane is skipped cleanly rather than charged a miss.
    """

    name = "tavily-crawl"
    vendor = "tavily"
    model_version = "tavily-crawl"
    ENDPOINT = "https://api.tavily.com/crawl"

    def crawl(self, seed_url, *, max_pages, max_depth, site):
        key = _need(_env("TAVILY_API_KEY"), self.name)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"url": seed_url, "max_depth": max_depth, "limit": max_pages, "format": "markdown"}
        with httpx.Client(timeout=CRAWL_TIMEOUT, follow_redirects=True) as client:
            try:
                data = _post_json_with_retry(client, self.ENDPOINT, headers, payload)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (401, 403, 404):
                    raise VendorUnavailable(f"{self.name}: crawl not enabled ({exc.response.status_code})")
                raise
        results = data.get("results") or []
        pages = [
            {"url": r.get("url") or "", "content": r.get("raw_content") or r.get("content") or ""}
            for r in results
        ]
        pages = [p for p in pages if p["url"]]
        return pages, pricing.page_cost("tavily", len(pages))


@register("spider-crawl")
class SpiderCrawl(_CrawlAdapter):
    """Spider Cloud /crawl — fast crawl with smart HTTP/headless escalation."""

    name = "spider-crawl"
    vendor = "spider"
    model_version = "spider-cloud"
    ENDPOINT = "https://api.spider.cloud/crawl"

    def crawl(self, seed_url, *, max_pages, max_depth, site):
        key = _need(_env("SPIDER_API_KEY"), self.name)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "url": seed_url,
            "limit": max_pages,
            "depth": max_depth,
            "request": "smart",          # HTTP first, escalate to headless Chrome as needed
            "return_format": "markdown",
            "store_data": False,
        }
        with httpx.Client(timeout=CRAWL_TIMEOUT, follow_redirects=True) as client:
            data = _post_json_with_retry(client, self.ENDPOINT, headers, payload)
        rows: Iterable[dict[str, Any]] = data if isinstance(data, list) else (data.get("content") or [])
        pages = [
            {"url": r.get("url") or "", "content": r.get("content") or r.get("markdown") or ""}
            for r in rows if isinstance(r, dict)
        ]
        pages = [p for p in pages if p["url"]]
        return pages, pricing.page_cost("spider", len(pages))


@register("apify-crawl")
class ApifyCrawl(_CrawlAdapter):
    """Apify website-content-crawler — multi-page crawl returning text per URL.

    Same actor the extraction adapter used for single pages, here driven across a
    site (``maxCrawlPages`` / ``maxCrawlDepth`` > 1).
    """

    name = "apify-crawl"
    vendor = "apify"
    model_version = "apify/website-content-crawler"

    def crawl(self, seed_url, *, max_pages, max_depth, site):
        key = _need(_env("APIFY_API_KEY"), self.name)
        actor = _env("APIFY_CRAWL_ACTOR") or "apify/website-content-crawler"
        actor_path = actor.replace("/", "~")
        endpoint = (
            f"https://api.apify.com/v2/acts/{actor_path}/run-sync-get-dataset-items?token={key}"
        )
        payload = {
            "startUrls": [{"url": seed_url}],
            "maxCrawlPages": max_pages,
            "maxCrawlDepth": max_depth,
        }
        with httpx.Client(timeout=CRAWL_TIMEOUT, follow_redirects=True) as client:
            r = client.post(endpoint, json=payload)
            r.raise_for_status()
            items = r.json() or []
        pages = [
            {"url": it.get("url") or "", "content": it.get("text") or it.get("markdown") or ""}
            for it in items if isinstance(it, dict)
        ]
        pages = [p for p in pages if p["url"]]
        return pages, pricing.page_cost("apify", len(pages))


# Registered crawl adapters. The four keyless _LocalCrawler variants are the
# reproducible, offline systems-under-test for the public snapshot; the four
# hosted vendors are the live-run systems-under-test (skipped without a key).
LOCAL = [BfsShallow, BfsDeep, SitemapCrawl, RenderCrawl]
HOSTED = [FirecrawlCrawl, TavilyCrawl, SpiderCrawl, ApifyCrawl]
ALL = LOCAL + HOSTED
