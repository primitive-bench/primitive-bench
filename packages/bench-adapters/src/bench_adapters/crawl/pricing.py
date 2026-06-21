"""Pinned price table for crawl cost capture (USD).

Hosted crawlers bill per *page fetched* (Firecrawl credits, Spider requests,
Apify compute units) — unlike rerankers there is no token meter, the unit is a
page. `cost_usd` on each crawl result records the **list price** of the call (what
it would cost outside a free tier), so the leaderboard's cost dimension stays
honest even when a run lands inside a vendor's free allowance.

Prices are the per-page list price off each vendor's public pricing page
(2026-06), reduced to a single $/page figure for the cost dimension. They are
deliberately approximate — crawling is billed in plan-specific credit bundles —
and MUST be refreshed deliberately, never silently. The benchmark's published gate
is *coverage*, not cost; cost rides alongside as a secondary dimension.
"""
from __future__ import annotations

PRICING_VERSION = "2026-06"

# USD per page fetched (list price, approximate — see module docstring).
PAGE_PRICES: dict[str, float] = {
    "firecrawl": 0.00083,  # ~1 credit/page; Standard plan ~$83 / 100k credits
    "tavily": 0.008,       # crawl ~1 API credit/page; pay-as-you-go ~$0.008/credit
    "spider": 0.0003,      # smart-mode request, approx blended HTTP/headless
    "apify": 0.001,        # website-content-crawler, approx compute + platform usage
}


def page_cost(vendor: str, n_pages: int) -> float:
    """List-price USD for fetching `n_pages` with `vendor` (0.0 if unknown)."""
    return max(0, int(n_pages)) * PAGE_PRICES.get(vendor, 0.0)
