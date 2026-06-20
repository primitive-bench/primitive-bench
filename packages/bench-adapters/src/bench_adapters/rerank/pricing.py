"""Pinned price table for rerank cost capture (USD).

Hosted rerankers bill two ways: token-priced (Voyage, Jina — per input token over
query+documents) and search-priced (Cohere — per "search", one query against up to
100 documents). `cost_usd` on each result records the **list price** of the call
(what it would cost outside a free tier), so the leaderboard's cost dimension is
honest even when a run lands inside a vendor's free allowance. Prices pinned per
vendor public pricing page (2026-06); refresh deliberately, never silently.
"""
from __future__ import annotations

PRICING_VERSION = "2026-06"

# USD per 1M input tokens (query + all candidate documents) for token-priced rerankers.
TOKEN_PRICES: dict[str, float] = {
    "voyage": 0.05,   # rerank-2 ($0.05 / 1M tok; first 200M tok free per account)
    "jina": 0.02,     # jina-reranker-v2 ($0.02 / 1M tok; first 10M tok free per key)
}

# USD per "search" (one query, <=100 docs each <500 tok) for search-priced rerankers.
SEARCH_PRICES: dict[str, float] = {
    "cohere": 0.002,  # rerank-3.5 ($2.00 / 1k searches)
}


def token_cost(vendor: str, total_tokens: int) -> float:
    """List-price USD for a token-billed rerank call (0.0 if vendor unknown)."""
    return total_tokens * TOKEN_PRICES.get(vendor, 0.0) / 1_000_000.0


def search_cost(vendor: str, searches: float = 1.0) -> float:
    """List-price USD for a search-billed rerank call (0.0 if vendor unknown)."""
    return searches * SEARCH_PRICES.get(vendor, 0.0)
