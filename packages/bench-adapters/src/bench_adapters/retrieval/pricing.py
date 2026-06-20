"""Pinned price table for retrieval (embedding) cost capture (USD).

Hosted embedders are token-priced (per input token over the query + every candidate
document). `cost_usd` on each result records the **list price** of the call (what it
would cost outside a free tier), so the leaderboard's cost dimension stays honest even
when a run lands inside a vendor's free allowance. Prices pinned per vendor public
pricing page (verified 2026-06); refresh deliberately, never silently.
"""
from __future__ import annotations

PRICING_VERSION = "2026-06"

# USD per 1M input tokens, keyed by the adapter's `price_key`.
TOKEN_PRICES: dict[str, float] = {
    "openai-3-large": 0.13,   # text-embedding-3-large ($0.13 / 1M tok)
    "openai-3-small": 0.02,   # text-embedding-3-small ($0.02 / 1M tok)
    "cohere-embed-v4": 0.12,  # embed-v4.0 text ($0.12 / 1M tok)
    "voyage-4-large": 0.12,   # voyage-4-large ($0.12 / 1M tok; first 200M tok free per account)
}


def token_cost(price_key: str, total_tokens: int) -> float:
    """List-price USD for a token-billed embedding call (0.0 if key unknown)."""
    return total_tokens * TOKEN_PRICES.get(price_key, 0.0) / 1_000_000.0
