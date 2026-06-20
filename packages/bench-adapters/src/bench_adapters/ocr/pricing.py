"""Pinned price table for OCR cost capture (USD).

Token-priced vision models bill per input/output token; dedicated OCR APIs bill
per page. Prices are pinned per model snapshot and recorded (by version) in the
run manifest; refresh deliberately, never silently. Sources: each vendor's public
pricing page (2026-06).
"""
from __future__ import annotations

# (input_usd_per_mtok, output_usd_per_mtok) for token-priced vision models.
TOKEN_PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.5-2026-04-23": (5.00, 30.00),
    "gemini-2.5-pro": (1.25, 10.00),  # <=200K context tier
    "gemini-3.1-pro": (2.00, 12.00),  # <=200K context tier
    "gemini-3.1-pro-preview": (2.00, 12.00),
}

# USD per page for dedicated OCR APIs.
PAGE_PRICES: dict[str, float] = {
    "mistral-ocr-2512": 1.00 / 1000.0,  # $1 / 1k pages (standard)
    "mistral-ocr-latest": 1.00 / 1000.0,
}

PRICING_VERSION = "2026-06"


def token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for a token-priced model call (0.0 if model unknown)."""
    pin, pout = TOKEN_PRICES.get(model, (0.0, 0.0))
    return (input_tokens * pin + output_tokens * pout) / 1_000_000.0


def page_cost(model: str, pages: int = 1) -> float:
    """Cost in USD for a per-page OCR API call (0.0 if model unknown)."""
    return PAGE_PRICES.get(model, 0.0) * pages
