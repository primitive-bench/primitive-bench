"""Pinned price table for vector-DB query cost capture (USD).

ANN-Benchmarks reports recall-vs-QPS; VectorDBBench adds a cost dimension (QP$ —
queries per dollar). Self-hosted engines (FAISS, hnswlib, Qdrant-local, pgvector,
Milvus, …) have no per-query list price, so `cost_usd = 0.0` — their cost is the
build/serve compute, captured separately as build time + index memory. Hosted
clouds bill per query (read units), so we record the **list price** of each query
(what it would cost outside any free tier), keeping the leaderboard's cost
dimension honest exactly like the rerank pricing table.

CAVEAT: hosted vector-search pricing is usage-shaped (read units scale with top_k,
dimensionality, and replicas) and changes often. The per-1k-query figures below are
deliberately conservative list-price approximations pinned to 2026-06; refresh them
against each vendor's pricing page, never silently.
"""
from __future__ import annotations

PRICING_VERSION = "2026-06"

# USD per 1,000 queries (list-price approximation) for hosted vector search.
QUERY_PRICES_PER_1K: dict[str, float] = {
    "pinecone": 0.40,        # serverless read units, small top_k
    "zilliz-cloud": 0.30,    # Zilliz Cloud (managed Milvus)
    "weaviate-cloud": 0.30,  # Weaviate Cloud Serverless
}


def query_cost(vendor: str, n_queries: float = 1.0) -> float:
    """List-price USD for `n_queries` against a hosted engine (0.0 if self-hosted)."""
    return n_queries * QUERY_PRICES_PER_1K.get(vendor, 0.0) / 1000.0
