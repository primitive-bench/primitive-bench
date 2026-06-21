"""Top-k dense retrieval over a chunker's chunks (the downstream task).

The embedder rows are L2-normalized, so cosine similarity is a dot product. We embed
every chunk once per (corpus, chunker), then answer each query by ranking chunks by
``chunk_embeddings @ query``. This is the minimal, exact retrieval the chunking
benchmark needs — no external vector DB, deterministic given the embedder.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from eval_chunking.embedders import Embedder


def embed_chunks(embedder: Embedder, chunks: Sequence[dict[str, Any]]) -> np.ndarray:
    """(n_chunks, dim) L2-normalized embedding matrix for a chunker's chunks."""
    if not chunks:
        return np.zeros((0, 1), dtype="float32")
    return embedder.embed([c["text"] for c in chunks])


def retrieve(
    embedder: Embedder,
    chunk_embeddings: np.ndarray,
    chunks: Sequence[dict[str, Any]],
    query: str,
    k: int,
) -> list[dict[str, Any]]:
    """Top-k chunks for `query` by cosine similarity (highest first)."""
    if chunk_embeddings.shape[0] == 0 or not chunks:
        return []
    q = embedder.embed([query])
    if q.shape[0] == 0:
        return []
    sims = chunk_embeddings @ q[0]
    k = min(k, len(chunks))
    # argpartition for the top-k, then sort that slice by score descending.
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [chunks[i] for i in top]
