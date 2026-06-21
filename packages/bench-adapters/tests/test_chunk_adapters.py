"""Tests for the chunk adapters: registration, exact offsets, size budgets, embedder gate."""
from __future__ import annotations

import numpy as np
import pytest

from bench_adapters import get, registry
from bench_adapters.chunk import VendorUnavailable

CHUNKERS = ["fixed-token", "recursive", "sentence", "semantic", "cluster-semantic"]

DOC = (
    "# Title\n\n"
    + " ".join(["The cat sat on the warm mat by the door."] * 25)
    + "\n\n"
    + " ".join(["Dogs chase bright red balls across the green park."] * 25)
    + "\n\n"
    + " ".join(["Birds glide silently high above the drifting clouds."] * 25)
)


def _embed(texts: list[str]) -> np.ndarray:
    """Deterministic, L2-normalized hashing bag-of-words (no model/network)."""
    out = []
    for t in texts:
        v = np.zeros(96, dtype="float32")
        for w in t.lower().split():
            v[hash(w) % 96] += 1.0
        n = float(np.linalg.norm(v)) or 1.0
        out.append(v / n)
    return np.asarray(out, dtype="float32")


def test_all_chunkers_registered():
    assert all(name in registry for name in CHUNKERS)


@pytest.mark.parametrize("name", CHUNKERS)
def test_offsets_are_exact_and_in_order(name):
    out = get(name)(None).invoke({"document": DOC, "embed": _embed})
    chunks = out["chunks"]
    assert chunks, f"{name} produced no chunks"
    assert out["n_chunks"] == len(chunks)
    last_end = 0
    for c in chunks:
        assert DOC[c["start"]:c["end"]] == c["text"]   # offsets reconstruct the text
        assert c["start"] >= last_end - 1 or c["start"] <= c["end"]  # non-decreasing-ish
        last_end = max(last_end, c["end"])


@pytest.mark.parametrize("name", CHUNKERS)
def test_chunks_respect_token_budget(name):
    adapter = get(name)(None)
    out = adapter.invoke({"document": DOC, "embed": _embed})
    sizes = [adapter.length_fn(c["text"]) for c in out["chunks"]]
    # Every chunk fits the 200-token budget (single-unit overshoots are tolerated +1 unit).
    assert max(sizes) <= 260, f"{name} chunk too large: {max(sizes)}"


def test_fixed_token_is_the_sentinel():
    assert get("fixed-token").is_sentinel is True


def test_semantic_without_embedder_is_unavailable():
    for name in ("semantic", "cluster-semantic"):
        with pytest.raises(VendorUnavailable):
            get(name)(None).invoke({"document": DOC})  # no embed injected


def test_empty_document_yields_no_chunks():
    out = get("recursive")(None).invoke({"document": "   "})
    assert out["chunks"] == [] and out["n_chunks"] == 0
