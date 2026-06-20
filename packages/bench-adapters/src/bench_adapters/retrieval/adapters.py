"""Concrete RETRIEVAL adapters: (query, candidate pool) -> ranked candidate ids.

A retriever here is a *bi-encoder*: embed the query and embed each candidate document
independently, then rank candidates by cosine similarity to the query. `invoke(item)`
reads `item['query']` and `item['candidates']` (a list of `{"id", "text"}` — a BM25
per-query pool from the golden set), embeds + ranks, and returns the bench-adapters
result dict with `retrieved_ids` (the candidate ids best-first). This is first-stage
retrieval restricted to a fixed pool, which keeps the eval keyless/deterministic — the
contrast with the reranker vertical (cross-encoders) is the model class, not the harness.

Two free local bi-encoders (sentence-transformers) act as the keyless baseline /
regression sentinel; three hosted embedding APIs (OpenAI, Cohere, Voyage) are the
systems-under-test. Following the rerank/search/extract adapters: keys are read
straight from the environment and `VendorUnavailable` is raised when a vendor cannot
run (missing key, missing dep, model load failure) so the harness skips that lane
cleanly instead of charging it a miss it never had a chance at.

`cost_usd` records the call's **list price** (see pricing.py) even when the run is
inside a vendor's free tier — so the leaderboard's cost dimension stays honest.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any

import httpx

from bench_adapters.registry import Adapter, register
from bench_adapters.retrieval import pricing

EMBED_TIMEOUT = 60.0


class VendorUnavailable(Exception):
    """Raised when a retrieval adapter cannot run (missing key / dep / model)."""


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


def _candidates(item: dict[str, Any]) -> list[dict[str, Any]]:
    """The candidate pool to rank: [{"id", "text"}, ...] (empty if absent)."""
    return list(item.get("candidates") or [])


def _cosine_rank(qvec: list[float], dvecs: list[list[float]]) -> list[int]:
    """Indices of `dvecs` sorted by descending cosine similarity to `qvec`.

    Pure-Python (no numpy import at module load, so `import bench_adapters` stays safe
    on a minimal install); a ~100-candidate pool of a few-hundred dims is trivial."""
    qn = math.sqrt(sum(x * x for x in qvec)) or 1e-12
    sims: list[tuple[float, int]] = []
    for i, d in enumerate(dvecs):
        dot = sum(a * b for a, b in zip(qvec, d))
        dn = math.sqrt(sum(x * x for x in d)) or 1e-12
        sims.append((dot / (qn * dn), i))
    sims.sort(key=lambda t: -t[0])
    return [i for _s, i in sims]


class _RetrievalAdapter(Adapter):
    """Base: turn (query, candidate pool) into a ranked list of candidate ids.

    Subclasses implement `retrieve(query, candidates) -> (ranked_ids, cost_usd)`.
    `invoke(item)` wraps that with latency measurement and the result-dict shape.
    """

    name: str = ""
    vendor: str = ""
    model_version: str = "unknown"
    is_sentinel: bool = False

    def retrieve(self, query: str, candidates: list[dict[str, Any]]) -> tuple[list[str], float]:
        raise NotImplementedError

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        query = str(item.get("query") or "")
        cands = _candidates(item)
        t0 = time.monotonic()
        retrieved_ids, cost = self.retrieve(query, cands)
        latency_ms = (time.monotonic() - t0) * 1000.0
        return {
            "raw_output": ",".join(retrieved_ids),
            "retrieved_ids": retrieved_ids,
            "latency_ms": latency_ms,
            "cost_usd": cost,
        }


# --------------------------------------------------------------------------- #
# Free local bi-encoders (sentence-transformers). No key, no network, $0.
# --------------------------------------------------------------------------- #
class _LocalBiEncoder(_RetrievalAdapter):
    """Embed query + each candidate with a bi-encoder, rank by cosine.

    The model is loaded lazily and cached on the class (one load per process).
    sentence-transformers / the model weights missing -> VendorUnavailable, so a
    keyless machine simply skips the lane instead of crashing the run. Optional
    `query_prefix`/`doc_prefix` carry the model's recommended retrieval instructions
    (e.g. E5's `query:`/`passage:`), which materially affect retrieval quality.
    """

    vendor = "local"
    model_id = ""
    query_prefix = ""
    doc_prefix = ""
    _model: Any = None

    def _load(self) -> Any:
        if type(self)._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # dep missing
                raise VendorUnavailable(f"{self.name}: sentence-transformers not installed ({exc})")
            try:
                type(self)._model = SentenceTransformer(self.model_id)
            except Exception as exc:  # weights download / load failed
                raise VendorUnavailable(f"{self.name}: model load failed ({exc})")
        return type(self)._model

    def retrieve(self, query: str, candidates: list[dict[str, Any]]) -> tuple[list[str], float]:
        if not candidates:
            return [], 0.0
        model = self._load()
        qvec = model.encode([self.query_prefix + query], normalize_embeddings=True)[0].tolist()
        docs = [self.doc_prefix + str(c.get("text", "")) for c in candidates]
        dvecs = [v.tolist() for v in model.encode(docs, normalize_embeddings=True)]
        order = _cosine_rank(qvec, dvecs)
        return [str(candidates[i]["id"]) for i in order], 0.0


@register("bge-small")
class BgeSmall(_LocalBiEncoder):
    """BAAI bge-small-en-v1.5 — tiny, fast, keyless. The regression sentinel."""

    name = "bge-small"
    model_id = "BAAI/bge-small-en-v1.5"
    model_version = "bge-small-en-v1.5"
    query_prefix = "Represent this sentence for searching relevant passages: "
    is_sentinel = True


@register("e5-small")
class E5Small(_LocalBiEncoder):
    """intfloat/e5-small-v2 — strong small open-weights bi-encoder (still $0)."""

    name = "e5-small"
    model_id = "intfloat/e5-small-v2"
    model_version = "e5-small-v2"
    query_prefix = "query: "
    doc_prefix = "passage: "


# --------------------------------------------------------------------------- #
# Hosted embedding APIs. Key from env; cost = list price (see pricing.py).
# --------------------------------------------------------------------------- #
class _ApiEmbedder(_RetrievalAdapter):
    """Base for hosted embedding endpoints. Subclasses set the endpoint/model/key and
    implement `_payload` + `_parse`. Query and documents are embedded separately so
    asymmetric models (Cohere/Voyage input types) get the right instruction."""

    endpoint = ""
    model = ""
    key_envs: tuple[str, ...] = ()
    price_key = ""

    def _headers(self, key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _payload(self, texts: list[str], *, is_query: bool) -> dict[str, Any]:
        raise NotImplementedError

    def _parse(self, data: dict[str, Any]) -> tuple[list[list[float]], int]:
        """-> (one vector per input text, total_tokens for the call)."""
        raise NotImplementedError

    def _embed(self, texts: list[str], *, is_query: bool, key: str) -> tuple[list[list[float]], int]:
        with httpx.Client(timeout=EMBED_TIMEOUT) as client:
            r = client.post(self.endpoint, headers=self._headers(key),
                            json=self._payload(texts, is_query=is_query))
            r.raise_for_status()
            data = r.json()
        return self._parse(data)

    def retrieve(self, query: str, candidates: list[dict[str, Any]]) -> tuple[list[str], float]:
        if not candidates:
            return [], 0.0
        key = _need(_env(*self.key_envs), self.name)
        qvecs, qtok = self._embed([query], is_query=True, key=key)
        docs = [str(c.get("text", "")) for c in candidates]
        dvecs, dtok = self._embed(docs, is_query=False, key=key)
        order = _cosine_rank(qvecs[0], dvecs)
        ids = [str(candidates[i]["id"]) for i in order]
        return ids, pricing.token_cost(self.price_key, qtok + dtok)


@register("openai-embed")
class OpenAIEmbed(_ApiEmbedder):
    name = "openai-embed"
    vendor = "openai"
    model = "text-embedding-3-large"
    model_version = "text-embedding-3-large"
    endpoint = "https://api.openai.com/v1/embeddings"
    key_envs = ("OPENAI_API_KEY",)
    price_key = "openai-3-large"

    def _payload(self, texts: list[str], *, is_query: bool) -> dict[str, Any]:
        return {"model": self.model, "input": texts}  # symmetric; no input_type

    def _parse(self, data: dict[str, Any]) -> tuple[list[list[float]], int]:
        vecs = [d["embedding"] for d in data.get("data", [])]
        tokens = int((data.get("usage") or {}).get("total_tokens", 0))
        return vecs, tokens


@register("openai-embed-small")
class OpenAIEmbedSmall(OpenAIEmbed):
    """Cheap OpenAI lane (text-embedding-3-small) — not in DEFAULT_VENDORS."""

    name = "openai-embed-small"
    model = "text-embedding-3-small"
    model_version = "text-embedding-3-small"
    price_key = "openai-3-small"


@register("voyage-embed")
class VoyageEmbed(_ApiEmbedder):
    name = "voyage-embed"
    vendor = "voyage"
    model = "voyage-4-large"
    model_version = "voyage-4-large"
    endpoint = "https://api.voyageai.com/v1/embeddings"
    key_envs = ("VOYAGE_API_KEY",)
    price_key = "voyage-4-large"

    def _payload(self, texts: list[str], *, is_query: bool) -> dict[str, Any]:
        return {"model": self.model, "input": texts,
                "input_type": "query" if is_query else "document"}

    def _parse(self, data: dict[str, Any]) -> tuple[list[list[float]], int]:
        vecs = [d["embedding"] for d in data.get("data", [])]
        tokens = int((data.get("usage") or {}).get("total_tokens", 0))
        return vecs, tokens


@register("cohere-embed")
class CohereEmbed(_ApiEmbedder):
    name = "cohere-embed"
    vendor = "cohere"
    model = "embed-v4.0"
    model_version = "embed-v4.0"
    endpoint = "https://api.cohere.com/v2/embed"
    key_envs = ("COHERE_API_KEY",)
    price_key = "cohere-embed-v4"

    def _payload(self, texts: list[str], *, is_query: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "texts": texts,
            "input_type": "search_query" if is_query else "search_document",
            "embedding_types": ["float"],
        }

    def _parse(self, data: dict[str, Any]) -> tuple[list[list[float]], int]:
        vecs = ((data.get("embeddings") or {}).get("float")) or []
        billed = (data.get("meta") or {}).get("billed_units") or {}
        tokens = int(billed.get("input_tokens", 0))
        return vecs, tokens


ALL = [BgeSmall, E5Small, OpenAIEmbed, OpenAIEmbedSmall, VoyageEmbed, CohereEmbed]
