"""Concrete RERANK adapters: (query, candidate list) -> reordered candidate ids.

A reranker is a *pure function over a fixed candidate list* — no corpus, no index,
no first-stage retrieval. `invoke(item)` reads `item['query']` and
`item['candidates']` (a list of `{"id", "text"}`), scores/reorders the candidates,
and returns the bench-adapters result dict with `reordered_ids` (the candidate ids
in the model's preferred order, best first).

Two free local cross-encoders (sentence-transformers) act as the keyless baseline /
regression sentinel; three hosted rerank APIs (Voyage, Jina, Cohere) are the
systems-under-test. Following the search/extract adapters: keys are read straight
from the environment and `VendorUnavailable` is raised when a vendor cannot run
(missing key, missing dep, model load failure) so the harness skips that lane
cleanly instead of charging it a miss it never had a chance at.

`cost_usd` records the call's **list price** (see pricing.py) even when the run is
inside a vendor's free tier — so the leaderboard's cost dimension stays honest.
"""
from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx

from bench_adapters.registry import Adapter, register
from bench_adapters.rerank import pricing

RERANK_TIMEOUT = 60.0

_RETRY_STATUS = {429, 500, 502, 503, 504}
RERANK_MAX_ATTEMPTS = 7
_MAX_BACKOFF = 30.0


def _post_json_with_retry(client: httpx.Client, url: str, headers: dict[str, str],
                          payload: dict[str, Any]) -> dict[str, Any]:
    """POST with exponential backoff on 429/5xx, honoring Retry-After when present."""
    for attempt in range(RERANK_MAX_ATTEMPTS):
        r = client.post(url, headers=headers, json=payload)
        if r.status_code in _RETRY_STATUS and attempt < RERANK_MAX_ATTEMPTS - 1:
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


class VendorUnavailable(Exception):
    """Raised when a rerank adapter cannot run (missing key / dep / model)."""


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
    """The candidate list to rerank: [{"id", "text"}, ...] (empty if absent)."""
    return list(item.get("candidates") or [])


class _RerankAdapter(Adapter):
    """Base: turn (query, candidates) into a reordered list of candidate ids.

    Subclasses implement `rerank(query, candidates) -> (reordered_ids, cost_usd)`.
    `invoke(item)` wraps that with latency measurement and the result-dict shape.
    """

    name: str = ""
    vendor: str = ""
    model_version: str = "unknown"
    is_sentinel: bool = False

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> tuple[list[str], float]:
        raise NotImplementedError

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        query = str(item.get("query") or "")
        cands = _candidates(item)
        t0 = time.monotonic()
        reordered_ids, cost = self.rerank(query, cands)
        latency_ms = (time.monotonic() - t0) * 1000.0
        return {
            "raw_output": ",".join(reordered_ids),
            "reordered_ids": reordered_ids,
            "latency_ms": latency_ms,
            "cost_usd": cost,
        }


# --------------------------------------------------------------------------- #
# Free local cross-encoders (sentence-transformers). No key, no network, $0.
# --------------------------------------------------------------------------- #
class _LocalCrossEncoder(_RerankAdapter):
    """Score each (query, candidate) pair with a cross-encoder, sort descending.

    The model is loaded lazily and cached on the class (one load per process).
    sentence-transformers / the model weights missing -> VendorUnavailable, so a
    keyless machine simply skips the lane instead of crashing the run.
    """

    vendor = "local"
    model_id = ""
    _model: Any = None

    def _load(self) -> Any:
        if type(self)._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:  # dep missing
                raise VendorUnavailable(f"{self.name}: sentence-transformers not installed ({exc})")
            try:
                type(self)._model = CrossEncoder(self.model_id)
            except Exception as exc:  # weights download / load failed
                raise VendorUnavailable(f"{self.name}: model load failed ({exc})")
        return type(self)._model

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> tuple[list[str], float]:
        if not candidates:
            return [], 0.0
        model = self._load()
        pairs = [[query, str(c.get("text", ""))] for c in candidates]
        scores = model.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
        return [str(candidates[i]["id"]) for i in order], 0.0


@register("ce-minilm")
class CeMiniLM(_LocalCrossEncoder):
    """MS MARCO MiniLM cross-encoder — tiny, fast, keyless. The regression sentinel."""

    name = "ce-minilm"
    model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_version = "ms-marco-MiniLM-L-6-v2"
    is_sentinel = True


@register("bge-reranker")
class BgeReranker(_LocalCrossEncoder):
    """BAAI bge-reranker-v2-m3 — stronger open-weights cross-encoder (still $0)."""

    name = "bge-reranker"
    model_id = "BAAI/bge-reranker-v2-m3"
    model_version = "bge-reranker-v2-m3"


# --------------------------------------------------------------------------- #
# Hosted rerank APIs. Key from env; cost = list price (see pricing.py).
# --------------------------------------------------------------------------- #
class _ApiReranker(_RerankAdapter):
    """Base for hosted rerank endpoints with a `{results:[{index, relevance_score}]}`
    response shape. Subclasses set the endpoint/model/key and a `_cost(data, docs)`."""

    endpoint = ""
    model = ""
    key_envs: tuple[str, ...] = ()

    def _headers(self, key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _payload(self, query: str, docs: list[str]) -> dict[str, Any]:
        return {"model": self.model, "query": query, "documents": docs, "top_n": len(docs)}

    def _results(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """The per-document results array (Voyage calls it `data`, others `results`)."""
        return data.get("results") or data.get("data") or []

    def _cost(self, data: dict[str, Any], n_docs: int) -> float:
        raise NotImplementedError

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> tuple[list[str], float]:
        if not candidates:
            return [], 0.0
        key = _need(_env(*self.key_envs), self.name)
        docs = [str(c.get("text", "")) for c in candidates]
        with httpx.Client(timeout=RERANK_TIMEOUT) as client:
            data = _post_json_with_retry(client, self.endpoint,
                                         self._headers(key), self._payload(query, docs))
        results = self._results(data)
        ranked = sorted(results, key=lambda d: -float(d.get("relevance_score", 0.0)))
        order = [int(d["index"]) for d in ranked if 0 <= int(d.get("index", -1)) < len(candidates)]
        ids = [str(candidates[i]["id"]) for i in order]
        return ids, self._cost(data, len(docs))


@register("voyage-rerank")
class VoyageRerank(_ApiReranker):
    name = "voyage-rerank"
    vendor = "voyage"
    model = "rerank-2"
    model_version = "rerank-2"
    endpoint = "https://api.voyageai.com/v1/rerank"
    key_envs = ("VOYAGE_API_KEY",)

    def _payload(self, query: str, docs: list[str]) -> dict[str, Any]:
        return {"model": self.model, "query": query, "documents": docs, "top_k": len(docs)}

    def _cost(self, data: dict[str, Any], n_docs: int) -> float:
        tokens = int((data.get("usage") or {}).get("total_tokens", 0))
        return pricing.token_cost("voyage", tokens)


@register("jina-rerank")
class JinaRerank(_ApiReranker):
    name = "jina-rerank"
    vendor = "jina"
    model = "jina-reranker-v2-base-multilingual"
    model_version = "jina-reranker-v2-base-multilingual"
    endpoint = "https://api.jina.ai/v1/rerank"
    key_envs = ("JINA_API_KEY",)

    def _cost(self, data: dict[str, Any], n_docs: int) -> float:
        tokens = int((data.get("usage") or {}).get("total_tokens", 0))
        return pricing.token_cost("jina", tokens)


@register("cohere-rerank")
class CohereRerank(_ApiReranker):
    name = "cohere-rerank"
    vendor = "cohere"
    model = "rerank-v3.5"
    model_version = "rerank-v3.5"
    endpoint = "https://api.cohere.com/v2/rerank"
    key_envs = ("COHERE_API_KEY",)

    def _cost(self, data: dict[str, Any], n_docs: int) -> float:
        # Cohere bills "search units": 1 query with <=100 docs (<500 tok each) = 1 unit.
        billed = ((data.get("meta") or {}).get("billed_units") or {}).get("search_units")
        searches = float(billed) if billed is not None else max(1.0, n_docs / 100.0)
        return pricing.search_cost("cohere", searches)


ALL = [CeMiniLM, BgeReranker, VoyageRerank, JinaRerank, CohereRerank]
