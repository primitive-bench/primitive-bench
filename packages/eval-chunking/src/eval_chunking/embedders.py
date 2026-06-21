"""The retrieval embedder — held CONSTANT across chunkers so only chunking varies.

Chunking quality is measured *downstream*: chunk a corpus, embed the chunks, build
an index, retrieve top-k for each query, and score token overlap with the gold
spans. For that comparison to isolate the chunker, the embedder must be identical
across all chunkers — so the eval picks one embedder and shares it (it is also
injected into the semantic chunkers' boundary detection, so even *they* use the same
vectors).

Two tiers:

  * ``tfidf-local`` (DEFAULT) — a corpus-fit TF-IDF vectorizer (scikit-learn). No
    model download, no key, fully deterministic, so the whole loop runs offline and
    in CI. It is a genuine lexical retriever; the absolute scores are lower than a
    dense model's, but because it is held constant it ranks chunkers fairly.
  * dense production embedders — the *best available* models for the published
    snapshot: local sentence-transformers (``st:<model_id>``, e.g.
    ``BAAI/bge-large-en-v1.5``, ``Snowflake/snowflake-arctic-embed-l-v2.0``) and the
    hosted SOTA APIs (OpenAI ``text-embedding-3-large``, Voyage ``voyage-3-large``,
    Cohere ``embed-v4.0``, Jina ``jina-embeddings-v3``, Google
    ``gemini-embedding-001``). These need a download or an API key.

Every embedder returns an ``(n, dim)`` float32 array whose **rows are L2-normalized**,
so cosine similarity is a plain dot product. ``fit(corpus_text)`` is meaningful only
for TF-IDF (it learns the corpus vocabulary/IDF); it is a no-op for pretrained
models. The recommended production embedder for the public snapshot is
``voyage-3-large`` (top of the MTEB retrieval board as of 2026-06); pin whichever
model you publish in ``Task.dataset_version``.
"""
from __future__ import annotations

import os
import re
from typing import Any, Protocol

import numpy as np

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)", re.UNICODE)


class EmbedderUnavailable(Exception):
    """Raised when an embedder cannot run (missing dep / key / model)."""


class Embedder(Protocol):
    name: str

    def fit(self, corpus_text: str) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...
    @property
    def cost_usd(self) -> float: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype("float32")


# --------------------------------------------------------------------------- #
# tfidf-local — offline, deterministic default
# --------------------------------------------------------------------------- #
class TfidfEmbedder:
    """Corpus-fit TF-IDF (L2-normalized). Vocabulary depends only on the corpus —
    not on any chunker — so the embedding space is identical across chunkers."""

    def __init__(self, max_features: int = 4096):
        self.name = "tfidf-local"
        self.max_features = max_features
        self._vec: Any = None
        self._dim = max_features

    def fit(self, corpus_text: str) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception as exc:  # pragma: no cover
            raise EmbedderUnavailable(f"scikit-learn required for tfidf-local: {exc}")
        sentences = [m.group(0) for m in _SENTENCE_RE.finditer(corpus_text) if m.group(0).strip()]
        if len(sentences) < 2:
            sentences = [corpus_text or " "]
        self._vec = TfidfVectorizer(
            max_features=self.max_features, sublinear_tf=True, stop_words="english"
        )
        self._vec.fit(sentences)
        self._dim = len(self._vec.vocabulary_)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._vec is None:
            raise EmbedderUnavailable("tfidf-local used before fit()")
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")
        # TfidfVectorizer rows are already L2-normalized (norm='l2').
        return self._vec.transform(texts).toarray().astype("float32")

    @property
    def cost_usd(self) -> float:
        return 0.0


# --------------------------------------------------------------------------- #
# local sentence-transformers (production, free, needs a download)
# --------------------------------------------------------------------------- #
class SentenceTransformerEmbedder:
    """Any sentence-transformers model id (``st:BAAI/bge-large-en-v1.5`` etc.)."""

    def __init__(self, model_id: str):
        self.name = f"st:{model_id}"
        self.model_id = model_id
        self._model: Any = None

    def fit(self, corpus_text: str) -> None:
        return None  # pretrained; nothing to learn from the corpus

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                raise EmbedderUnavailable(f"sentence-transformers not installed: {exc}")
            try:
                self._model = SentenceTransformer(self.model_id)
            except Exception as exc:  # weights download/load failed (e.g. offline)
                raise EmbedderUnavailable(f"{self.model_id} load failed: {exc}")
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        vecs = self._load().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype="float32")

    @property
    def cost_usd(self) -> float:
        return 0.0


# --------------------------------------------------------------------------- #
# hosted SOTA embedders (production, need an API key)
# --------------------------------------------------------------------------- #
def _env(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return ""


class _HostedEmbedder:
    """Base for hosted embedding APIs. Tracks list-price cost over input tokens."""

    name = ""
    endpoint = ""
    model = ""
    key_envs: tuple[str, ...] = ()
    price_per_1m_tokens = 0.0  # USD / 1M input tokens (list price)
    timeout = 60.0

    def __init__(self) -> None:
        self._cost = 0.0

    def fit(self, corpus_text: str) -> None:
        return None

    def _headers(self, key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _payload(self, texts: list[str]) -> dict[str, Any]:
        return {"model": self.model, "input": texts}

    def _vectors(self, data: dict[str, Any]) -> list[list[float]]:
        return [row["embedding"] for row in data["data"]]

    def _tokens(self, data: dict[str, Any], texts: list[str]) -> int:
        usage = data.get("usage") or {}
        return int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        import httpx

        key = _env(*self.key_envs)
        if not key:
            raise EmbedderUnavailable(f"{self.name}: {'/'.join(self.key_envs)} unset")
        out: list[list[float]] = []
        tokens = 0
        with httpx.Client(timeout=self.timeout) as client:
            for i in range(0, len(texts), 128):  # batch to stay under request limits
                batch = texts[i:i + 128]
                r = client.post(self.endpoint, headers=self._headers(key),
                                json=self._payload(batch))
                r.raise_for_status()
                data = r.json()
                out.extend(self._vectors(data))
                tokens += self._tokens(data, batch)
        self._cost += tokens * self.price_per_1m_tokens / 1_000_000.0
        return _l2_normalize(np.asarray(out, dtype="float32"))

    @property
    def cost_usd(self) -> float:
        return self._cost


class OpenAIEmbedder(_HostedEmbedder):
    name = "openai-text-embedding-3-large"
    endpoint = "https://api.openai.com/v1/embeddings"
    model = "text-embedding-3-large"
    key_envs = ("OPENAI_API_KEY",)
    price_per_1m_tokens = 0.13


class VoyageEmbedder(_HostedEmbedder):
    name = "voyage-3-large"
    endpoint = "https://api.voyageai.com/v1/embeddings"
    model = "voyage-3-large"
    key_envs = ("VOYAGE_API_KEY",)
    price_per_1m_tokens = 0.18


class JinaEmbedder(_HostedEmbedder):
    name = "jina-embeddings-v3"
    endpoint = "https://api.jina.ai/v1/embeddings"
    model = "jina-embeddings-v3"
    key_envs = ("JINA_API_KEY",)
    price_per_1m_tokens = 0.02


class CohereEmbedder(_HostedEmbedder):
    name = "cohere-embed-v4"
    endpoint = "https://api.cohere.com/v2/embed"
    model = "embed-v4.0"
    key_envs = ("COHERE_API_KEY",)
    price_per_1m_tokens = 0.12

    def _payload(self, texts: list[str]) -> dict[str, Any]:
        return {"model": self.model, "texts": texts, "input_type": "search_document",
                "embedding_types": ["float"]}

    def _vectors(self, data: dict[str, Any]) -> list[list[float]]:
        emb = data["embeddings"]
        return emb["float"] if isinstance(emb, dict) else emb

    def _tokens(self, data: dict[str, Any], texts: list[str]) -> int:
        meta = (data.get("meta") or {}).get("billed_units") or {}
        return int(meta.get("input_tokens") or 0)


# Friendly aliases for the strongest current local + hosted embedders.
_ALIASES: dict[str, str] = {
    "bge-large": "st:BAAI/bge-large-en-v1.5",
    "arctic-l": "st:Snowflake/snowflake-arctic-embed-l-v2.0",
    "e5-large": "st:intfloat/e5-large-v2",
    "gte-large": "st:Alibaba-NLP/gte-large-en-v1.5",
}
_HOSTED: dict[str, type[_HostedEmbedder]] = {
    "openai": OpenAIEmbedder,
    "voyage": VoyageEmbedder,
    "jina": JinaEmbedder,
    "cohere": CohereEmbedder,
}


def make_embedder(name: str) -> Embedder:
    """Resolve an embedder spec to an instance.

    'tfidf-local' (default) | 'openai'|'voyage'|'jina'|'cohere' | an alias
    ('bge-large', 'arctic-l', ...) | 'st:<hf-model-id>' for any sentence-transformers
    model.
    """
    name = name or "tfidf-local"
    if name in ("tfidf-local", "tfidf"):
        return TfidfEmbedder()
    if name in _HOSTED:
        return _HOSTED[name]()
    if name in _ALIASES:
        name = _ALIASES[name]
    if name.startswith("st:"):
        return SentenceTransformerEmbedder(name[3:])
    # Bare HF model id -> sentence-transformers.
    return SentenceTransformerEmbedder(name)
