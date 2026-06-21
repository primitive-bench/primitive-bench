"""Concrete CHUNK adapters: a document -> a list of character-span chunks.

A *chunker* is the system-under-test for the chunking primitive. Unlike a reranker
(a pure function over one query's candidate list), a chunker is a pure function
over a **document**: `invoke({"document": text, ...})` returns the chunks it would
index, each as `{"text", "start", "end"}` where `start`/`end` are character offsets
into the source document. The offsets are load-bearing — the eval scores token-range
overlap between retrieved chunks and character-indexed gold spans, so a chunker that
returned only text (no offsets) could not be scored.

The families mirror the strategies compared in Chroma's "Evaluating Chunking
Strategies for Retrieval" (2024) plus the ubiquitous production default:

  * ``fixed-token``      — fixed-width token windows with overlap (the naive
                           baseline / regression **sentinel**; expected stable, not
                           to win). LangChain ``CharacterTextSplitter`` family.
  * ``recursive``        — hierarchical split on ``["\\n\\n","\\n",".","?","!"," ",""]``
                           merged up to a token budget. LangChain
                           ``RecursiveCharacterTextSplitter`` — the most-deployed
                           strategy and the report's strong, cheap performer.
  * ``sentence``         — greedy sentence packing to a token budget (respects
                           sentence boundaries; no mid-sentence cuts).
  * ``semantic``         — embedding breakpoint chunking (Kamradt): split into
                           sentences, embed, cut where adjacent cosine distance
                           exceeds a percentile threshold.
  * ``cluster-semantic`` — Chroma's ClusterSemanticChunker: small base pieces,
                           embedded, merged by dynamic programming to maximize
                           intra-chunk similarity under a max-size budget.

``semantic`` and ``cluster-semantic`` need embeddings; the eval injects a shared,
corpus-fixed embedder as ``item["embed"]`` (a ``list[str] -> np.ndarray`` of
L2-normalized rows) so the *embedder is held constant across chunkers* and only the
chunking varies. If no embedder is injected they raise ``VendorUnavailable`` and the
harness skips the lane cleanly (mirroring a missing API key elsewhere).

Chunkers are local and free, so ``cost_usd`` is 0.0 — the embedding cost (shared by
all chunkers) is accounted for by the eval, not charged to a chunker.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable, Sequence

from bench_adapters.chunk.tokenization import LengthFn, get_length_function, token_spans
from bench_adapters.registry import Adapter, register

# Char span of a chunk in the source document.
Span = tuple[int, int]

# The canonical recursive separator hierarchy (Chroma RecursiveTokenChunker /
# LangChain RecursiveCharacterTextSplitter): paragraph -> line -> sentence -> word.
RECURSIVE_SEPARATORS = ["\n\n", "\n", ".", "?", "!", " ", ""]

# Sentence boundary: end punctuation followed by whitespace (kept simple + offline).
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)", re.UNICODE)


class VendorUnavailable(Exception):
    """Raised when a chunker cannot run (e.g. a semantic chunker with no embedder)."""


# --------------------------------------------------------------------------- #
# Shared span helpers
# --------------------------------------------------------------------------- #
def _spans_to_chunks(text: str, spans: Sequence[Span]) -> list[dict[str, Any]]:
    """Materialize char spans into chunk dicts, dropping empty/whitespace spans."""
    out: list[dict[str, Any]] = []
    for start, end in spans:
        if end <= start:
            continue
        body = text[start:end]
        if not body.strip():
            continue
        out.append({"text": body, "start": int(start), "end": int(end)})
    return out


def _sentence_spans(text: str) -> list[Span]:
    """Character spans of sentences (end punctuation / newline delimited)."""
    return [(m.start(), m.end()) for m in _SENTENCE_RE.finditer(text) if m.end() > m.start()]


def _pack_spans(units: Sequence[Span], text: str, max_tokens: int, length_fn: LengthFn,
                overlap_units: int = 0) -> list[Span]:
    """Greedily merge contiguous unit spans into chunk spans under a token budget.

    A unit (sentence / piece) is never split: if a single unit already exceeds
    `max_tokens` it becomes its own chunk. `overlap_units` re-seeds each new chunk
    with the trailing N units of the previous one (sentence-level overlap).
    """
    chunks: list[Span] = []
    cur: list[Span] = []
    for span in units:
        candidate = cur + [span]
        merged_text = text[candidate[0][0]:candidate[-1][1]]
        if cur and length_fn(merged_text) > max_tokens:
            chunks.append((cur[0][0], cur[-1][1]))
            cur = cur[-overlap_units:] + [span] if overlap_units else [span]
        else:
            cur = candidate
    if cur:
        chunks.append((cur[0][0], cur[-1][1]))
    return chunks


# --------------------------------------------------------------------------- #
# Base adapter
# --------------------------------------------------------------------------- #
class _ChunkAdapter(Adapter):
    """Base: turn one document into character-span chunks.

    Subclasses implement ``chunk(text, embed) -> list[Span]``. ``invoke`` wraps that
    with latency measurement and the bench-adapters result-dict shape. ``embed`` is
    the injected shared embedder (or ``None``).
    """

    name: str = ""
    vendor: str = "local"
    model_version: str = "unknown"
    is_sentinel: bool = False
    needs_embeddings: bool = False

    # Tunables (set per registered subclass).
    chunk_size: int = 200          # token budget per chunk (the report's sweet spot)
    chunk_overlap: int = 0
    length_name: str | None = None  # tokenizer for the length function

    def __init__(self, spec: Any | None = None):
        super().__init__(spec)
        self.length_fn: LengthFn = get_length_function(self.length_name)

    def chunk(self, text: str, embed: Callable[[list[str]], Any] | None) -> list[Span]:
        raise NotImplementedError

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        text = str(item.get("document") or "")
        embed = item.get("embed")
        if self.needs_embeddings and embed is None:
            raise VendorUnavailable(f"{self.name}: no embedder injected (needs_embeddings)")
        t0 = time.monotonic()
        spans = self.chunk(text, embed) if text.strip() else []
        latency_ms = (time.monotonic() - t0) * 1000.0
        chunks = _spans_to_chunks(text, spans)
        return {
            "chunks": chunks,
            "n_chunks": len(chunks),
            "raw_output": f"{len(chunks)} chunks",
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
        }


# --------------------------------------------------------------------------- #
# fixed-token — naive fixed-width token windows (sentinel)
# --------------------------------------------------------------------------- #
class _FixedToken(_ChunkAdapter):
    """Fixed token windows with overlap. Cuts mid-sentence — the naive baseline."""

    def chunk(self, text: str, embed: Callable[[list[str]], Any] | None) -> list[Span]:
        toks = token_spans(text)
        if not toks:
            return []
        size = max(1, self.chunk_size)
        step = max(1, size - self.chunk_overlap)
        spans: list[Span] = []
        for i in range(0, len(toks), step):
            window = toks[i:i + size]
            if not window:
                break
            spans.append((window[0][0], window[-1][1]))
            if i + size >= len(toks):
                break
        return spans


@register("fixed-token")
class FixedToken200(_FixedToken):
    name = "fixed-token"
    model_version = "fixed-token@200/0"
    is_sentinel = True
    chunk_size = 200
    chunk_overlap = 0


# --------------------------------------------------------------------------- #
# recursive — hierarchical separator splitting (production default)
# --------------------------------------------------------------------------- #
class _Recursive(_ChunkAdapter):
    """Recursive separator split, offset-preserving, merged to a token budget.

    Atomic pieces are produced by descending the separator hierarchy until a piece
    fits the budget (or no separator remains), then greedily merged back up to the
    budget. Faithful to RecursiveCharacterTextSplitter while keeping exact offsets.
    """

    separators = RECURSIVE_SEPARATORS

    def _atoms(self, text: str, base: int, seps: list[str]) -> list[Span]:
        """Spans no larger than the budget where the separators allow it."""
        if not text:
            return []
        if self.length_fn(text) <= self.chunk_size or not seps:
            return [(base, base + len(text))]
        sep = seps[0]
        rest = seps[1:]
        if sep == "":
            return [(base, base + len(text))]
        pieces: list[Span] = []
        cursor = 0
        # keep_separator: attach the separator to the END of the preceding piece.
        for part in _split_keepsep(text, sep):
            if part:
                start = base + cursor
                pieces.extend(self._atoms(part, start, rest))
                cursor += len(part)
        return pieces

    def chunk(self, text: str, embed: Callable[[list[str]], Any] | None) -> list[Span]:
        atoms = self._atoms(text, 0, list(self.separators))
        return _merge_atoms(atoms, text, self.chunk_size, self.chunk_overlap, self.length_fn)


def _split_keepsep(text: str, sep: str) -> list[str]:
    """Split on `sep`, keeping the separator attached to the preceding fragment."""
    parts = text.split(sep)
    out: list[str] = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            out.append(p + sep)
        elif p:
            out.append(p)
    return out


def _merge_atoms(atoms: Sequence[Span], text: str, max_tokens: int, overlap: int,
                 length_fn: LengthFn) -> list[Span]:
    """Greedily merge contiguous atom spans up to the token budget (token overlap)."""
    chunks: list[Span] = []
    cur: list[Span] = []
    for span in atoms:
        candidate = cur + [span]
        if cur and length_fn(text[candidate[0][0]:candidate[-1][1]]) > max_tokens:
            chunks.append((cur[0][0], cur[-1][1]))
            cur = _overlap_tail(cur, text, overlap, length_fn) + [span] if overlap else [span]
        else:
            cur = candidate
    if cur:
        chunks.append((cur[0][0], cur[-1][1]))
    return chunks


def _overlap_tail(spans: list[Span], text: str, overlap: int, length_fn: LengthFn) -> list[Span]:
    """Trailing spans whose combined length is ~`overlap` tokens (for re-seeding)."""
    tail: list[Span] = []
    for span in reversed(spans):
        tail.insert(0, span)
        if length_fn(text[tail[0][0]:tail[-1][1]]) >= overlap:
            break
    return tail


@register("recursive")
class Recursive200(_Recursive):
    name = "recursive"
    model_version = "recursive@200/0"
    chunk_size = 200
    chunk_overlap = 0


# --------------------------------------------------------------------------- #
# sentence — greedy sentence packing to a token budget
# --------------------------------------------------------------------------- #
class _Sentence(_ChunkAdapter):
    """Pack whole sentences up to the budget; never cut a sentence."""

    def chunk(self, text: str, embed: Callable[[list[str]], Any] | None) -> list[Span]:
        sents = _sentence_spans(text)
        if not sents:
            return [(0, len(text))]
        return _pack_spans(sents, text, self.chunk_size, self.length_fn,
                           overlap_units=1 if self.chunk_overlap else 0)


@register("sentence")
class Sentence200(_Sentence):
    name = "sentence"
    model_version = "sentence@200"
    chunk_size = 200
    chunk_overlap = 0


# --------------------------------------------------------------------------- #
# semantic — embedding breakpoint chunking (Kamradt)
# --------------------------------------------------------------------------- #
class _Semantic(_ChunkAdapter):
    """Cut where adjacent-sentence embedding distance exceeds a percentile.

    Sentences are embedded with the injected shared embedder; the breakpoint
    threshold is the `breakpoint_percentile` of adjacent cosine distances. A
    `max_chunk_size` cap keeps a low-variance run from forming one giant chunk.
    """

    needs_embeddings = True
    breakpoint_percentile = 95.0

    def chunk(self, text: str, embed: Callable[[list[str]], Any] | None) -> list[Span]:
        import numpy as np

        sents = _sentence_spans(text)
        if len(sents) <= 1:
            return list(sents) or [(0, len(text))]
        embs = np.asarray(embed([text[s:e] for s, e in sents]), dtype=float)  # type: ignore[misc]
        # Rows arrive L2-normalized -> cosine similarity is the dot product.
        sims = np.sum(embs[:-1] * embs[1:], axis=1)
        distances = 1.0 - sims
        threshold = float(np.percentile(distances, self.breakpoint_percentile))

        groups: list[list[Span]] = [[sents[0]]]
        for i in range(1, len(sents)):
            over_budget = self.length_fn(text[groups[-1][0][0]:sents[i][1]]) > self.chunk_size
            if distances[i - 1] > threshold or over_budget:
                groups.append([sents[i]])
            else:
                groups[-1].append(sents[i])
        return [(g[0][0], g[-1][1]) for g in groups]


@register("semantic")
class Semantic(_Semantic):
    name = "semantic"
    model_version = "semantic-kamradt@200/p95"
    chunk_size = 200             # hard cap; matched to the other strategies (size held constant)
    breakpoint_percentile = 95.0


# --------------------------------------------------------------------------- #
# cluster-semantic — Chroma ClusterSemanticChunker (DP over piece similarities)
# --------------------------------------------------------------------------- #
class _ClusterSemantic(_ChunkAdapter):
    """Merge small embedded pieces by DP to maximize intra-chunk similarity.

    Faithful to Chroma's ClusterSemanticChunker: split into ~`min_chunk_size`-token
    base pieces (recursive), embed them, then choose contiguous clusters that
    maximize summed pairwise (mean-centered) similarity, with each cluster bounded by
    `max_chunk_size` tokens and at most `max_chunk_size//min_chunk_size` pieces. The
    reward is computed inside a bounded DP window, so memory is O(n) rather than the
    O(n^2) of a full similarity matrix (lets it run on large corpora).
    """

    needs_embeddings = True
    min_chunk_size = 50
    max_chunk_size = 400

    def _base_pieces(self, text: str) -> list[Span]:
        splitter = _Recursive()
        splitter.chunk_size = self.min_chunk_size
        splitter.length_fn = self.length_fn
        return splitter.chunk(text, None) or [(0, len(text))]

    def chunk(self, text: str, embed: Callable[[list[str]], Any] | None) -> list[Span]:
        import numpy as np

        pieces = self._base_pieces(text)
        n = len(pieces)
        if n <= 1:
            return pieces
        embs = np.asarray(embed([text[s:e] for s, e in pieces]), dtype=float)  # type: ignore[misc]
        embs = embs - embs.mean(axis=0, keepdims=True)  # mean-center (Chroma)
        max_pieces = max(1, self.max_chunk_size // max(1, self.min_chunk_size))

        # DP over piece prefixes: dp[i] = best reward for pieces[:i]; back[i] = the
        # chosen cluster start. reward(a..b) = sum of upper-triangle similarities.
        dp = [0.0] * (n + 1)
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            best = float("-inf")
            best_start = i - 1
            for size in range(1, max_pieces + 1):
                start = i - size
                if start < 0:
                    break
                if self.length_fn(text[pieces[start][0]:pieces[i - 1][1]]) > self.max_chunk_size \
                        and size > 1:
                    break
                block = embs[start:i]
                sim = float(np.triu(block @ block.T, k=1).sum())
                cand = dp[start] + sim
                if cand > best:
                    best = cand
                    best_start = start
            dp[i] = best
            back[i] = best_start

        spans: list[Span] = []
        i = n
        while i > 0:
            start = back[i]
            spans.append((pieces[start][0], pieces[i - 1][1]))
            i = start
        spans.reverse()
        return spans


@register("cluster-semantic")
class ClusterSemantic(_ClusterSemantic):
    name = "cluster-semantic"
    model_version = "cluster-semantic@50-200"
    min_chunk_size = 50
    max_chunk_size = 200         # matched to the other strategies (size held constant)


ALL = [FixedToken200, Recursive200, Sentence200, Semantic, ClusterSemantic]
