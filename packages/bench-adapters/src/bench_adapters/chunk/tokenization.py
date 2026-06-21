"""Pluggable token-length functions for the chunk adapters.

A chunker's notion of "size" must be a *token* count (so chunk budgets line up
with what a downstream LLM/embedder sees), and — crucially for the chunking
benchmark — every chunk we emit must carry its exact **character span** in the
source document, because the scorer measures token-range overlap against
character-indexed gold reference spans (Chroma methodology).

`tiktoken` (OpenAI cl100k_base) is the faithful tokenizer but it downloads its BPE
table on first use, so it is unavailable on an offline/CI machine. We therefore
default to a dependency-free, fully-deterministic **regex word/punctuation**
tokenizer that (a) never touches the network and (b) yields per-token character
offsets directly. Because the benchmark compares chunkers under one *shared*
length function, the absolute tokenizer choice does not bias the comparison — it
only rescales chunk_size — so the regex default is a fair, reproducible unit.

Select the faithful tokenizer for a production run by passing
`length="tiktoken"` (or setting `CHUNK_TOKENIZER=tiktoken`) where the BPE table
can be fetched/cached.
"""
from __future__ import annotations

import os
import re
from typing import Callable

# A token is a maximal run of word characters OR a single non-space punctuation
# char. `finditer` hands back the char span of every token for free, which is what
# lets the offset-preserving chunkers below stay exact.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# A callable that returns the number of tokens in a string.
LengthFn = Callable[[str], int]


def token_spans(text: str) -> list[tuple[int, int]]:
    """Character (start, end) span of every token in `text`, in order."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def regex_token_count(text: str) -> int:
    """Offline, deterministic token count (word + punctuation tokens)."""
    return sum(1 for _ in _TOKEN_RE.finditer(text))


_TIKTOKEN_ENC = None


def _tiktoken_count(text: str) -> int:
    """cl100k_base token count (lazy; falls back to regex if unavailable)."""
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        try:
            import tiktoken

            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:  # dep missing or BPE table undownloadable (offline)
            _TIKTOKEN_ENC = False
    if _TIKTOKEN_ENC is False:
        return regex_token_count(text)
    return len(_TIKTOKEN_ENC.encode(text))


def get_length_function(name: str | None = None) -> LengthFn:
    """Resolve a length function by name ('regex' | 'tiktoken').

    Defaults to the env var `CHUNK_TOKENIZER`, then to the offline 'regex' counter.
    """
    name = (name or os.environ.get("CHUNK_TOKENIZER") or "regex").lower()
    if name == "tiktoken":
        return _tiktoken_count
    return regex_token_count
