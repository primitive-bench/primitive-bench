"""Load chunking golden rows + their corpora from a public-split directory.

A chunking row is a query + gold reference spans + a ``corpus_id``; the corpus text
itself is a sibling file (``corpora/<corpus_id>.md``), since inlining a 0.5 MB corpus
into every row would be absurd. The loader returns ``(rows, corpora)`` where
``corpora`` maps ``corpus_id -> text`` for exactly the corpora the rows reference.

Self-contained line reader (skips the leading ``#`` canary/comment header), mirroring
``eval_reranker.loader``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Parse a golden JSONL split, skipping blank and ``#`` comment lines."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def load_corpora(corpora_dir: str | Path, corpus_ids: set[str]) -> dict[str, str]:
    """Load ``{corpus_id: text}`` for the given ids from ``corpora_dir`` (.md/.txt)."""
    corpora_dir = Path(corpora_dir)
    out: dict[str, str] = {}
    for cid in sorted(corpus_ids):
        for ext in (".md", ".txt"):
            p = corpora_dir / f"{cid}{ext}"
            if p.exists():
                out[cid] = p.read_text(encoding="utf-8")
                break
        else:
            raise FileNotFoundError(f"corpus '{cid}' not found under {corpora_dir}")
    return out


def load_golden(
    jsonl_path: str | Path, corpora_dir: str | Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return ``(rows, corpora)`` for a split. ``corpora_dir`` defaults to
    ``<jsonl dir>/corpora``."""
    jsonl_path = Path(jsonl_path)
    rows = load_rows(jsonl_path)
    corpora_dir = Path(corpora_dir) if corpora_dir else jsonl_path.parent / "corpora"
    corpus_ids = {str(r.get("corpus_id") or r.get("domain")) for r in rows}
    corpora = load_corpora(corpora_dir, corpus_ids)
    return rows, corpora
