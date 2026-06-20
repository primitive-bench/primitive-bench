"""Load OCR golden rows from a public-split JSONL file.

Thin wrapper over `bench_core.storage.read_jsonl(skip_comments=True)` (which skips
the canary `#` header) that adds the OCR-specific step: resolving each row's
`page_image` relative to the JSONL file's directory into an absolute path the
adapters can open.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from bench_core.storage import read_jsonl


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Parse a golden JSONL split, skipping the canary header and resolving images."""
    p = Path(path)
    base = p.parent
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(p, skip_comments=True):
        img = row.get("page_image") or row.get("image")
        if img:
            row["page_image"] = str((base / img).resolve())
        rows.append(row)
    return rows
