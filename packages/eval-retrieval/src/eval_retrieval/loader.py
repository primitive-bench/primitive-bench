"""Load retrieval golden rows from a public-split JSONL file.

Self-contained line reader (skips the leading `#` canary/comment header). Retrieval
rows are self-contained (query + candidate pool texts with graded relevance), so
unlike OCR there is nothing to resolve on disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Parse a golden JSONL split, skipping blank and `#` comment lines."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows
