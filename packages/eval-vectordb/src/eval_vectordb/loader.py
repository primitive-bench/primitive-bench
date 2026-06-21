"""Load a golden JSONL split (skips blank lines and the `#` canary/comment header).

For vectordb the runnable corpora come from `datasets.py` (ANN-Benchmarks HDF5 /
HuggingFace / synthetic). This loader reads the small committed example split
(`golden-sets-public/vectordb/dev.example.jsonl`) — illustrative golden rows (a query
vector + its exact true-neighbor ids) used by tests and docs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows
