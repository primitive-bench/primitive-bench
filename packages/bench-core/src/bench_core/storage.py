"""Filesystem + DuckDB storage helpers.

Working artifacts are JSONL (append-only, git-friendly audit trail); frozen
snapshots are Parquet. DuckDB is the working query layer over both.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

DATA = Path("data")
RAW = DATA / "raw"
CANDIDATES = DATA / "candidates"
REJECTIONS = DATA / "rejections"
GOLDEN = DATA / "golden"
SPLITS = DATA / "splits"
RESULTS = DATA / "results"
SNAPSHOTS = DATA / "snapshots"

for _d in (RAW, CANDIDATES, REJECTIONS, GOLDEN, SPLITS, RESULTS, SNAPSHOTS):
    _d.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict], *, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    n = 0
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: Path, *, skip_comments: bool = False) -> Iterator[dict]:
    """Yield one dict per JSONL line. With `skip_comments`, `#` lines are skipped
    (public golden splits carry the BIG-bench canary as a leading `#` header)."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or (skip_comments and line.startswith("#")):
                continue
            yield json.loads(line)


def log_rejection(row_id: str, reason: str, detail: str = "", *, ts: str = "") -> None:
    """Nothing is silently dropped (plan rule 4). Every reject is logged."""
    write_jsonl(
        REJECTIONS / "rejections.jsonl",
        [{"row_id": row_id, "reason": reason, "detail": detail, "ts": ts}],
        append=True,
    )


ADJUDICATION_FILE = DATA / "adjudication.jsonl"
PROMOTIONS_FILE = DATA / "promotions.jsonl"


def log_adjudication(row_id: str, action: str, detail: dict, *, ts: str = "") -> None:
    """Public adjudication log: every edge-case ruling (e.g. an auto-promoted
    mirror) is recorded once and applied uniformly."""
    write_jsonl(ADJUDICATION_FILE, [{"row_id": row_id, "action": action,
                                     "detail": detail, "ts": ts}], append=True)
