"""OCR Task + Scorer (olmOCR-bench, pass@test).

`Task` yields one item per olmOCR-bench test case (a rule + the page image it
applies to); `Scorer` runs the rule against a vendor's transcription via
`scoring.score_test`. The leaderboard slices each item by document type and test
type, so one board becomes many ("vision LLMs win tables; tesseract holds dense
text") — the "one winner is a lie" surface for OCR.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from bench_core import Scorer as _Scorer, Task as _Task
from bench_schemas import ScorerOutput
from bench_schemas.models import Primitive

from eval_ocr.scoring import score_test

_SLICES_FILE = Path(__file__).resolve().parents[2] / "slices.yaml"


def _transcription(raw: dict[str, Any]) -> str:
    """Pull the page transcription out of a bench_adapters OCR result dict."""
    return str(raw.get("raw_output") or raw.get("text") or "")


class Scorer(_Scorer):
    """pass@test scorer: hit iff the row's olmOCR-bench rule passes on the output.

    A page the adapter couldn't attempt (missing key, refusal, truncation — flagged
    on the raw dict) is an *uncharged* non-attempt (`correct=None`), never a fail.
    Misses are decomposed onto `miss_reason` (absent / fuzzy_miss / wrong_order / …).
    """

    def score(self, item: dict[str, Any], raw: dict[str, Any]) -> ScorerOutput:
        if raw.get("error"):
            return ScorerOutput(correct=None, miss_reason="failed",
                                rationale=str(raw.get("error"))[:200])
        if raw.get("non_attempt"):
            return ScorerOutput(correct=None, miss_reason=str(raw.get("non_attempt")))
        return score_test(item, _transcription(raw))


class Task(_Task):
    primitive = Primitive.OCR
    task_version = "ocr@1"
    dataset_version = "ocr-2026.06"

    def __init__(self, test_cases: Iterable[dict[str, Any]] | None = None) -> None:
        """`test_cases` are olmOCR-bench rows (type/text/before/after/… + page_image).

        When omitted the Task carries no rows (the public DEV split is loaded by the
        harness via `eval_ocr.loader.load_rows`). Each yielded item keeps the full
        rule payload (the Scorer dispatches on `type`) plus id/slices/tier/image.
        """
        self._rows = list(test_cases or [])

    def items(self) -> Iterable[dict[str, Any]]:
        for r in self._rows:
            item = dict(r)  # keep the full rule payload for the Scorer
            item["id"] = r.get("row_id") or r.get("id")
            item["slices"] = _slices_for(r)
            item["ground_truth_tier"] = r.get("ground_truth_tier", "verified_external")
            yield item

    def scorer(self) -> _Scorer:
        return Scorer()


def _slices_for(row: dict[str, Any]) -> list[str]:
    """Per-item slice tags: document type × test type, plus any explicit row slices."""
    out: list[str] = list(row.get("slices", []))
    doc_type = row.get("doc_type")
    if doc_type:
        out.append(f"doc_type:{doc_type}")
    ttype = row.get("type")
    if ttype:
        out.append(f"test_type:{ttype}")
    return list(dict.fromkeys(out))  # de-dupe, preserve order


def load_slices() -> list[dict[str, Any]]:
    """Read the OCR slice definitions from slices.yaml."""
    if not _SLICES_FILE.exists():
        return []
    data = yaml.safe_load(_SLICES_FILE.read_text()) or {}
    return list(data.get("slices", []))
