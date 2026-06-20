"""eval-ocr — Primitive Bench vertical (OCR, pass@test over olmOCR-bench).

Hand a vendor a document-page image, get back the transcription, and score it by
**pass@test** against the olmOCR-bench unit tests (present/absent/order/...).
pass@test is paired-binary, so the leaderboard's McNemar/Wilson separability gate
applies directly — per-slice winners (or TIE bands), never one global ranking.

Implements `bench_core.Task` + `bench_core.Scorer`. The public DEV split lives in
golden-sets-public/ocr/ (a small example + a local generator); the held-out test
split lives only behind the private eval server. Slices: slices.yaml.
"""

from eval_ocr.task import Task
from eval_ocr.scoring import score_test
from eval_ocr.runner import run_sync

__all__ = ["Task", "score_test", "run_sync"]
