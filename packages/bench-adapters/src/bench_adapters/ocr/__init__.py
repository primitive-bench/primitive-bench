"""OCR vendor adapters (document-page image -> transcription).

Importing this subpackage auto-registers every OCR adapter via the
`@register("name")` decorators in `adapters`.

Two adapter shapes:
  * prompted VLMs (claude-sonnet-ocr, gpt-ocr, gemini-ocr) take the shared,
    version-pinned `OCR_PROMPT`;
  * native OCR systems (tesseract, mistral-ocr, deepseek-ocr) transcribe with no
    prompt.

Each adapter raises `VendorUnavailable` when its key/binary is missing so the
harness skips it cleanly (uncharged) rather than scoring a miss it never had a
chance at.
"""
from __future__ import annotations

from bench_adapters.ocr.adapters import OCR_PROMPT, RateLimited, VendorUnavailable
from bench_adapters.ocr import adapters as adapters  # noqa: F401  (side-effect registration)

__all__ = ["adapters", "OCR_PROMPT", "RateLimited", "VendorUnavailable"]
