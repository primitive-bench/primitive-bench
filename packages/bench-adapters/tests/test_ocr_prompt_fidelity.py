"""Guard: the shared OCR prompt must stay byte-identical to olmOCR's own image-only
bench prompt, so our pass@test numbers reflect the same instruction olmOCR-bench
applies (not a paraphrase). Source: allenai/olmocr olmocr/bench/prompts.py::
build_openai_silver_data_prompt_no_document_anchoring (Apache-2.0). If this fails,
the prompt drifted — re-sync it or update this expected text deliberately.
"""
from bench_adapters.ocr.adapters import OCR_PROMPT

OLMOCR_NO_ANCHORING_PROMPT = (
    "Below is the image of one page of a PDF document. "
    "Just return the plain text representation of this document as if you were reading it naturally.\n"
    "Turn equations into a LaTeX representation, and tables into markdown format. "
    "Remove the headers and footers, but keep references and footnotes.\n"
    "Read any natural handwriting.\n"
    "This is likely one page out of several in the document, so be sure to preserve "
    "any sentences that come from the previous page, or continue onto the next page, "
    "exactly as they are.\n"
    "If there is no text at all that you think you should read, you can output null.\n"
    "Do not hallucinate."
)


def test_ocr_prompt_is_olmocr_verbatim():
    assert OCR_PROMPT == OLMOCR_NO_ANCHORING_PROMPT
