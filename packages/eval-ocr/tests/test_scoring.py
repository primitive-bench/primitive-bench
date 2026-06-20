"""Unit tests for the olmOCR-bench pass@test evaluators (pure, no network/tesseract)."""
from __future__ import annotations

from eval_ocr.scoring import normalize_text, score_test, strip_chatter

PAGE = (
    "Primitive Bench: Evaluating OCR Systems\n"
    "Abstract. We benchmark optical character recognition across document types.\n"
    "1. Introduction\n"
    "The reading order of a multi-column page matters for retrieval."
)


# --- normalization -------------------------------------------------------
def test_normalize_collapses_whitespace_and_folds_fancy_chars():
    assert normalize_text("a   b\n\tc") == "a b c"
    assert normalize_text("“quote”  ‘x’ – y") == '"quote" \'x\' - y'


def test_strip_chatter_removes_fence_and_preamble():
    assert strip_chatter("```\nHELLO\n```") == "HELLO"
    assert strip_chatter("```markdown\nHELLO\n```") == "HELLO"
    assert strip_chatter("Here is the transcription:\nHELLO") == "HELLO"
    assert strip_chatter("plain text") == "plain text"


# --- present / absent ----------------------------------------------------
def test_present_hit():
    out = score_test({"type": "present", "text": "benchmark optical character recognition"}, PAGE)
    assert out.correct is True and out.score == 1.0


def test_present_miss_is_absent():
    out = score_test({"type": "present", "text": "wholly unrelated sentence here"}, PAGE)
    assert out.correct is False and out.miss_reason == "absent"


def test_absent_hit_when_text_not_present():
    out = score_test({"type": "absent", "text": "CONFIDENTIAL DRAFT"}, PAGE)
    assert out.correct is True


def test_absent_miss_when_text_present():
    out = score_test({"type": "absent", "text": "Introduction"}, PAGE)
    assert out.correct is False and out.miss_reason == "unexpected_present"


def test_max_diffs_tolerates_typos():
    typoed = "benchmark optcal charcter recogniton"  # 3 deletions
    strict = score_test({"type": "present", "text": typoed, "max_diffs": 0}, PAGE)
    lenient = score_test({"type": "present", "text": typoed, "max_diffs": 6}, PAGE)
    assert strict.correct is False
    assert lenient.correct is True


def test_case_sensitivity():
    cs = score_test({"type": "present", "text": "PRIMITIVE BENCH", "case_sensitive": True}, PAGE)
    ci = score_test({"type": "present", "text": "PRIMITIVE BENCH", "case_sensitive": False}, PAGE)
    assert cs.correct is False
    assert ci.correct is True


# --- order ---------------------------------------------------------------
def test_order_hit():
    out = score_test({"type": "order", "before": "Abstract", "after": "Introduction"}, PAGE)
    assert out.correct is True


def test_order_wrong_direction():
    out = score_test({"type": "order", "before": "Introduction", "after": "Abstract"}, PAGE)
    assert out.correct is False and out.miss_reason == "wrong_order"


def test_order_fragment_missing():
    out = score_test({"type": "order", "before": "Abstract", "after": "nonexistent fragment"}, PAGE)
    assert out.correct is False and out.miss_reason == "fragment_missing"


# --- baseline ------------------------------------------------------------
def test_baseline_ok():
    assert score_test({"type": "baseline"}, PAGE).correct is True


def test_baseline_empty():
    out = score_test({"type": "baseline"}, "   ")
    assert out.correct is False and out.miss_reason == "empty"


def test_baseline_repetition_loop():
    out = score_test({"type": "baseline"}, "spam " * 60)
    assert out.correct is False and out.miss_reason == "repetition_loop"


# --- deferred / unknown --------------------------------------------------
# --- table -------------------------------------------------------------------
TABLE_MD = "| Region | Sales |\n| --- | --- |\n| North | 100 |\n| South | 200 |"
TABLE_HTML = "<table><tr><th>Region</th><th>Sales</th></tr><tr><td>North</td><td>100</td></tr></table>"


def test_table_cell_with_neighbor_and_heading():
    out = score_test({"type": "table", "cell": "100", "left": "North", "top_heading": "Sales"},
                     TABLE_MD)
    assert out.correct is True


def test_table_wrong_neighbor_fails():
    out = score_test({"type": "table", "cell": "100", "left": "South"}, TABLE_MD)
    assert out.correct is False and out.miss_reason == "table_mismatch"


def test_table_html_parsed():
    assert score_test({"type": "table", "cell": "North", "right": "100"}, TABLE_HTML).correct is True


# --- format ------------------------------------------------------------------
def test_format_bold_hit():
    assert score_test({"type": "format", "format": "bold", "text": "Important"},
                      "Some **Important** note").correct is True


def test_format_heading_miss_when_plain():
    out = score_test({"type": "format", "format": "heading", "text": "Introduction"},
                     "Introduction without a heading marker")
    assert out.correct is False and out.miss_reason == "format_absent"


# --- footnote ----------------------------------------------------------------
def test_footnote_marker_with_context():
    text = "See the prior work[^3] for details.\n\n[^3]: Smith et al. 2020"
    assert score_test({"type": "footnote", "marker": "3", "appears_after_marker": "Smith"},
                      text).correct is True


def test_footnote_absent():
    out = score_test({"type": "footnote", "marker": "9"}, "no footnotes here")
    assert out.correct is False and out.miss_reason == "footnote_absent"


# --- math (exact-match) ------------------------------------------------------
def test_math_exact_match_hit():
    assert score_test({"type": "math", "math": "E = mc^2"},
                      r"the equation \(E = mc^2\) is famous").correct is True


def test_math_mismatch():
    out = score_test({"type": "math", "math": "E = mc^2"}, r"unrelated \(a + b\) here")
    assert out.correct is False and out.miss_reason == "math_mismatch"


def test_math_absent():
    out = score_test({"type": "math", "math": "x^2"}, "no equations at all")
    assert out.correct is False and out.miss_reason == "math_absent"


def test_unknown_type_is_uncharged():
    out = score_test({"type": "wat"}, PAGE)
    assert out.correct is None and out.miss_reason == "unknown_test_type"
