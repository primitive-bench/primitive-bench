"""pass@test scoring for the OCR primitive (olmOCR-bench rules).

Each olmOCR-bench test case is a rule applied to a vendor's page transcription;
the cell is a HIT iff the rule passes. pass@test is a *paired binary* outcome,
exactly what the leaderboard's McNemar/Wilson separability gate consumes.

The matching rules are ported from allenai/olmocr (`olmocr/bench/tests.py` +
`table_parsing`, Apache-2.0) so our numbers stay comparable to the published
olmOCR-bench leaderboard, using the same libraries the benchmark uses
(`rapidfuzz` for fuzzy matching, `fuzzysearch` for reading order).

Test types (`type`): `present`/`absent`, `order`, `baseline`, `table`, `format`,
`footnote`, `math`. Markdown structure is preserved for the structural tests
(table/format parse it); the only cleanup is `strip_chatter`, which removes the
wrapping ```fences and "Here is the transcription:" preambles a prompted VLM adds.

`math` deviation (deliberate, for an elegant pure-Python package): olmOCR-bench's
math test renders LaTeX via KaTeX (JS) and compares images. We implement the
exact-match pass (extract → normalize → compare) and omit the rendered-equality
fallback, so math passes are a strict subset of the published numbers — documented
rather than dragging a JS/browser renderer into the harness.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from rapidfuzz import fuzz

try:  # fuzzysearch is a hard dep; guard only so import never hard-fails in odd envs
    from fuzzysearch import find_near_matches
except Exception:  # pragma: no cover
    find_near_matches = None  # type: ignore[assignment]

from bench_schemas import ScorerOutput

SUPPORTED_TYPES = frozenset(
    {"present", "absent", "order", "baseline", "table", "format", "footnote", "math"}
)

_WS = re.compile(r"\s+")
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", re.DOTALL)
_PREAMBLE = re.compile(
    r"^\s*(here(?:'s| is)[^\n:]*:|transcription:|the (?:transcribed )?text[^\n:]*:)\s*\n",
    re.IGNORECASE,
)
# Fancy-char folding; Unicode whitespace (nbsp, thin/figure spaces, ...) is handled
# by `_WS` (Python `\s`), so only non-space substitutions live here.
_FANCY = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-", "ﬁ": "fi", "ﬂ": "fl",
}


def strip_chatter(text: str) -> str:
    """Remove a wrapping ```fence and a leading 'Here is the transcription:' line,
    preserving the markdown inside (tables/formatting) — only chat scaffolding goes."""
    if not text:
        return ""
    m = _FENCE.match(text.strip())
    if m:
        text = m.group(1)
    return _PREAMBLE.sub("", text, count=1)


def normalize_text(s: str) -> str:
    """NFC + markdown-emphasis stripping + whitespace collapse + fancy-char folding.

    Strips bold/italic markers so '**word**' matches 'word' (olmocr's normalize_text
    contract). Casing is handled per-test via `case_sensitive`, not here.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"<br/?>", " ", s)
    s = re.sub(r"</?(b|i|strong|em)>", "", s)
    s = re.sub(r"(\*\*|__)(.*?)\1", r"\2", s)
    s = re.sub(r"(\*|_)(.*?)\1", r"\2", s)
    for k, v in _FANCY.items():
        s = s.replace(k, v)
    return _WS.sub(" ", s).strip()


def _ratio(a: str, b: str) -> float:
    return fuzz.ratio(a, b) / 100.0


def _partial(needle: str, haystack: str) -> float:
    if not needle:
        return 1.0
    if not haystack:
        return 0.0
    return fuzz.partial_ratio(needle, haystack) / 100.0


def _threshold(text: str, max_diffs: int) -> float:
    return 1.0 - (max_diffs / max(1, len(text)))


# --- present / absent ----------------------------------------------------------
def _presence(test: dict[str, Any], norm: str, *, want: bool) -> ScorerOutput:
    needle = normalize_text(str(test.get("text", "")))
    hay = norm
    if test.get("first_n"):
        hay = hay[: int(test["first_n"])]
    if test.get("last_n"):
        hay = hay[-int(test["last_n"]):]
    if not test.get("case_sensitive", True):
        needle, hay = needle.lower(), hay.lower()
    ratio = _partial(needle, hay)
    found = ratio >= _threshold(needle, int(test.get("max_diffs", 0)))
    correct = found if want else (not found)
    metrics = {"partial_ratio": round(ratio, 4)}
    if correct:
        return ScorerOutput(correct=True, score=1.0, metrics=metrics)
    reason = ("absent" if ratio < 0.5 else "fuzzy_miss") if want else "unexpected_present"
    return ScorerOutput(correct=False, score=0.0, miss_reason=reason, metrics=metrics)


# --- order ---------------------------------------------------------------------
def _order(test: dict[str, Any], norm: str) -> ScorerOutput:
    if find_near_matches is None:  # pragma: no cover
        return ScorerOutput(correct=None, miss_reason="fuzzysearch_unavailable")
    md = int(test.get("max_diffs", 0))
    before, after = normalize_text(str(test.get("before", ""))), normalize_text(str(test.get("after", "")))
    bm = find_near_matches(before, norm, max_l_dist=md) if before else []
    am = find_near_matches(after, norm, max_l_dist=md) if after else []
    if not bm or not am:
        return ScorerOutput(correct=False, score=0.0, miss_reason="fragment_missing")
    ok = min(m.start for m in bm) < max(m.start for m in am)
    return ScorerOutput(correct=ok, score=1.0 if ok else 0.0,
                        miss_reason=None if ok else "wrong_order")


# --- baseline ------------------------------------------------------------------
_REPEAT = re.compile(r"(.{1,40}?)\1{29,}", re.DOTALL)


def _baseline(test: dict[str, Any], norm: str) -> ScorerOutput:
    max_length = test.get("max_length")
    alnum = sum(c.isalnum() for c in norm)
    if max_length is not None:
        ok = alnum <= int(max_length)
        return ScorerOutput(correct=ok, score=1.0 if ok else 0.0,
                            miss_reason=None if ok else "not_blank")
    if alnum < 1:
        return ScorerOutput(correct=False, score=0.0, miss_reason="empty")
    if _REPEAT.search(norm):
        return ScorerOutput(correct=False, score=0.0, miss_reason="repetition_loop")
    return ScorerOutput(correct=True, score=1.0)


# --- table ---------------------------------------------------------------------
def _parse_md_tables(text: str) -> list[list[list[str]]]:
    tables, rows = [], []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|") and len(s) > 1:
            cells = [c.strip() for c in s[1:-1].split("|")]
            if all(set(c) <= set("-: ") for c in cells if c):  # separator row
                continue
            rows.append(cells)
        elif rows:
            tables.append(rows)
            rows = []
    if rows:
        tables.append(rows)
    return tables


def _parse_html_tables(text: str) -> list[list[list[str]]]:
    out = []
    for tbl in re.findall(r"<table[^>]*>(.*?)</table>", text, re.S | re.I):
        rows = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.S | re.I):
            cells = [re.sub(r"<[^>]+>", "", c).strip()
                     for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)]
            if cells:
                rows.append(cells)
        if rows:
            out.append(rows)
    return out


def _cell(grid: list[list[str]], r: int, c: int) -> str | None:
    if 0 <= r < len(grid) and 0 <= c < len(grid[r]):
        return grid[r][c]
    return None


def _table(test: dict[str, Any], raw: str) -> ScorerOutput:
    target = normalize_text(str(test.get("cell", "")))
    md = int(test.get("max_diffs", 0))

    def passes(a: str, b: str) -> bool:
        return _ratio(normalize_text(a), normalize_text(b)) >= max(0.5, _threshold(b, md))

    grids = [] if test.get("ignore_markdown_tables") else _parse_md_tables(raw)
    grids += _parse_html_tables(raw)
    neighbors = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    for grid in grids:
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if _ratio(normalize_text(val), target) < max(0.5, _threshold(target, md)):
                    continue
                ok = True
                for key, (dr, dc) in neighbors.items():
                    want = test.get(key)
                    got = _cell(grid, r + dr, c + dc)
                    if want is not None and (got is None or not passes(str(want), got)):
                        ok = False
                        break
                for key, got in (("top_heading", _cell(grid, 0, c)),
                                 ("left_heading", _cell(grid, r, 0))):
                    want = test.get(key)
                    if ok and want is not None and (got is None or not passes(str(want), got)):
                        ok = False
                if ok:
                    return ScorerOutput(correct=True, score=1.0)
    return ScorerOutput(correct=False, score=0.0, miss_reason="table_mismatch")


# --- format --------------------------------------------------------------------
_FORMAT_PATTERNS: dict[str, list[str]] = {
    "heading": [r"(?m)^#{1,6}\s+(.+?)\s*$", r"<h[1-6][^>]*>(.*?)</h[1-6]>"],
    "bold": [r"\*\*(.*?)\*\*", r"__(.*?)__", r"<b[^>]*>(.*?)</b>", r"<strong[^>]*>(.*?)</strong>"],
    "italic": [r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)",
               r"<i[^>]*>(.*?)</i>", r"<em[^>]*>(.*?)</em>"],
}


def _format(test: dict[str, Any], raw: str) -> ScorerOutput:
    needle = normalize_text(str(test.get("text", "")))
    cs = test.get("case_sensitive", True)
    thr = _threshold(needle, int(test.get("max_diffs", 0)))
    n = needle if cs else needle.lower()
    for pat in _FORMAT_PATTERNS.get(str(test.get("format", "")), []):
        for span in re.findall(pat, raw, re.S):
            hay = normalize_text(span)
            hay = hay if cs else hay.lower()
            if _partial(n, hay) >= thr:
                return ScorerOutput(correct=True, score=1.0)
    return ScorerOutput(correct=False, score=0.0, miss_reason="format_absent")


# --- footnote ------------------------------------------------------------------
_SUPERSCRIPT = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def _footnote(test: dict[str, Any], raw: str) -> ScorerOutput:
    marker = str(test.get("marker", ""))
    md = int(test.get("max_diffs", 0))
    pos = -1
    for pat in (rf"\[\^{re.escape(marker)}\](?!:)", rf"<sup[^>]*>\s*{re.escape(marker)}\s*</sup>"):
        m = re.search(pat, raw)
        if m:
            pos = m.start()
            break
    if pos < 0 and marker.isdigit():
        sup = "".join(_SUPERSCRIPT[int(d)] for d in marker)
        pos = raw.find(sup) if sup and sup in raw else -1
    if pos < 0:
        return ScorerOutput(correct=False, score=0.0, miss_reason="footnote_absent")
    for key, window in (("appears_before_marker", raw[max(0, pos - 200):pos]),
                        ("appears_after_marker", raw[pos:pos + 200])):
        want = test.get(key)
        if not want:
            continue
        clean = re.sub(r"[^0-9a-zA-Z]", "", window).lower()
        wn = re.sub(r"[^0-9a-zA-Z]", "", normalize_text(str(want))).lower()
        if _partial(wn, clean) < _threshold(wn, md):
            return ScorerOutput(correct=False, score=0.0, miss_reason="footnote_context_miss")
    return ScorerOutput(correct=True, score=1.0)


# --- math (exact-match; rendered-equality fallback omitted — see module docstring)
_MATH_PATTERNS = [r"\\\((.+?)\\\)", r"\\\[(.+?)\\\]", r"\$\$(.+?)\$\$"]
_DOLLAR_PATTERN = r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)"


def _norm_math(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _math(test: dict[str, Any], raw: str) -> ScorerOutput:
    target = _norm_math(str(test.get("math", "")))
    if not target:
        return ScorerOutput(correct=None, miss_reason="no_target")
    patterns = list(_MATH_PATTERNS)
    if not test.get("ignore_dollar_delimited"):
        patterns.append(_DOLLAR_PATTERN)
    eqs = [e for pat in patterns for e in re.findall(pat, raw, re.S)]
    if not eqs:
        return ScorerOutput(correct=False, score=0.0, miss_reason="math_absent")
    for e in eqs:
        ne = _norm_math(e)
        if ne == target or _ratio(ne, target) >= 0.97:
            return ScorerOutput(correct=True, score=1.0,
                                rationale="exact-match (rendered-equality fallback omitted)")
    return ScorerOutput(correct=False, score=0.0, miss_reason="math_mismatch")


_EVALUATORS = {
    "present": lambda t, norm, raw: _presence(t, norm, want=True),
    "absent": lambda t, norm, raw: _presence(t, norm, want=False),
    "order": lambda t, norm, raw: _order(t, norm),
    "baseline": lambda t, norm, raw: _baseline(t, norm),
    "table": lambda t, norm, raw: _table(t, raw),
    "format": lambda t, norm, raw: _format(t, raw),
    "footnote": lambda t, norm, raw: _footnote(t, raw),
    "math": lambda t, norm, raw: _math(t, raw),
}


def score_test(test: dict[str, Any], text: str) -> ScorerOutput:
    """Score one olmOCR-bench test case against a page transcription.

    Content tests (present/absent/order/baseline) see the markdown-stripped
    `normalize_text`; structural tests (table/format/footnote/math) see the
    de-chattered raw markdown and do their own parsing/normalization.
    """
    ttype = str(test.get("type", "")).lower()
    evaluator = _EVALUATORS.get(ttype)
    if evaluator is None:
        return ScorerOutput(correct=None, miss_reason="unknown_test_type",
                            rationale=f"unrecognized test type {ttype!r}")
    raw = strip_chatter(text)
    return evaluator(test, normalize_text(raw), raw)
