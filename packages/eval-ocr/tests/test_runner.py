"""Resume + rate-limit-resilience tests for the OCR runner (keyless, in-memory).

Uses fake adapters (no network/binary) so these run anywhere, incl. CI.
"""
from __future__ import annotations

from pathlib import Path

from bench_adapters.ocr import RateLimited
from bench_adapters.registry import Adapter, register

from eval_ocr.runner import run_sync


@register("fake-ok")
class _FakeOK(Adapter):
    vendor = "fake"
    model_version = "fake-1"
    is_sentinel = False
    mode = "native_ocr"
    calls = 0

    def invoke(self, item):  # noqa: ANN001
        type(self).calls += 1
        text = "hello world benchmark report"
        return {"raw_output": text, "text": text, "latency_ms": 1.0, "cost_usd": 0.01,
                "mode": self.mode}


_FLAKY = {"fail_once": True}


@register("fake-flaky")
class _FakeFlaky(Adapter):
    vendor = "fake"
    model_version = "fake-1"
    is_sentinel = False
    mode = "native_ocr"

    def invoke(self, item):  # noqa: ANN001
        if _FLAKY["fail_once"]:
            _FLAKY["fail_once"] = False
            raise RateLimited("simulated 429")
        text = "hello world benchmark report"
        return {"raw_output": text, "text": text, "latency_ms": 1.0, "cost_usd": 0.0,
                "mode": self.mode}


def _rows():
    # 2 pages x 2 present-tests; all match the fake transcription
    rows = []
    for p in ("p1", "p2"):
        for i, frag in enumerate(("hello world", "benchmark report")):
            rows.append({"row_id": f"{p}-{i}", "page_image": f"{p}.png",
                         "type": "present", "text": frag, "doc_type": "arxiv"})
    return rows


def _lines(run_dir: Path) -> int:
    f = run_dir / "items.jsonl"
    return sum(1 for ln in f.read_text().splitlines() if ln.strip()) if f.exists() else 0


def test_resume_skips_done_no_repay(tmp_path):
    _FakeOK.calls = 0
    rows = _rows()
    # first run
    run_sync(rows, "r1", vendors=["fake-ok"], out_root=str(tmp_path))
    assert _FakeOK.calls == 2  # one invoke per unique page, not per test
    assert _lines(tmp_path / "r1") == 4

    # resume: everything already done -> no new invokes, no dupes
    run_sync(rows, "r1", vendors=["fake-ok"], out_root=str(tmp_path))
    assert _FakeOK.calls == 2  # unchanged — did not re-pay
    assert _lines(tmp_path / "r1") == 4


def test_partial_then_resume_completes(tmp_path):
    _FakeOK.calls = 0
    rows = _rows()
    # only the first page's tests exist -> 1 invoke, 2 items
    run_sync(rows[:2], "r2", vendors=["fake-ok"], out_root=str(tmp_path))
    assert _FakeOK.calls == 1 and _lines(tmp_path / "r2") == 2
    # resume with the full set -> only the missing page is invoked
    run_sync(rows, "r2", vendors=["fake-ok"], out_root=str(tmp_path))
    assert _FakeOK.calls == 2 and _lines(tmp_path / "r2") == 4


def test_rate_limited_page_not_recorded_then_retried(tmp_path):
    _FLAKY["fail_once"] = True
    rows = _rows()[:1]  # single test, single page
    # first run: RateLimited -> nothing checkpointed
    run_sync(rows, "r3", vendors=["fake-flaky"], out_root=str(tmp_path))
    assert _lines(tmp_path / "r3") == 0
    # resume: now succeeds and is recorded as a pass
    recs = run_sync(rows, "r3", vendors=["fake-flaky"], out_root=str(tmp_path))
    assert _lines(tmp_path / "r3") == 1
    flaky = [r for r in recs if r.adapter == "fake-flaky"]
    assert len(flaky) == 1 and flaky[0].output.correct is True


def test_budget_cap_stops_early(tmp_path):
    _FakeOK.calls = 0
    rows = _rows()  # 2 pages @ $0.01/page = $0.02 total
    run_sync(rows, "r4", vendors=["fake-ok"], out_root=str(tmp_path), max_cost_usd=0.005)
    # cap hit after the first page; second page left for resume
    assert _FakeOK.calls == 1 and _lines(tmp_path / "r4") == 2
