"""Runner + report tests using deterministic fake chunkers (no network/model).

Two fake chunkers make the outcome independent of the embedder: ``_test-whole``
returns the entire document as one chunk (the reference is always recovered ->
coverage hit), ``_test-miss`` returns only the first 10 characters (the reference,
placed past char 10, is never recovered -> miss). This exercises the full path:
run_sync -> chunk -> embed (tfidf-local) -> retrieve -> score -> report -> counts.toml.
"""
from __future__ import annotations

import tomllib
from typing import Any

from bench_adapters import register, registry
from bench_adapters.registry import Adapter

from eval_chunking import report
from eval_chunking.runner import run_sync

CORPUS = {"c1": "HEADER ZONE. The capital of France is Paris, a large European city. The end."}

# References start at char 13 (after "HEADER ZONE. "), so the 10-char miss chunker
# can never recover them.
ROWS = [
    {"row_id": "r1", "corpus_id": "c1", "question": "What is the capital of France?",
     "references": [{"start_index": 13, "end_index": 43}],
     "slices": ["domain:c1", "reference_dispersion:single"]},
    {"row_id": "r2", "corpus_id": "c1", "question": "Describe Paris.",
     "references": [{"start_index": 13, "end_index": 66}],
     "slices": ["domain:c1", "reference_dispersion:single"]},
]


class _Whole(Adapter):
    is_sentinel = False

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        doc = item["document"]
        return {"chunks": [{"text": doc, "start": 0, "end": len(doc)}], "n_chunks": 1,
                "latency_ms": 0.0, "cost_usd": 0.0}


class _Miss(Adapter):
    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        doc = item["document"]
        return {"chunks": [{"text": doc[:10], "start": 0, "end": 10}], "n_chunks": 1,
                "latency_ms": 0.0, "cost_usd": 0.0}


for _name, _cls in (("_test-whole", _Whole), ("_test-miss", _Miss)):
    if _name not in registry:
        register(_name)(_cls)


def test_run_sync_whole_beats_miss(tmp_path):
    recs = run_sync(ROWS, CORPUS, "test-run", chunkers=["_test-whole", "_test-miss"],
                    out_root=str(tmp_path), write_snapshot=False)
    by_adapter: dict[str, list[bool]] = {}
    for r in recs:
        by_adapter.setdefault(r.adapter, []).append(r.output.correct)
    assert all(by_adapter["_test-whole"]) is True          # reference always recovered
    assert not any(by_adapter["_test-miss"])               # reference never recovered
    assert (tmp_path / "test-run" / "items.jsonl").exists()
    assert (tmp_path / "test-run" / "slices.jsonl").exists()
    assert (tmp_path / "test-run" / "manifest.json").exists()


def test_collect_counts_shape(tmp_path):
    recs = run_sync(ROWS, CORPUS, "test-run2", chunkers=["_test-whole", "_test-miss"],
                    out_root=str(tmp_path), write_snapshot=False)
    counts = report.collect_counts(recs)
    overall = dict((a, (k, n)) for a, k, n in counts["all"])
    assert overall["_test-whole"] == (2, 2)                # 2/2 coverage hits
    assert overall["_test-miss"] == (0, 2)                 # 0/2
    assert "domain:c1" in counts


def test_counts_toml_is_parseable(tmp_path):
    recs = run_sync(ROWS, CORPUS, "test-run3", chunkers=["_test-whole"],
                    out_root=str(tmp_path), write_snapshot=False)
    out = tmp_path / "chunking.counts.toml"
    report.write_counts_toml(recs, out, run_id="test")
    spec = tomllib.loads(out.read_text())
    assert spec["primitive"] == "chunking"
    assert spec["metric_name"] == "coverage_at_5"
    assert spec["slices"]["domain:c1"][0][0] == "_test-whole"
