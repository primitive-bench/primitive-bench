"""Runner + report tests using deterministic fake rerankers (no network/model).

Two oracles read the row's own relevance (real adapters never do) so the run is
fully deterministic: `_test-oracle` ranks relevant-first (always hit@1), `_test-worst`
ranks relevant-last (never hit@1). This exercises the full path:
run_sync -> ItemResults -> report.run/collect_counts -> counts.toml.
"""
from __future__ import annotations

import tomllib
from typing import Any

from bench_adapters import register, registry
from bench_adapters.registry import Adapter

from eval_reranker import report
from eval_reranker.runner import run_sync

ROWS = [
    {"row_id": "r1", "domain": "scientific", "query": "q1",
     "candidates": [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}, {"id": "c", "text": "z"}],
     "relevance": {"a": 0, "b": 1, "c": 0}, "n_relevant": 1,
     "slices": ["domain:scientific", "hard_negative_density:low"]},
    {"row_id": "r2", "domain": "tech_qa", "query": "q2",
     "candidates": [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}, {"id": "c", "text": "z"}],
     "relevance": {"a": 1, "b": 0, "c": 0}, "n_relevant": 1,
     "slices": ["domain:tech_qa", "hard_negative_density:low"]},
]


def _order_by_relevance(item: dict[str, Any], *, best_first: bool) -> dict[str, Any]:
    rel = item.get("relevance", {})
    ids = [c["id"] for c in item["candidates"]]
    sign = -1.0 if best_first else 1.0
    order = sorted(ids, key=lambda i: sign * float(rel.get(str(i), 0)))
    return {"reordered_ids": order, "latency_ms": 0.0, "cost_usd": 0.0, "raw_output": ",".join(order)}


class _Oracle(Adapter):
    is_sentinel = False

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        return _order_by_relevance(item, best_first=True)


class _Worst(Adapter):
    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        return _order_by_relevance(item, best_first=False)


# Register once (pytest may import the module more than once in a session).
for _name, _cls in (("_test-oracle", _Oracle), ("_test-worst", _Worst)):
    if _name not in registry:
        register(_name)(_cls)


def test_run_sync_oracle_beats_worst(tmp_path):
    recs = run_sync(ROWS, "test-run", vendors=["_test-oracle", "_test-worst"],
                    out_root=str(tmp_path), write_snapshot=False)
    by_adapter: dict[str, list[bool]] = {}
    for r in recs:
        by_adapter.setdefault(r.adapter, []).append(r.output.correct)
    assert all(by_adapter["_test-oracle"]) is True          # always hit@1
    assert not any(by_adapter["_test-worst"])               # never hit@1
    # run dir artifacts written
    assert (tmp_path / "test-run" / "items.jsonl").exists()
    assert (tmp_path / "test-run" / "slices.jsonl").exists()
    assert (tmp_path / "test-run" / "manifest.json").exists()


def test_collect_counts_shape(tmp_path):
    recs = run_sync(ROWS, "test-run2", vendors=["_test-oracle", "_test-worst"],
                    out_root=str(tmp_path), write_snapshot=False)
    counts = report.collect_counts(recs)
    overall = dict((a, (k, n)) for a, k, n in counts["all"])
    assert overall["_test-oracle"] == (2, 2)               # 2/2 hit@1
    assert overall["_test-worst"] == (0, 2)                # 0/2 hit@1
    # each domain slice present with one item
    assert "domain:scientific" in counts and "domain:tech_qa" in counts


def test_counts_toml_is_parseable(tmp_path):
    recs = run_sync(ROWS, "test-run3", vendors=["_test-oracle"],
                    out_root=str(tmp_path), write_snapshot=False)
    out = tmp_path / "reranker.counts.toml"
    report.write_counts_toml(recs, out, run_id="test")
    spec = tomllib.loads(out.read_text())
    assert spec["primitive"] == "reranker"
    assert spec["metric_name"] == "hit_at_1"
    assert spec["slices"]["domain:scientific"][0][0] == "_test-oracle"
