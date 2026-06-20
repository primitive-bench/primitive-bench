"""Runner + report tests using deterministic fake retrievers (no network/model).

Two oracles read the row's own relevance (real adapters never do) so the run is fully
deterministic: `_test-ret-oracle` ranks relevant-first (relevant doc lands at rank 1,
success@10 true), `_test-ret-worst` ranks relevant-last (rank 12, outside top-10,
success@10 false). This exercises the full path:
run_sync -> ItemResults -> report.run/collect_counts -> counts.toml.
"""
from __future__ import annotations

import tomllib
from typing import Any

from bench_adapters import register, registry
from bench_adapters.registry import Adapter

from eval_retrieval import report
from eval_retrieval.runner import run_sync


def _row(row_id: str, domain: str) -> dict[str, Any]:
    # 12-candidate pool so a demoted relevant doc can fall outside the top-10.
    cands = [{"id": f"d{i}", "text": f"t{i}"} for i in range(12)]
    rel = {f"d{i}": (1 if i == 5 else 0) for i in range(12)}
    return {"row_id": row_id, "domain": domain, "query": "q",
            "candidates": cands, "relevance": rel, "n_relevant": 1,
            "slices": [f"domain:{domain}", "relevant_set:single"]}


ROWS = [_row("r1", "scientific"), _row("r2", "medical")]


def _order_by_relevance(item: dict[str, Any], *, best_first: bool) -> dict[str, Any]:
    rel = item.get("relevance", {})
    ids = [c["id"] for c in item["candidates"]]
    sign = -1.0 if best_first else 1.0
    order = sorted(ids, key=lambda i: sign * float(rel.get(str(i), 0)))
    return {"retrieved_ids": order, "latency_ms": 0.0, "cost_usd": 0.0,
            "raw_output": ",".join(order)}


class _Oracle(Adapter):
    is_sentinel = False

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        return _order_by_relevance(item, best_first=True)


class _Worst(Adapter):
    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:
        return _order_by_relevance(item, best_first=False)


# Register once (pytest may import the module more than once in a session).
for _name, _cls in (("_test-ret-oracle", _Oracle), ("_test-ret-worst", _Worst)):
    if _name not in registry:
        register(_name)(_cls)


def test_run_sync_oracle_beats_worst(tmp_path):
    recs = run_sync(ROWS, "test-run", vendors=["_test-ret-oracle", "_test-ret-worst"],
                    out_root=str(tmp_path), write_snapshot=False)
    by_adapter: dict[str, list[bool]] = {}
    for r in recs:
        by_adapter.setdefault(r.adapter, []).append(r.output.correct)
    assert all(by_adapter["_test-ret-oracle"]) is True       # relevant always in top-10
    assert not any(by_adapter["_test-ret-worst"])            # relevant always demoted past 10
    assert (tmp_path / "test-run" / "items.jsonl").exists()
    assert (tmp_path / "test-run" / "slices.jsonl").exists()
    assert (tmp_path / "test-run" / "manifest.json").exists()


def test_collect_counts_shape(tmp_path):
    recs = run_sync(ROWS, "test-run2", vendors=["_test-ret-oracle", "_test-ret-worst"],
                    out_root=str(tmp_path), write_snapshot=False)
    counts = report.collect_counts(recs)
    overall = dict((a, (k, n)) for a, k, n in counts["all"])
    assert overall["_test-ret-oracle"] == (2, 2)             # 2/2 success@10
    assert overall["_test-ret-worst"] == (0, 2)              # 0/2 success@10
    assert "domain:scientific" in counts and "domain:medical" in counts


def test_counts_toml_is_parseable(tmp_path):
    recs = run_sync(ROWS, "test-run3", vendors=["_test-ret-oracle"],
                    out_root=str(tmp_path), write_snapshot=False)
    out = tmp_path / "retrieval.counts.toml"
    report.write_counts_toml(recs, out, run_id="test")
    spec = tomllib.loads(out.read_text())
    assert spec["primitive"] == "retrieval"
    assert spec["metric_name"] == "success_at_10"
    assert spec["slices"]["domain:scientific"][0][0] == "_test-ret-oracle"
