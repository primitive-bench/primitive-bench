"""Runner + report tests using deterministic fake engines on the synthetic dataset.

`_vec-oracle` returns exact neighbors (recall=1.0); `_vec-half` returns the exact
top-(k/2) and out-of-range filler (recall=0.5). This exercises the full path:
run_sync -> ItemResults -> report.run/collect_counts -> counts.toml — and proves the
fractional-recall -> integer (Σhits, ΣK) counts aggregation.
"""
from __future__ import annotations

import tomllib

from bench_adapters import register, registry
from bench_adapters.vectordb.engines import BruteForceNumpy, VectorEngine

from eval_vectordb import report
from eval_vectordb.runner import run_sync


class _Oracle(VectorEngine):
    name = "_vec-oracle"
    vendor = "test"

    def build(self, base, metric, params):
        self._bf = BruteForceNumpy()
        self._bf.build(base, metric, params)

    def query(self, vector, k):
        return self._bf.query(vector, k)


class _Half(VectorEngine):
    name = "_vec-half"
    vendor = "test"

    def build(self, base, metric, params):
        self._bf = BruteForceNumpy()
        self._bf.build(base, metric, params)

    def query(self, vector, k):
        keep = self._bf.query(vector, k)[: k // 2]      # exact top-(k/2) -> all hits
        filler = [10_000_000 + i for i in range(k - len(keep))]  # out-of-range -> misses
        return keep + filler


for _name, _cls in (("_vec-oracle", _Oracle), ("_vec-half", _Half)):
    if _name not in registry:
        register(_name)(_cls)


def test_oracle_perfect_half_recall(tmp_path):
    recs = run_sync(["synthetic"], "t1", engines=["_vec-oracle", "_vec-half"],
                    out_root=str(tmp_path), write_snapshot=False)
    counts = report.collect_counts(recs)
    overall = {a: (k, n) for a, k, n in counts["all"]}
    ko, no = overall["_vec-oracle"]
    kh, nh = overall["_vec-half"]
    assert ko == no                       # exact recall = 1.0 (Σhits == ΣK)
    assert abs(kh / nh - 0.5) < 1e-9      # half recall = 0.5
    # run dir artifacts written
    assert (tmp_path / "t1" / "items.jsonl").exists()
    assert (tmp_path / "t1" / "slices.jsonl").exists()
    assert (tmp_path / "t1" / "manifest.json").exists()


def test_counts_have_budget_and_dataset_slices(tmp_path):
    recs = run_sync(["synthetic"], "t2", engines=["_vec-oracle", "_vec-half"],
                    out_root=str(tmp_path), write_snapshot=False)
    counts = report.collect_counts(recs)
    assert "dataset:synthetic" in counts
    assert "budget:accurate" in counts
    assert "metric:euclidean" in counts
    # every (adapter, hits, n) row is integer with hits <= n (valid Wilson input)
    for rows in counts.values():
        for _a, k, n in rows:
            assert isinstance(k, int) and isinstance(n, int) and 0 <= k <= n


def test_counts_toml_is_parseable(tmp_path):
    recs = run_sync(["synthetic"], "t3", engines=["_vec-oracle"],
                    out_root=str(tmp_path), write_snapshot=False)
    out = tmp_path / "vectordb.counts.toml"
    report.write_counts_toml(recs, out, run_id="test")
    spec = tomllib.loads(out.read_text())
    assert spec["primitive"] == "vectordb"
    assert spec["metric_name"] == "recall_at_10"
    assert spec["slices"]["dataset:synthetic"][0][0] == "_vec-oracle"


def test_separability_emerges_between_oracle_and_half(tmp_path):
    recs = run_sync(["synthetic"], "t4", engines=["_vec-oracle", "_vec-half"],
                    out_root=str(tmp_path), write_snapshot=False)
    slice_results, board = report.run(recs)
    overall = next(b for b in board if b["slice"] == "all")
    # 1.0 vs 0.5 over hundreds of neighbor-slots -> oracle wins, separable.
    assert overall["winner"] == "_vec-oracle"
