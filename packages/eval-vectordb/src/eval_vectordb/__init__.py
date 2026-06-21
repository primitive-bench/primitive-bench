"""eval-vectordb — Primitive Bench vertical (vector DB / ANN: recall@10 vs QPS / cost).

Build each engine's index over a dataset's base vectors, query it for every test
vector, and score **recall@10 against the exact top-K nearest neighbors** (computed by
brute force → `GroundTruthTier.VERIFIED_EXTERNAL`). recall@10 is a proportion
(Σhits/ΣK), so the McNemar/Wilson leaderboard gate consumes it directly — per-slice
winners (or TIE bands), never one global ranking. The recall-vs-QPS-vs-cost story
(ANN-Benchmarks + VectorDBBench) rides alongside as continuous companions.

Implements `bench_core.Task` + `bench_core.Scorer`. Engines live in
`bench_adapters.vectordb` (OSS in-process + Dockerized servers + hosted clouds);
datasets in `eval_vectordb.datasets` (ANN-Benchmarks HDF5 / HuggingFace / synthetic).
Slices: slices.yaml (dataset / metric / dim / budget).
"""

from eval_vectordb.task import Task
from eval_vectordb.scoring import score_query
from eval_vectordb.runner import run_sync

__all__ = ["Task", "score_query", "run_sync"]
