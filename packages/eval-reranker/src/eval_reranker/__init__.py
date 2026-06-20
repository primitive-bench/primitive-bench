"""eval-reranker — Primitive Bench vertical (reranking over MTEB reranking sets).

Hand a reranker a query + a fixed candidate list, get back the reordering, and
score it with the BEIR/MTEB-standard IR metrics (nDCG@10/MAP/MRR). The leaderboard
separability surface is **hit@1** — a paired-binary outcome the McNemar/Wilson gate
consumes — so per-slice winners (or TIE bands), never one global ranking.

Implements `bench_core.Task` + `bench_core.Scorer`. The public DEV split lives in
golden-sets-public/reranker/ (a small committed example + a local MTEB generator);
the held-out test split lives only behind the private eval server. Slices:
slices.yaml (domain / candidate_depth / hard_negative_density).
"""

from eval_reranker.task import Task
from eval_reranker.scoring import score_rerank
from eval_reranker.runner import run_sync

__all__ = ["Task", "score_rerank", "run_sync"]
