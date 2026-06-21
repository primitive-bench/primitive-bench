"""eval-chunking — Primitive Bench vertical (downstream retrieval over chunk strategies).

Chunking is judged by its DOWNSTREAM effect: chunk a corpus, embed the chunks with a
*fixed* embedder, retrieve top-k for each query, and score how well the retrieved
chunks recover the query's gold reference spans (Chroma "Evaluating Chunking
Strategies for Retrieval" methodology — recall / precision / IoU / precision_Ω on
character ranges). The leaderboard separability surface is **coverage@k** (recall@k ≥
target) — a paired-binary outcome the McNemar/Wilson gate consumes — so per-slice
winners (or TIE bands), never one global ranking.

Implements `bench_core.Task` + `bench_core.Scorer`. The public DEV split lives in
golden-sets-public/chunking/ (a tiny committed example + a local Chroma generator);
the held-out test split lives only behind the private eval server. Slices:
slices.yaml (domain / reference_dispersion). The chunkers themselves are bench_adapters
chunk adapters (fixed-token, recursive, sentence, semantic, cluster-semantic).
"""

from eval_chunking.task import Task
from eval_chunking.scoring import score_chunking
from eval_chunking.runner import run_sync

__all__ = ["Task", "score_chunking", "run_sync"]
