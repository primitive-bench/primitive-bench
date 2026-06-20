"""eval-retrieval — Primitive Bench vertical.

Embeddings/retrieval — BEIR/MTEB-style per-domain slices

Implements bench_core.Task + bench_core.Scorer for this primitive. The public
golden DEV split lives in golden-sets-public/retrieval/; the held-out test split lives
only behind the private eval server. Slice definitions: slices.yaml.

STATUS: implemented. Cloned from eval-reranker — bi-encoder embedding retrievers
rank a per-query candidate pool; success@10 is the separability gate, nDCG@10/
recall@10/MAP/MRR ride alongside.
"""

from eval_retrieval.task import Task

__all__ = ["Task"]
