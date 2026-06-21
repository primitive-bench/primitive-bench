"""Deterministic slice keys for the vectordb primitive (from dataset facts only).

Slices are the "one winner is a lie" surface: a recall@10 winner is published per
slice only when adapters are statistically separable there. The axes:

  * ``dataset:<name>``  — which corpus/embedding family (sift-1m, glove-100,
    cohere-wiki-v3-1024, openai3-1536, …). Different embedding geometries reorder
    engines.
  * ``metric:<euclidean|angular>`` — L2 vs cosine; engines tuned for one can lag the other.
  * ``dim:<low|mid|high>`` — dimensionality bucket; high-dim curse hits graphs and
    quantizers differently.
  * ``budget:<fast|accurate>`` — the operating point (low vs high ef/nprobe). This is
    the recall-vs-QPS tradeoff surface: at ``fast`` recall is lower but QPS high; at
    ``accurate`` recall is high but QPS low, and engines separate differently in each.

`budget` is config-derived (passed by the runner from the engine's param band), the
rest are dataset-derived. All keys are assigned at item time, never shuffled.
"""
from __future__ import annotations


def dim_bucket(dim: int) -> str:
    """Bucket dimensionality: low (<256), mid (256–1024), high (>1024)."""
    if dim < 256:
        return "low"
    if dim <= 1024:
        return "mid"
    return "high"


def slice_keys(dataset: str, metric: str, dim: int, budget: str) -> list[str]:
    """The frozen slice keys one (engine, dataset, config, query) item participates in."""
    return [
        f"dataset:{dataset}",
        f"metric:{metric}",
        f"dim:{dim_bucket(int(dim))}",
        f"budget:{budget}",
    ]
