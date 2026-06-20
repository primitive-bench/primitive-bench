"""bench-adapters / rerank — reranker systems-under-test.

(query, candidate list) -> reordered candidate ids. Importing this subpackage
registers all rerank adapters via `@register` side effects:
  ce-minilm, bge-reranker (free local cross-encoders),
  voyage-rerank, jina-rerank, cohere-rerank (hosted APIs).
"""

from bench_adapters.rerank.adapters import VendorUnavailable

from bench_adapters.rerank import adapters as adapters  # noqa: F401  (registration side effects)

__all__ = ["VendorUnavailable"]
