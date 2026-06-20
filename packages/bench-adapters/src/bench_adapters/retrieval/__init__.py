"""bench-adapters / retrieval — first-stage retrieval systems-under-test.

(query, candidate pool) -> ranked candidate ids, by bi-encoder cosine. Importing this
subpackage registers all retrieval adapters via `@register` side effects:
  bge-small, e5-small (free local bi-encoders),
  openai-embed, openai-embed-small, voyage-embed, cohere-embed (hosted APIs).
"""

from bench_adapters.retrieval.adapters import VendorUnavailable

from bench_adapters.retrieval import adapters as adapters  # noqa: F401  (registration side effects)

__all__ = ["VendorUnavailable"]
