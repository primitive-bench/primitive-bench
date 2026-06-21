"""bench-adapters / chunk — chunking strategies as systems-under-test.

A document -> a list of character-span chunks. Importing this subpackage registers
all chunk adapters via ``@register`` side effects:
  fixed-token (sentinel), recursive, sentence, semantic, cluster-semantic.

``semantic`` and ``cluster-semantic`` consume a shared embedder injected by the eval
as ``item["embed"]`` so the embedder is held constant across chunkers.
"""

from bench_adapters.chunk.adapters import VendorUnavailable

from bench_adapters.chunk import adapters as adapters  # noqa: F401  (registration side effects)

__all__ = ["VendorUnavailable"]
