"""bench-adapters — provider/primitive adapter SDK (lm-eval registry pattern).

Adapters wrap a system-under-test behind a uniform `invoke(item) -> dict`. They
are registered by name so configs/CLI can reference them as strings.

OCR reference adapters (lane B first deliverable):
  claude-sonnet-ocr, gemini-ocr, gpt5-ocr, mistral-ocr, deepseek-ocr,
  tesseract (regression sentinel — expected stable, not to win).

Register a new adapter:

    from bench_adapters import register, Adapter

    @register("qdrant")
    class QdrantAdapter(Adapter):
        ...

See DECISIONS.md D-06.
"""

from bench_adapters.registry import Adapter, get, register, registry

# Import the primitive subpackages for their side effects: each module decorates
# its adapter classes with @register("name"), so importing them populates the
# registry. Done here so that `import bench_adapters` auto-registers everything.
from bench_adapters import extract as extract  # noqa: E402,F401
from bench_adapters import search as search  # noqa: E402,F401
from bench_adapters import ocr as ocr  # noqa: E402,F401

__all__ = ["Adapter", "register", "get", "registry", "search", "extract", "ocr"]
