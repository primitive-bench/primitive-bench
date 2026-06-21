"""bench-adapters / vectordb — vector DB / ANN engines as systems-under-test.

Unlike the other primitives' stateless `invoke(item)` adapters, a vector engine
**build()s** an index over the dataset's base vectors once, then **query()s** it many
times. Importing this subpackage registers every engine via `@register` side effects:

  bruteforce-numpy, hnswlib, faiss-flat, faiss-hnsw, faiss-ivf, annoy, qdrant-local,
  lancedb                                         (OSS in-process, keyless)
  pgvector, milvus, weaviate, elasticsearch       (Dockerized servers)
  pinecone, zilliz-cloud, weaviate-cloud          (hosted clouds, publish_restricted)

See `engines.py` for the lifecycle and `pricing.py` for the hosted-query cost table.
"""

from bench_adapters.vectordb.engines import VectorEngine, VendorUnavailable

from bench_adapters.vectordb import engines as engines  # noqa: F401  (registration side effects)

__all__ = ["VectorEngine", "VendorUnavailable"]
