"""CLI: build vector engines over ANN datasets and score recall@10 vs QPS / latency / cost.

    # keyless offline smoke (synthetic dataset, OSS in-process engines):
    uv run python -m eval_vectordb --datasets synthetic --run-id vectordb-smoke

    # medium real run (ANN-Benchmarks + modern embeddings; engines auto-skip if absent):
    uv run python -m eval_vectordb \
        --datasets sift-128-euclidean,glove-100-angular,gist-960-euclidean,cohere-wiki-v3-1024,openai3-1536 \
        --engines faiss-hnsw,faiss-ivf,hnswlib,annoy,qdrant-local,lancedb,pgvector,milvus \
        --run-id vectordb-dev

Engines requiring a service (pgvector/milvus/weaviate/elasticsearch) or a key
(pinecone/zilliz-cloud/weaviate-cloud) skip cleanly when unavailable.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from eval_vectordb import datasets as datasets_mod
from eval_vectordb.runner import DEFAULT_ENGINES, run_sync


def _load_env() -> None:
    """Load the nearest `.env` (CWD upward) into os.environ for service/API keys.

    No hard dotenv dep — a minimal KEY=VALUE parser. Existing env vars win
    (`setdefault`)."""
    for d in (Path.cwd(), *Path.cwd().parents):
        env = d / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return


def main() -> None:
    _load_env()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="synthetic",
                    help="comma-separated dataset names (synthetic / sift-128-euclidean / cohere-wiki-v3-1024 / ...)")
    ap.add_argument("--engines", default=None,
                    help="comma-separated engine names (default: OSS in-process set)")
    ap.add_argument("--run-id", default="vectordb-dev")
    ap.add_argument("--limit", type=int, default=None, help="cap queries per dataset (debug)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-limit", type=int, default=datasets_mod.DEFAULT_BASE_LIMIT,
                    help="subsample base vectors to this many (medium scale)")
    ap.add_argument("--query-limit", type=int, default=datasets_mod.DEFAULT_QUERY_LIMIT,
                    help="subsample queries to this many")
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()

    datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    engines = (tuple(e.strip() for e in args.engines.split(",") if e.strip())
               if args.engines else DEFAULT_ENGINES)
    run_sync(datasets, args.run_id, engines=engines, seed=args.seed, limit=args.limit,
             base_limit=args.base_limit, query_limit=args.query_limit, out_root=args.out_root)


if __name__ == "__main__":
    main()
