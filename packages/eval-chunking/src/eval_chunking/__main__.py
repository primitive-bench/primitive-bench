"""CLI: run the chunking probe over a golden split.

    # keyless, fully-offline smoke test (committed example corpora + tfidf-local):
    uv run python -m eval_chunking \
        --golden golden-sets-public/chunking/dev.example.jsonl --run-id chunking-smoke

    # full Chroma subset (generate it first with tools/ingest_chroma_chunking.py):
    uv run python -m eval_chunking \
        --golden golden-sets-public/chunking/dev.jsonl --run-id chunking-dev

    # publish-grade run with a SOTA dense embedder (needs the key / a download):
    uv run python -m eval_chunking --golden golden-sets-public/chunking/dev.jsonl \
        --embedder voyage --run-id chunking-voyage
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from eval_chunking.loader import load_golden
from eval_chunking.runner import DEFAULT_CHUNKERS, run_sync
from eval_chunking.scoring import RETRIEVE_K


def _load_env() -> None:
    """Load the nearest `.env` (CWD upward) into os.environ for API keys.

    No hard dotenv dep — a minimal KEY=VALUE parser. Existing env vars win
    (`setdefault`). Walking up means a run from the worktree finds the repo-root
    `.env`, so embedder keys 'just work' without exporting anything."""
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
    ap.add_argument("--golden", required=True, help="path to a golden JSONL split")
    ap.add_argument("--corpora", default=None,
                    help="corpora dir (default: <golden dir>/corpora)")
    ap.add_argument("--chunkers", default=None,
                    help="comma-separated chunker names (default: all registered)")
    ap.add_argument("--embedder", default="tfidf-local",
                    help="retrieval embedder: tfidf-local (default) | openai | voyage | "
                         "cohere | jina | st:<model> | alias (bge-large, arctic-l)")
    ap.add_argument("--k", type=int, default=RETRIEVE_K, help="top-k chunks retrieved")
    ap.add_argument("--run-id", default="chunking-dev")
    ap.add_argument("--limit", type=int, default=None, help="cap queries (debug)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()

    rows, corpora = load_golden(args.golden, args.corpora)
    chunkers = (tuple(c.strip() for c in args.chunkers.split(",") if c.strip())
                if args.chunkers else DEFAULT_CHUNKERS)
    run_sync(rows, corpora, args.run_id, chunkers=chunkers, embedder=args.embedder,
             k=args.k, seed=args.seed, limit=args.limit, out_root=args.out_root)


if __name__ == "__main__":
    main()
