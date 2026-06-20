"""CLI: run the retrieval probe over a golden split.

    # keyless smoke test (free local bi-encoder only):
    uv run python -m eval_retrieval \
        --golden golden-sets-public/retrieval/dev.example.jsonl \
        --adapters bge-small --run-id retrieval-smoke

    # full subset run (all registered adapters; APIs skipped if their key is unset):
    uv run python -m eval_retrieval \
        --golden golden-sets-public/retrieval/dev.jsonl --run-id retrieval-dev
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from eval_retrieval.loader import load_rows
from eval_retrieval.runner import DEFAULT_VENDORS, run_sync


def _load_env() -> None:
    """Load the nearest `.env` (CWD upward) into os.environ for API keys.

    No hard dotenv dep — a minimal KEY=VALUE parser. Existing env vars win
    (`setdefault`). Walking up means a run from the worktree finds the repo-root
    `.env`, so adapter keys 'just work' without exporting anything."""
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
    ap.add_argument("--adapters", default=None,
                    help="comma-separated adapter names (default: all registered)")
    ap.add_argument("--run-id", default="retrieval-dev")
    ap.add_argument("--limit", type=int, default=None, help="cap items (debug)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()

    rows = load_rows(args.golden)
    vendors = (tuple(a.strip() for a in args.adapters.split(",") if a.strip())
               if args.adapters else DEFAULT_VENDORS)
    run_sync(rows, args.run_id, vendors=vendors, seed=args.seed,
             limit=args.limit, out_root=args.out_root)


if __name__ == "__main__":
    main()
