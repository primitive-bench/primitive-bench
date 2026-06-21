"""CLI: run the crawl probe over a golden split.

    # keyless reproducible run (offline controlled graphs; regenerates the snapshot):
    uv run python -m eval_crawl \
        --golden golden-sets-public/crawl/dev.example.jsonl --run-id crawl-public-snapshot

    # live run incl. hosted vendors (APIs skipped if their key is unset):
    uv run python -m eval_crawl \
        --golden golden-sets-public/crawl/dev.jsonl \
        --adapters bfs-deep,firecrawl-crawl,tavily-crawl,spider-crawl,apify-crawl \
        --run-id crawl-dev
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from eval_crawl.loader import load_rows
from eval_crawl.runner import DEFAULT_VENDORS, run_sync


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
                    help="comma-separated adapter names (default: keyless local strategies)")
    ap.add_argument("--run-id", default="crawl-dev")
    ap.add_argument("--limit", type=int, default=None, help="cap targets (debug)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--no-snapshot", action="store_true",
                    help="do not (re)write snapshots/crawl.counts.toml")
    args = ap.parse_args()

    rows = load_rows(args.golden)
    vendors = (tuple(a.strip() for a in args.adapters.split(",") if a.strip())
               if args.adapters else DEFAULT_VENDORS)
    run_sync(rows, args.run_id, vendors=vendors, seed=args.seed,
             limit=args.limit, out_root=args.out_root, write_snapshot=not args.no_snapshot)


if __name__ == "__main__":
    main()
