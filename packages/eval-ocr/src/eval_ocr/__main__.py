"""`python -m eval_ocr` (or the `eval-ocr` console script) — run the OCR eval.

Runs the resumable pass@test probe over a golden JSONL split. Re-running with the
same --run-id resumes (finished work is skipped, never re-paid).

Examples:
    # keyless, local tesseract only:
    python -m eval_ocr golden-sets-public/ocr/dev.example.jsonl --vendors tesseract
    # full six-adapter run with a spend cap (needs keys in .env):
    python -m eval_ocr golden-sets-public/ocr/dev.jsonl --run-id full --max-cost 40
"""
from __future__ import annotations

import argparse

import bench_core.config  # noqa: F401  — loads .env so adapter API keys are visible
from eval_ocr.loader import load_rows
from eval_ocr.runner import DEFAULT_VENDORS, run_sync


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="eval-ocr", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dataset", help="path to a golden JSONL split")
    ap.add_argument("--run-id", default="ocr-run", help="resume key + run-dir name")
    ap.add_argument("--vendors", default=",".join(DEFAULT_VENDORS),
                    help="comma-separated adapter names")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of test cases")
    ap.add_argument("--reps", type=int, default=1, help="repeats per page (majority vote)")
    ap.add_argument("--rpm", type=float, default=None, help="pace paid lanes to N requests/min")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="stop cleanly at this USD spend (resumable)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args(argv)

    rows = load_rows(args.dataset)
    vendors = [v.strip() for v in args.vendors.split(",") if v.strip()]
    run_sync(rows, args.run_id, vendors=vendors, seed=args.seed, limit=args.limit,
             reps=args.reps, out_root=args.out_root, rpm=args.rpm, max_cost_usd=args.max_cost)


if __name__ == "__main__":
    main()
