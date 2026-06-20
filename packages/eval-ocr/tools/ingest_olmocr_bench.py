"""Generate the OCR dev split locally from olmOCR-bench (NOT committed).

olmOCR-bench (AllenAI, ODC-BY-1.0) is redistributable as a *database*, but the
underlying PDF page *content* carries its own copyright — so we do NOT commit
rendered page images. This pulls olmOCR-bench to the user's machine, renders the
tested pages locally (pymupdf, pinned DPI), and emits a git-ignored `dev.jsonl` +
`images/olmocr/`. Only the tiny self-owned `dev.example.jsonl` is committed.

Deterministic by construction (no globbing / no string-munged filenames):
  * the corpus is the fixed set of category files in `CATEGORIES`;
  * each row's `pdf` field IS the canonical path under `bench_data/pdfs/`, so PDFs
    resolve exactly (a missing one is reported, not silently rglob'd);
  * rendered images mirror the source path (`images/olmocr/<category>/<stem>_pN.png`)
    — collision-free without name mangling;
  * `row_id` is the row's own olmOCR-bench `id`.

Run (downloads ~GB the first time):
    uv run python packages/eval-ocr/tools/ingest_olmocr_bench.py                 # everything
    uv run python packages/eval-ocr/tools/ingest_olmocr_bench.py --per-file 25   # balanced subset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_ocr._paths import CANARY, GOLDEN_OCR_DIR

RENDER_DPI = 200  # pinned — changing DPI changes OCR -> scores (folded into dataset_version)

# The fixed olmOCR-bench corpus: category test file -> our doc_type slice.
CATEGORIES = {
    "arxiv_math": "arxiv",
    "old_scans": "old_scan",
    "old_scans_math": "old_scan",
    "headers_footers": "old_scan",
    "multi_column": "multi_column",
    "long_tiny_text": "multi_column",
    "table_tests": "table",
}


def _render(pdf: Path, page: int, out: Path) -> None:
    import fitz  # pymupdf, lazy

    doc = fitz.open(pdf)
    idx = max(0, min(int(page) - 1, doc.page_count - 1))  # olmOCR pages are 1-indexed
    out.parent.mkdir(parents=True, exist_ok=True)
    doc[idx].get_pixmap(dpi=RENDER_DPI).save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-file", type=int, default=None,
                    help="take only the first N rows of each category (balanced subset)")
    ap.add_argument("--repo", default="allenai/olmOCR-bench")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"huggingface-hub required: {exc}") from exc

    snap = Path(snapshot_download(args.repo, repo_type="dataset",
                                  allow_patterns=["bench_data/**"]))
    pdf_root = snap / "bench_data" / "pdfs"
    print(f"snapshot: {snap}")

    rows_out: list[dict] = []
    missing: list[str] = []
    for stem, doc_type in CATEGORIES.items():  # fixed order
        jf = snap / "bench_data" / f"{stem}.jsonl"
        if not jf.exists():
            missing.append(jf.name)
            continue
        lines = [ln for ln in jf.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if args.per_file:
            lines = lines[: args.per_file]
        for ln in lines:
            t = json.loads(ln)
            if "type" not in t or not t.get("pdf"):
                continue
            pdf = pdf_root / t["pdf"]
            if not pdf.exists():
                missing.append(t["pdf"])
                continue
            page = int(t.get("page", 1))
            img_rel = f"images/olmocr/{t['pdf'][:-4]}_p{page}.png"  # mirror source path
            img_abs = GOLDEN_OCR_DIR / img_rel
            if not img_abs.exists():
                _render(pdf, page, img_abs)
            rows_out.append({
                **t,
                "row_id": t["id"],
                "page_image": img_rel,
                "doc_type": doc_type,
                "ground_truth_tier": "verified_external",
                "canary": CANARY,
            })

    out_file = GOLDEN_OCR_DIR / "dev.jsonl"  # git-ignored
    with out_file.open("w", encoding="utf-8") as f:
        f.write(f"# {CANARY}  (do not train on this file; see ../CANARY)\n")
        f.write("# Generated locally from olmOCR-bench (ODC-BY); page images are NOT redistributed.\n")
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    pages = len({r["page_image"] for r in rows_out})
    print(f"wrote {len(rows_out)} rows ({pages} unique pages) -> {out_file}  (DPI={RENDER_DPI})")
    if missing:
        print(f"WARNING: {len(missing)} missing PDFs/files (e.g. {missing[:3]})")


if __name__ == "__main__":
    main()
