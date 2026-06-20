# golden-sets-public / ocr

PUBLIC dev split for the OCR primitive. Canary-marked; reproducible by anyone.

**Held-out test answers NEVER live here** — they sit behind the private eval
server (primitivebench-platform/apps/eval-server) with HMAC-keyed split integrity.

## What ships here vs. generated locally

The OCR scorer uses the **olmOCR-bench** unit tests (pass@test). olmOCR-bench is
ODC-BY-1.0 (its *database* is redistributable), but the underlying PDF page
*content* (arXiv papers, scans) carries its own copyright — so **we do not
redistribute rendered page images**. Mirroring the websearch/extraction split
model:

- `dev.example.jsonl` — a tiny, **self-authored** illustrative set (pages we wrote
  and rendered ourselves, CC-BY), committed here. Enough to run the loop end-to-end
  with the keyless `tesseract` adapter.
- `images/example_*.png` — the committed example page images (self-authored).
- `dev.jsonl` + `images/olmocr/` — the **full** dev split, generated locally and
  **git-ignored** (never committed). Build it with:
  ```bash
  uv run python packages/eval-ocr/tools/ingest_olmocr_bench.py --limit 150
  ```
- `CANARY` — the canary GUID for contamination detection (BIG-bench convention),
  embedded as the header comment of each split file.

## Row schema (olmOCR-bench shape)

```jsonc
{ "row_id": "...", "page_image": "images/...png", "type": "present|absent|order|baseline|table|math",
  "text": "...", "before": "...", "after": "...", "max_diffs": 0, "case_sensitive": true,
  "doc_type": "arxiv|old_scan|table|multi_column|math",
  "ground_truth_tier": "verified_external", "canary": "PRIMITIVEBENCH-CANARY-GUID:..." }
```

Slices are assigned deterministically from `doc_type` and `type` (see
`packages/eval-ocr/slices.yaml`).

## Provenance

The full dev split is generated from
[allenai/olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench)
(**ODC-BY-1.0**); the pass-rule semantics are ported from
[allenai/olmocr](https://github.com/allenai/olmocr) (`olmocr/bench/tests.py`,
Apache-2.0). `dev.example.jsonl` and its images are original, authored by the
Primitive Bench team and released under CC-BY-4.0.
