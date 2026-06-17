# benchpublic

> Vendor-neutral, MIT-licensed, reproducible eval harness for **AI infrastructure
> primitives** — OCR, web search, vector DBs, rerankers, retrieval, extraction,
> chunking, crawling, memory.

This is the public trust anchor for Primitive Bench. Everything here is
reproducible by anyone: the harness engine, the statistics library, the adapter
SDK, the per-primitive eval packages, and the **public dev splits**. Held-out
golden answers never live in this repo — they sit behind the private eval server.

**Thesis:** "One winner is a lie." No primitive wins every slice. We publish
per-slice, per-constraint results with confidence intervals and statistical
separability — never a single global ranking.

## Layout

```
packages/
  bench-schemas/   # THE FROZEN CONTRACT — RunManifest, ItemResult, SliceResult, ScorerOutput, AdapterSpec
  bench-core/      # harness engine: deterministic seeding, run/manifest, per-run dirs (ann-benchmarks pattern)
  bench-stats/     # McNemar, Wilson, bootstrap CIs, hit@k, nDCG/MAP/MRR, Bradley-Terry
  bench-adapters/  # provider/primitive adapter SDK (lm-eval registry pattern)
  eval-*/          # one package per primitive: public golden dev set + scorer + slice defs
apps/
  cli/             # `benchpublic` CLI: init / run / view / submit
  docs/            # methodology v3 + DECISIONS.md
golden-sets-public/  # PUBLIC dev splits only (canary-marked). Held-out answers NEVER here.
```

## The Gate

`bench-schemas` is frozen at `v0.1.0`. Every package imports types **only** from
it and writes **only** files it owns — no shared mutable state. That boundary is
what lets the build lanes run in parallel without colliding. See
[`apps/docs/DECISIONS.md`](apps/docs/DECISIONS.md) (D-03).

## Quickstart (once packages land)

```bash
uv sync
uv run benchpublic run --primitive ocr --config configs/ocr.yaml
uv run benchpublic view ./runs/<run_id>
```

## License

MIT. Reuse freely. We learn from EleutherAI lm-evaluation-harness, UK AISI
Inspect, Stanford HELM, ann-benchmarks, VectorDBBench, and OmniDocBench — all
MIT/Apache-2.0. We do **not** vendor GPL/commercial-dual code (e.g. marker core).
