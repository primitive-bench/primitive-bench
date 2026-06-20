<p align="center">
  <img src="docs/assets/primitive-bench-logo.svg" alt="Primitive Bench" width="110">
</p>

<h1 align="center">Primitive Bench</h1>

<p align="center">
  <b>The vendor-neutral benchmark for AI infrastructure primitives.</b><br>
  OCR · web search · vector DBs · rerankers · retrieval · extraction · chunking · crawling · memory
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="CONTRIBUTING.md"><img alt="PRs welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
</p>

<p align="center">
  <a href="https://www.primitivebench.com/">Website</a> ·
  <a href="apps/docs">Docs</a> ·
  <a href="apps/docs/methodology/v3.md">Methodology</a> ·
  <a href="CONTRIBUTING.md">Contributing</a> ·
  <a href="https://www.linkedin.com/company/primitive-bench">LinkedIn</a>
</p>

---

## The vendor-neutral benchmark for AI infrastructure primitives

Modern AI products are assembled from **infrastructure primitives** — OCR, web search, vector
databases, rerankers, retrieval, extraction, chunking, crawling, memory. Choosing the right one is
mostly folklore today. **Primitive Bench turns that choice into evidence.**

> ### "One winner is a lie."
> No primitive wins every slice. We publish **per-slice, per-constraint** results with confidence
> intervals and statistical separability — never a single global leaderboard.

This repo is the **public trust anchor**: the harness engine, the statistics library, the adapter
SDK, the per-primitive eval packages, and the **public dev splits**. Held-out golden answers never
live here — they sit behind the private eval server, so the scores stay honest.

## What we benchmark

| Primitive | Package | What it measures | Status |
|---|---|---|---|
| Web search | `eval-websearch` | hit@k against golden-URL equivalence classes, sliced by intent | ✅ Live |
| Extraction | `eval-extraction` | token survival of clean main-content extraction | ✅ Live |
| OCR | `eval-ocr` | text fidelity across document types | 🚧 Planned |
| Vector DBs | `eval-vectordb` | recall / latency / cost across index configs | 🚧 Planned |
| Rerankers | `eval-reranker` | nDCG / MAP uplift over first-stage retrieval | 🚧 Planned |
| Retrieval | `eval-retrieval` | nDCG@k, MAP@k, MRR@k, Recall@k | 🚧 Planned |
| Chunking | `eval-chunking` | downstream retrieval quality by chunk strategy | 🚧 Planned |
| Crawling | `eval-crawl` | coverage & freshness of fetched content | 🚧 Planned |
| Memory | — | long-horizon recall (LoCoMo-style) | 🗺️ Roadmap |

Filling in a 🚧 is the highest-impact **first contribution** — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Why Primitive Bench

- **No fake #1.** A winner is named for a slice only when it's **statistically separable** from the
  runner-up (McNemar *p* < α, non-overlapping CIs); otherwise we publish a tie band.
- **Real evals, not reviews.** Every claim is backed by a **canonical, citable** statistic — McNemar,
  Wilson intervals, seeded bootstrap, Bradley-Terry / Elo.
- **Reproducible by anyone.** Deterministic seeds, pinned versions, and **public dev splits** reproduce
  public runs bit-for-bit.
- **Neutral arbiter.** No pay-to-rank. Three-tier ground truth (verified-external,
  authoritative-registry, sentinel-planted) with canary markers for contamination detection.

## Quickstart

> Packages aren't on PyPI yet — run from a clone for now.

```bash
uv sync
uv run bench run --primitive ocr --config configs/ocr.yaml
uv run bench view ./runs/<run_id>
```

The `bench` CLI scaffolds a config (`bench init`), runs an eval (`bench run`), summarizes slices with
separability badges (`bench view`), and submits to the held-out eval server for scores only
(`bench submit`).

## Query it from your agent (MCP)

The benchmark isn't just a site to read — it's a tool your **AI agent can query while it
reasons**. The [Primitive Bench MCP server](apps/mcp) exposes the per-slice leaderboards over the
Model Context Protocol, so a coding agent can ask *"which web search API wins for government
registry lookups?"* and get back the slice-specific winner — or an honest **TIE band** — with
Wilson CIs and a citation. That's "one winner is a lie," embodied: the agent supplies the
constraints, the benchmark supplies the statistically honest answer. No server-side LLM, so
queries are free.

**Add it to Claude Code:**

```bash
claude mcp add --transport http primitive-bench https://<your-deploy>/mcp
```

Live for **websearch** and **extraction** today — see [`apps/mcp`](apps/mcp) to run it locally or
deploy your own.

## How it works

Primitive Bench uses the proven harness shape — **dataset → Task → Adapter → Scorer → result schema**
(converging with EleutherAI lm-eval, UK AISI Inspect, and Stanford HELM).

**The Gate.** `bench-schemas` is the **frozen contract** (`v0.1.0`): every package imports types only
from it and writes only files it owns — no shared mutable state. That boundary is what lets the build
lanes run in parallel without colliding. See [`apps/docs/DECISIONS.md`](apps/docs/DECISIONS.md) (D-03)
and the [methodology](apps/docs/methodology/v3.md).

## Repo layout

```
packages/
  bench-schemas/   # THE FROZEN CONTRACT — RunManifest, ItemResult, SliceResult, ScorerOutput, AdapterSpec
  bench-core/      # harness engine: deterministic seeding, run/manifest, per-run dirs
  bench-stats/     # McNemar, Wilson, bootstrap CIs, hit@k, nDCG/MAP/MRR, Bradley-Terry
  bench-adapters/  # provider/primitive adapter SDK (lm-eval registry pattern)
  eval-*/          # one package per primitive: public golden dev set + scorer + slice defs
apps/
  cli/             # the `bench` CLI: init / run / view / submit
  docs/            # methodology + DECISIONS.md
golden-sets-public/  # PUBLIC dev splits only (canary-marked). Held-out answers NEVER here.
```

## Contributing

We love contributions big and small — a new vendor adapter, a slice that separates two adapters, or a
whole stubbed primitive. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md); the best first issue is
implementing one of the 🚧 verticals using `eval-websearch` / `eval-extraction` as the template.

## License

- **Code:** [Apache-2.0](LICENSE).
- **Public datasets** under [`golden-sets-public/`](golden-sets-public/): [CC-BY-4.0](golden-sets-public/LICENSE-DATA).
- Third-party attribution is in [`NOTICE`](NOTICE). We learn from lm-evaluation-harness, Inspect, HELM,
  ann-benchmarks, VectorDBBench, and OmniDocBench — and we do **not** vendor GPL/commercial-dual code.

---

<p align="center">
  Built by the <b>Primitive Bench</b> team ·
  <a href="https://www.primitivebench.com/">primitivebench.com</a> ·
  <a href="https://www.linkedin.com/company/primitive-bench">LinkedIn</a>
</p>
