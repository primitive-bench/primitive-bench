<p align="center">
  <img src="docs/assets/primitive-bench-logo.svg" alt="Primitive Bench" width="110">
</p>

<h1 align="center">Primitive Bench</h1>

<p align="center">
  <b>The marketplace for verifiable AI outcomes.</b><br>
  State a task, get a fixed price upfront. We route across the right tools to complete it.<br>
  If the outcome isn't delivered, you get your money back.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="CONTRIBUTING.md"><img alt="PRs welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
</p>

<p align="center">
  <a href="https://cal.com/arlen-frederick-kumar-hs0w6h/primitivebench"><b>Book a call</b></a> ·
  <a href="https://www.primitivebench.com/">Website</a> ·
  <a href="apps/docs">Docs</a> ·
  <a href="apps/docs/methodology/v3.md">Methodology</a> ·
  <a href="CONTRIBUTING.md">Contributing</a> ·
  <a href="https://www.linkedin.com/company/primitive-bench">LinkedIn</a>
</p>

---

## From tool calls to guaranteed outcomes

A year ago, vibe coding arrived. Next, agents run full end-to-end workflows: choosing a domain
name, buying a design, paying a hosting provider, deploying the app. Developers and indie founders
will lead; enterprises will follow. Every one of those workflows is a chain of tool calls that has
to actually work.

Most tool marketplaces stop at the call: they hand your agent an API and wish it luck. **Primitive
Bench sells the outcome, not the call.**

- **State a task.** Describe what you want done, in plain language.
- **Get a fixed price upfront.** The estimator prices the job from known, published API pricing
  before you commit a cent.
- **We route and execute.** Primitive Bench internally selects and calls the sub-APIs that get it
  done, and traces every step.
- **Outcome guaranteed, or refunded.** If the outcome isn't delivered, you get your money back.

> ### Load your AI into the money, get a guaranteed outcome. If it isn't done, get your money back.

<p align="center">
  <a href="https://cal.com/arlen-frederick-kumar-hs0w6h/primitivebench"><b>Book a call →</b></a>
</p>

## Why this, not another tool aggregator

| | Aggregators (e.g. Apify) | **Primitive Bench** |
|---|---|---|
| Open marketplace | ✅ anyone can list tools | ✅ anyone can upload tools **and earn from them** |
| What you pay for | a tool call | **a verified outcome** |
| If it fails | your problem | **refund guarantee** |
| Cost | metered, discovered as you go | **fixed estimate upfront** from known API pricing |
| The layer | tool aggregation | **routing + observability over tools** |

An open marketplace alone is not a moat. The wedge is the layer on top: **route to the right tools,
observe what they actually did, and stand behind the result.** Primitive Bench is the observability
and routing layer for tools, not just a place to find them.

## The verification layer

An outcome guarantee is only as honest as the machinery that decides whether the outcome was
delivered. That machinery is what most of this repo is: a **vendor-neutral harness that measures
whether a tool actually did its job**, per slice, with real statistics.

The open engine here observes input and output, traces every tool call, and scores the result
against golden answers with confidence intervals and separability tests. That observability layer is
how Primitive Bench knows a task is "done" and where routing should send the next one. Held-out
golden answers never live in this repo; they sit behind the private eval server, so the scores stay
honest and the marketplace has a neutral arbiter no vendor can pay to move.

> ### "One winner is a lie."
> No tool wins every slice. We score **per-slice, per-constraint** with confidence intervals and
> statistical separability, so routing picks the right tool for *your* task instead of a single
> global favorite.

## What we measure

The primitives below are the building blocks routing composes into outcomes. Each is scored by a
public eval package so the routing and refund decisions rest on evidence, not folklore.

| Primitive | Package | What it measures | Status |
|---|---|---|---|
| Web search | `eval-websearch` | hit@k against golden-URL equivalence classes, sliced by intent | ✅ Live |
| Extraction | `eval-extraction` | token survival of clean main-content extraction | ✅ Live |
| OCR | `eval-ocr` | text fidelity across document types | 🚧 Planned |
| Vector DBs | `eval-vectordb` | recall / latency / cost across index configs | 🚧 Planned |
| Rerankers | `eval-reranker` | nDCG@10 / MAP / MRR + hit@1, sliced by domain & hard-negative density | ✅ Live |
| Retrieval | `eval-retrieval` | nDCG@10 / recall@10 / MAP / MRR + success@10, sliced by domain & relevant-set | ✅ Live |
| Chunking | `eval-chunking` | downstream retrieval quality by chunk strategy | 🚧 Planned |
| Crawling | `eval-crawl` | coverage & freshness of fetched content | 🚧 Planned |
| Memory | — | long-horizon recall (LoCoMo-style) | 🗺️ Roadmap |

Filling in a 🚧 is the highest-impact **first contribution**, and it directly widens what the
marketplace can route and guarantee. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## What keeps the guarantee honest

- **No fake #1.** A winner is named for a slice only when it's **statistically separable** from the
  runner-up (McNemar *p* < α, non-overlapping CIs); otherwise we publish a tie band, and routing
  treats them as interchangeable.
- **Real evals, not reviews.** Every claim is backed by a **canonical, citable** statistic: McNemar,
  Wilson intervals, seeded bootstrap, Bradley-Terry / Elo.
- **Reproducible by anyone.** Deterministic seeds, pinned versions, and **public dev splits**
  reproduce public runs bit-for-bit.
- **Neutral arbiter.** No pay-to-rank. Three-tier ground truth (verified-external,
  authoritative-registry, sentinel-planted) with canary markers for contamination detection.

## Quickstart

> Packages aren't on PyPI yet, so run from a clone for now.

```bash
uv sync
uv run bench run --primitive ocr --config configs/ocr.yaml
uv run bench view ./runs/<run_id>
```

The `bench` CLI scaffolds a config (`bench init`), runs an eval (`bench run`), summarizes slices with
separability badges (`bench view`), and submits to the held-out eval server for scores only
(`bench submit`).

## Query it from your agent (MCP)

The verification layer isn't just a site to read, it's a tool your **AI agent can query while it
reasons**. The [Primitive Bench MCP server](apps/mcp) exposes the per-slice leaderboards over the
Model Context Protocol, so a coding agent can ask *"which web search API wins for government
registry lookups?"* and get back the slice-specific winner, or an honest **TIE band**, with Wilson
CIs and a citation. That's the routing brain, exposed: the agent supplies the constraints, the
benchmark supplies the statistically honest answer. No server-side LLM, so queries are free.

**🟢 Live now** at `https://benchpublic.vercel.app/mcp`, add it to Claude Code:

```bash
claude mcp add --transport http primitive-bench https://benchpublic.vercel.app/mcp
```

Live for **websearch**, **extraction**, **reranker**, and **retrieval** today. See [`apps/mcp`](apps/mcp) to run it
locally or deploy your own.

## How it works

Primitive Bench uses the proven harness shape (**dataset → Task → Adapter → Scorer → result schema**,
converging with EleutherAI lm-eval, UK AISI Inspect, and Stanford HELM) as the observability
substrate under the marketplace.

**The Gate.** `bench-schemas` is the **frozen contract** (`v0.1.0`): every package imports types only
from it and writes only files it owns, with no shared mutable state. That boundary is what lets the
build lanes run in parallel without colliding. See [`apps/docs/DECISIONS.md`](apps/docs/DECISIONS.md) (D-03)
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

## Roadmap

The verification layer is live today. The outcome marketplace (fixed-price estimator, routing across
sub-APIs, and the refund guarantee) is being built on top of it. Near-term work:

- **Cost estimator** that prices a task from known, published API pricing before execution.
- **Routing** that turns a stated task into the sequence of tool calls that satisfies it.
- **Outcome verification thresholds**: a precise, per-task-type definition of "done" measured from
  the traced input/output, and the scope limits on what Primitive Bench will and won't take on.
- **Open contributor payouts** so anyone who uploads a tool earns when routing uses it.

Have a workflow you'd want guaranteed end-to-end?
**[Book a call →](https://cal.com/arlen-frederick-kumar-hs0w6h/primitivebench)**
That's exactly the customer discovery we're doing now.

## Contributing

We love contributions big and small: a new vendor adapter, a slice that separates two tools, or a
whole stubbed primitive. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md); the best first issue is
implementing one of the 🚧 verticals using `eval-websearch` / `eval-extraction` as the template.

## License

- **Code:** [Apache-2.0](LICENSE).
- **Public datasets** under [`golden-sets-public/`](golden-sets-public/): [CC-BY-4.0](golden-sets-public/LICENSE-DATA).
- Third-party attribution is in [`NOTICE`](NOTICE). We learn from lm-evaluation-harness, Inspect, HELM,
  ann-benchmarks, VectorDBBench, and OmniDocBench, and we do **not** vendor GPL/commercial-dual code.

---

<p align="center">
  Built by the <b>Primitive Bench</b> team ·
  <a href="https://www.primitivebench.com/">primitivebench.com</a> ·
  <a href="https://www.linkedin.com/company/primitive-bench">LinkedIn</a>
</p>
