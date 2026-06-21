# golden-sets-public / crawl

PUBLIC dev split for the crawl primitive. Canary-marked; reproducible by anyone.

**Held-out test answers NEVER live here** — they sit behind the private eval
server (primitivebench-platform/apps/eval-server) with HMAC-keyed split integrity.

The crawl task: hand a crawler a **seed URL**; it must DISCOVER and FETCH the pages
under it. For each golden TARGET page the cell is a hit iff the crawler **reached**
it (coverage) AND returned its **current content** so the `truth_token` survives
(freshness). That binary — `page_coverage` — is the leaderboard separability gate;
every miss decomposes into `not_reached` (coverage gap) vs `stale` (freshness gap)
vs blocked / token_absent / empty.

## What ships here vs. generated locally

- `dev.example.jsonl` — the **controlled** split, committed. Each "site" is a
  known-structure link graph on RFC-2606 `.example` domains (carried inline on the
  seed's first row), so coverage ground truth is **exact** and the keyless local
  crawlers run **offline, network-free, and reproducibly**. This is the scored set
  the public snapshot is generated from. Regenerate it with:
  ```bash
  uv run python packages/eval-crawl/tools/build_controlled_set.py
  ```
- `dev.jsonl` — the **live** split, generated locally from real-site sitemaps and
  **git-ignored** (never redistributed). Build it (needs network) with:
  ```bash
  uv run python packages/eval-crawl/tools/build_crawl_goldenset.py --per-seed 16 --seed 0
  ```
  Targets are sampled from each seed's `sitemap.xml` (the authoritative page
  registry), hop distance is measured by a bounded static BFS, and every target's
  `truth_token` is re-verified by `bench_core.verify.liveness_gate` before it is
  kept. Seeds are listed in `packages/eval-crawl/tools/seeds.toml`.
- `CANARY` — the contamination canary GUID (BIG-bench convention), embedded as the
  header comment of each split file.

## Row schema (a TARGET page)

```jsonc
{ "row_id": "docs_006",
  "seed_id": "docs",                                  // groups targets sharing a seed
  "seed_url": "https://docs.example/",                // the crawl entry point
  "target_url": "https://docs.example/s1/p2",         // the page that must be covered
  "equivalence_members": ["https://docs.example/s1/p2"],  // optional URL aliases
  "truth_token": "TOK-docs-s1-p2",                    // must survive in the fetched content
  "render": "static_html",                            // static_html | js_rendered
  "hops": 3,                                           // measured distance from the seed
  "site_type": "docs",                                // docs | blog | commerce | gov_registry
  "stratum": "sentinel",                              // sentinel | coverage | freshness
  "slices": ["site_type:docs", "render:static_html", "depth:deep", "freshness:stable"],
  "ground_truth_tier": "sentinel_planted",
  "site": { "pages": { "<url>": {"content","links","js_links"} }, "sitemap": [ ... ] } }  // inline graph (controlled split only; on the seed's first row)
```

The crawler is invoked once per `seed_id`; the scorer reads its returned pages and
scores each target (reached + token survives). Slices are assigned deterministically
from the target facts (see `packages/eval-crawl/src/eval_crawl/slicing.py`).

## Why a controlled set for the public snapshot

You can never know *every* page a real site has, so real-site coverage truth is
unknowable. The committed split therefore uses **controlled sites with known
structure** (a standard web-crawler-evaluation design): the inline graph IS the
ground truth, so recall is exact and anyone reproduces the numbers with no network.
The four keyless crawl strategies (`bfs-shallow`, `bfs-deep`, `sitemap-crawl`,
`render-crawl`) separate exactly on the slice axes — shallow misses deep pages, only
the JS-following crawl reaches `render:js_rendered` targets, only the sitemap-aware
crawl reaches sitemap-only pages. The live split then measures the hosted vendors
(Firecrawl / Tavily / Spider / Apify) over real seeds.

## Provenance & license

`dev.example.jsonl` is original, authored by the Primitive Bench team (the
controlled-set generator) and released under CC-BY-4.0. Live-split seeds
(`seeds.toml`) are public scraping sandboxes (`toscrape.com`,
`scrapethissite.com`) and sitemap-publishing docs sites, chosen to be ethical to
crawl; the live `dev.jsonl` derived from them is generated locally and not
redistributed.
