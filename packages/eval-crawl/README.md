# eval-crawl

Primitive Bench vertical for **crawling**: hand a crawler a **seed URL**, it
discovers and fetches the pages under it, and we score **page coverage &
freshness** — a target page is a hit iff the crawler **reached** it (coverage) AND
returned its **current content** so the golden `truth_token` survives (freshness).

Coverage and freshness are the two canonical axes of crawl quality (Olston &
Najork, *Web Crawling*, Foundations and Trends in IR, 2010; freshness/age formally
defined by Cho & Garcia-Molina, SIGMOD 2000 / ACM TODS 2003). `page_coverage` is a
paired-binary outcome, so the same McNemar/Wilson separability gate as every other
vertical applies — per-slice winners or honest TIE bands, never one global ranking.

## Headline result (public controlled snapshot)

The public board is generated **offline** from controlled known-structure sites
(exact coverage ground truth, network-free), comparing four real crawl strategies:

| Slice | Call |
| --- | --- |
| `render:js_rendered` | **WINNER `render-crawl`** — only a JS-following crawl reaches JS-rendered pages |
| `depth:deep` | **TIE** `bfs-deep` / `sitemap-crawl` / `render-crawl` (the shallow crawler reaches none) |
| `render:static_html` | **TIE** `sitemap-crawl` / `render-crawl` (both cover static fully) |
| `site_type:*` | TIE bands led by `render-crawl` |

The methodology highlight is the **miss decomposition**: a low score is never one
opaque number. `render-crawl` covers 100% with `not_reached=0`; `bfs-shallow`
covers 29% with `not_reached=40` — and that gap is a **coverage** failure
(discovery), reported apart from a **freshness** failure (`stale`, a reached page
served from a stale cache). The two axes also ride on every item as
`url_discovered` (pure URL-recall) and `content_fresh`.

> The controlled board separates the **render** and **depth** axes authentically;
> the **freshness** axis is primarily a *live*-run signal (a stale vendor index
> serving an old copy), so it shows as a TIE offline. The live board
> (`dev.jsonl` + the hosted vendors) refreshes these numbers over real sites.

## Modules

- `scoring.py` — `score_crawl` emits a `bench_schemas.ScorerOutput`: `correct` is
  `page_coverage` (reached AND fresh-content-delivered); misses decompose into
  `not_reached` / `stale` / `token_absent` / `blocked` / `empty`; `metrics` carries
  `url_discovered` + `content_fresh` so coverage and freshness stay separable.
- `runner.py` — invokes each crawler **once per seed** (a multi-page crawl is
  expensive), caches it, and scores every target under that seed. A
  `VendorUnavailable` skips a lane uncharged; a per-seed crawl error is uncharged
  for that seed's targets. Streams `bench_schemas.ItemResult` to the run dir.
- `report.py` — the page_coverage leaderboard with `bench_stats.wilson` intervals +
  the McNemar separability gate, the coverage/freshness axes summary, and the miss
  decomposition; `write_counts_toml` emits `snapshots/crawl.counts.toml`.
- `task.py` — `Task` / `Scorer` (subclass `bench_core`), `primitive = Primitive.CRAWL`.
- `slicing.py` / `slices.yaml` — the four slice axes: **render** (static vs JS — THE
  primary separator), **depth** (shallow vs deep), **site_type**, **freshness**.
- `tools/build_controlled_set.py` — builds the committed controlled `dev.example.jsonl`.
- `tools/build_crawl_goldenset.py` + `tools/seeds.toml` — builds the live `dev.jsonl`
  from real sitemaps (verified by the liveness gate; git-ignored).

## Adapters

Four keyless local crawl strategies (the reproducible offline board) + four hosted
crawl APIs (the live-run systems-under-test) — see
`packages/bench-adapters/README.md`:
`bfs-shallow`, `bfs-deep`, `sitemap-crawl`, `render-crawl` (keyless);
`firecrawl-crawl`, `tavily-crawl`, `spider-crawl`, `apify-crawl` (hosted).

## Run it

```bash
# keyless, reproducible (regenerates the public snapshot from the controlled split):
uv run python -m eval_crawl --golden golden-sets-public/crawl/dev.example.jsonl \
    --run-id crawl-public-snapshot

# live run incl. hosted vendors (build dev.jsonl first; APIs skipped without a key):
uv run python packages/eval-crawl/tools/build_crawl_goldenset.py --per-seed 16
uv run python -m eval_crawl --golden golden-sets-public/crawl/dev.jsonl \
    --adapters bfs-deep,firecrawl-crawl,tavily-crawl,spider-crawl,apify-crawl --run-id crawl-dev

uv run bench results emit    # -> data/results/public-snapshot/crawl.json + apps/mcp/data/crawl.json
```

## Provenance

Greenfield vertical authored for Primitive Bench (Apache-2.0 code, CC-BY-4.0 data),
following the `eval-reranker` / `eval-extraction` template. The live-split seeds are
public scraping sandboxes and sitemap-publishing docs sites, chosen to be ethical to
crawl.
