# eval-extraction

Primitive Bench vertical for **web_extraction**: hand a vendor a URL, get back
clean main content, and score it by **token survival** — a cell is a hit iff the
golden row's `truth_token` survives the vendor's extraction into main content.

## Headline result

On the public DEV split, the major extraction vendors are NOT separable on raw
token survival — but the *one that loses, loses for a reason that is not its
extraction quality*:

| Vendor          | Token survival | Note                                       |
| --------------- | -------------- | ------------------------------------------ |
| Firecrawl       | ~100%          |                                            |
| Jina            | ~100%          |                                            |
| Tavily (extract)| ~100%          |                                            |
| Exa (live)      | **33%**        | collapses on the Federal Register slice    |

The methodology highlight is the **miss decomposition**. Exa's 33% is not a
freshness or extraction-skill failure: of its 100 misses on the Federal Register
slice, **98/100 are `blocked`** (an anti-bot / access-wall interstitial —
"request access", "just a moment", Cloudflare), and only the remainder are
`token_absent`. **The gap was anti-bot blocking, not freshness.** A leaderboard
that reported only the 33% number would slander Exa's extractor; the miss
breakdown is what makes the actual capability gap (bot evasion) legible.

This is why `ScorerOutput.miss_reason` carries `blocked` / `truncated` /
`token_absent` / `empty`, and why a `blocked` miss is surfaced separately rather
than charged as a genuine extraction failure against the accuracy denominator.

## Modules

- `scoring.py` — token-survival scoring + miss classification. `score_extraction`
  emits a `bench_schemas.ScorerOutput`; `classify_miss` decomposes a miss into
  blocked / truncated / token_absent / empty. Depth-conditional survival
  (`token_locate`) separates title-zone tokens (CVE IDs at offset ~0, survive any
  truncation) from deep-body tokens (FR doc numbers, die under truncation).
- `probe.py` — the extract probe runner (from `extract/main.py`), adapted to call
  the bench_adapters extraction adapters (`firecrawl`, `jina`, `exa_live`,
  `tavily_extract`, `apify`) and emit `bench_schemas.ItemResult`. Re-applies the
  `bench_core.verify.liveness_gate` ("truth checked at use") so a row whose own
  gold page no longer carries the token is excluded batch-wide, never charged.
- `report.py` — the token-survival leaderboard (from `extract/report.py`):
  per-vendor survival rate with `bench_stats.wilson` intervals, plus the
  blocked/truncated/token_absent miss breakdown.
- `task.py` — `Task` / `Scorer` (subclass `bench_core`), `primitive =
  Primitive.EXTRACTION`.
- `slices.yaml` — extraction slices: the GoldenEvals intent vocabulary
  (`government_registry` is THE anti-bot slice, plus `company_lookup`,
  `technical_docs`, `citation_needed`, …), the `Stratum` set, and token-depth
  slices. web_extraction and web_search share one slice vocabulary.

## Provenance

Ported from arlenk2021/GoldenEvalsWebSearch (Apache-2.0 code, CC-BY-4.0 data; the
extract probe and token-survival report). Relicensed to MIT within primitive-bench by
the author. The intended companion source repo `arlenk2021/FindingsWebExtract` is
currently empty — nothing was pulled from it (see workspace TODO).
