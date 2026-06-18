# eval-websearch

Primitive Bench vertical for **web search**: three-tier ground truth, query-form
strata, and per-intent slice leaderboards with McNemar/Wilson separability.

A search vendor returns a ranked list of URLs for a query. We score `hit@{1,5}`
as **pure set membership** after URL normalization against each golden row's
equivalence class — no judge, no human. The honest discriminator is the
**descriptive** query form (the unique identifier removed), not the
`token_in_query` navigational form that every vendor saturates.

## The headline: slicing reveals the winner

A single leaderboard ("Vendor A is #1") hides the real story. Slice by query
**intent** and the winner changes per slice — and one slice produces a clear,
statistically separable winner where the global ranking would have blurred it:

| Slice | n | Winner / tie band | Loses here |
|---|---|---|---|
| company_lookup | 25 | **Exa (88%)** 🏆 | brave 0%, serpapi 0%, tavily 0% |
| government_registry | 90 | TIE: exa 77%, serpapi 67% | brave 49%, tavily 40% |
| citation_needed | 90 | TIE: exa 77%, serpapi 67% | brave 49%, tavily 40% |
| technical_docs | 25 | TIE: exa 100%, serpapi 92%, tavily 92% | brave 72% |

**Exa wins `company_lookup` 88% vs 0%** for every other vendor — a slice where
Brave, SerpAPI, and Tavily return *nothing* in the equivalence class. That signal
is invisible in a pooled leaderboard (Exa ties on most other slices); it only
appears once you slice by intent. A winner is named **only** when its Wilson
interval clears the runner-up's (`bench_stats.tied_rank_band`); otherwise the
slice publishes a TIE band, not a fake #1.

Full per-slice tables and the wins matrix: [`reports/slices-search.md`](reports/slices-search.md).
Extraction snapshot (the harness that validated the method): [`reports/snapshot-v2-analysis.md`](reports/snapshot-v2-analysis.md).

## The four ranking-specific instruments

1. **Mirror auto-promotion.** A vendor returning `govinfo.gov` instead of
   `federalregister.gov`, or `cve.org` instead of `nvd.nist.gov`, is *correct*.
   At scoring time any returned URL whose page holds the truth token is
   auto-promoted into the equivalence class (cross-domain allowed) and logged to
   adjudication — never scored as a false miss. (`scoring.find_promotions`)
2. **Query-form strata.** Every row carries `query_variants`: `token_in_query`
   (easy, navigational, saturates) and `descriptive` (no identifier, the honest
   discriminator and the default). (`queries.py`)
3. **Miss taxonomy.** Each miss is classified `ranked_below_k`,
   `mirror_not_in_class` (promoted), or `not_found`. `not_found` **cannot** be
   split into "not indexed" vs "ranked deeper than k" without the **sentinel**
   instrument. (`scoring.classify_miss`)
4. **Pinned k + normalization.** `hit@1` and `hit@5`, k fixed per snapshot. AMP
   variants normalize to canonical; redirect chains resolve to the final URL;
   doc-vs-PDF are not normalized equal (admitted via promotion instead).

The **sentinel retrievability verdict** is the only honest way to split a
`not_found` miss: because the bench owns the sentinel page and minted a unique
truth token, one `token_in_query` indexing probe tells `ranked_below_k` (indexed,
just ranked below k) from `not_indexed` (true index lag). (`scoring`,
`probe._sentinel_verdict`)

## The 10 intent slices

`company_lookup`, `government_registry`, `citation_needed`, `technical_docs`,
`fresh_news`, `pricing_pages`, `b2b_tools`, `docs_lookup`, `navigational`,
`long_tail`. Definitions and assignment rules: [`slices.yaml`](slices.yaml) /
`eval_websearch.slices`.

## Layout

| Module | What |
|---|---|
| `scoring.py` | hit@k set membership, miss taxonomy, sentinel verdict, `find_promotions` |
| `queries.py` | descriptive vs token_in_query query forms |
| `probe.py` | probe runner: liveness gate -> search adapters -> `ItemResult` |
| `slices.py` | 10 intent slices, deterministic assignment, per-slice wins matrix |
| `task.py` | `Task` / `Scorer` (subclass bench-core), `primitive=Primitive.WEBSEARCH` |

## Cross-package imports

```python
from bench_core.urls import EquivalenceClass
from bench_core.http import fetch
from bench_core.verify import liveness_gate
from bench_core.domain import Stratum, stratum_to_tier
from bench_stats import wilson, tied_rank_band
from bench_adapters import get               # search adapters: brave, exa, tavily, serpapi
from bench_schemas import Primitive, ItemResult, ScorerOutput
from bench_schemas.models import GroundTruthTier
```

## Provenance

Ported from arlenk2021/GoldenEvalsWebSearch (Apache-2.0 code, CC-BY-4.0 data).
Relicensed to MIT within primitive-bench by the author.
