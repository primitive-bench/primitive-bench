# Inter-package interface contract (porting coordination)

All porting lanes import across packages using EXACTLY these names. Do not invent
alternates. `bench-schemas` is FROZEN — import types from it, never modify it.

## bench-schemas (frozen, do not edit)
`from bench_schemas import RunManifest, ItemResult, SliceResult, ScorerOutput, AdapterSpec, StatTest`
`from bench_schemas.models import Primitive, GroundTruthTier`

## bench-core (owned by Lane CORE)
- `from bench_core import Task, Scorer, run_task, RunDir`  (already exist)
- `from bench_core.urls import EquivalenceClass, normalize_url`
    - `EquivalenceClass(canonical: str, members: list[str])`, method `.contains(url) -> bool`
- `from bench_core.http import fetch, FetchResult`
    - `async fetch(url) -> FetchResult` with fields `status:int, final_url:str, main_text:str, soft_404:bool`
- `from bench_core.split import hmac_split`   # exact-quota 70/30 per stratum, salt from env HMAC_SALT
- `from bench_core.verify import liveness_gate, Liveness`
- `from bench_core.goldgen import ...`        # registry-delta pump adapters (SEC EDGAR, Fed Register, NVD, GitHub)

## bench-stats (owned by Lane STATS)
- `from bench_stats import wilson, mcnemar, separable, bootstrap_ci, hit_at_k, ndcg_at_k, map_at_k, mrr_at_k`  (already exist)
- ALSO export (new): `mcnemar_pair, mcnemar_power, required_n, cmh_global, cusum, tied_rank_band`
- All public functions return `bench_schemas.StatTest` OR a documented dataclass; keep the existing
  signatures working (they have passing tests).

## bench-adapters (owned by Lane ADAPTERS)
- `from bench_adapters import Adapter, register, get, registry`  (already exist)
- Concrete adapters registered by name under:
    - search:     `brave, exa, tavily, serpapi`
    - extraction: `firecrawl, jina, exa_live, tavily_extract, apify`
- Each adapter `.invoke(item) -> dict` returns at least `{raw_output, latency_ms, cost_usd}` plus
  primitive extras (search: `returned_urls`; extraction: `main_text`).

## eval-websearch (owned by Lane WEBSEARCH) — owns packages/eval-websearch only
## eval-extraction (owned by Lane EXTRACTION) — owns packages/eval-extraction only

## golden-sets-public (owned by Lane GOLDEN)

Provenance: ported from arlenk2021/GoldenEvalsWebSearch (Apache-2.0 code, CC-BY-4.0 data).
The companion repo arlenk2021/FindingsWebExtract is currently empty (nothing pulled).
Relicensed to MIT within primitive-bench by the author; keep a one-line provenance note in
each package README.
