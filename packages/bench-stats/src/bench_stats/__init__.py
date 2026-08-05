"""bench-stats — the statistics library for Primitive Bench.

Canonical, citable methods only. These produce the StatTest objects carried on
SliceResult, and the `separable` trust gate the leaderboard depends on.

  mcnemar(...)        — paired comparison on the same test set (McNemar 1947, w/ continuity correction)
  wilson(...)         — CI for a single proportion (Wilson 1927; preferred over CLT for eval-sized n)
  bootstrap_ci(...)   — seeded bootstrap CI for non-proportion metrics (nDCG, MCC, AUROC)
  hit_at_k(...)       — retrieval hit@k
  ndcg_at_k / map_at_k / mrr_at_k  — BEIR/MTEB-standard IR metrics
  precision_at_k / recall_at_k / r_precision  — set-based IR metrics (TREC/pytrec_eval)
  separable(...)      — McNemar-based separability decision for two adapters on a slice

Richer reporting helpers (scipy/statsmodels-backed, ported from
arlenk2021/GoldenEvalsWebSearch) live in `bench_stats.reporting`:

  mcnemar_pair(...)   — paired McNemar (exact binomial < 25 discordant, else corrected)
  mcnemar_power(...)  — Connor 1987 power approximation for sizing from a pilot
  required_n(...)     — smallest n reaching target power for McNemar
  cmh_global(...)     — Cochran–Mantel–Haenszel omnibus across vendors/strata
  cusum(...)          — one-sided CUSUM for drift/regression on an anchor set
  tied_rank_band(...) — name a winner only when its Wilson interval clears the runner-up's

The proportions/resampling/retrieval modules stay scipy-free; only `reporting`
pulls in scipy/statsmodels. See DECISIONS.md D-04.
"""

from bench_stats.proportions import mcnemar, wilson, separable
from bench_stats.resampling import bootstrap_ci
from bench_stats.retrieval import (
    hit_at_k,
    ndcg_at_k,
    map_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
    r_precision,
)
from bench_stats.reporting import (
    mcnemar_pair,
    mcnemar_power,
    required_n,
    cmh_global,
    cusum,
    tied_rank_band,
)
from bench_stats.leaderboard import (
    SliceReport,
    PrimitiveReport,
    build_slice_report,
    build_primitive_report,
)

__all__ = [
    "mcnemar",
    "wilson",
    "separable",
    "bootstrap_ci",
    "hit_at_k",
    "ndcg_at_k",
    "map_at_k",
    "mrr_at_k",
    "precision_at_k",
    "recall_at_k",
    "r_precision",
    "mcnemar_pair",
    "mcnemar_power",
    "required_n",
    "cmh_global",
    "cusum",
    "tied_rank_band",
    "SliceReport",
    "PrimitiveReport",
    "build_slice_report",
    "build_primitive_report",
]
