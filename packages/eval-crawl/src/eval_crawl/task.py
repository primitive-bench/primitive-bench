"""crawl Task + Scorer (site coverage & freshness over seed crawls).

`Task` yields one item per golden TARGET page (a page that should be reachable
from a seed, plus the `truth_token` that must survive into the fetched content).
`Scorer` does not crawl — it scores a crawler's seed-crawl result for that target
via `scoring.score_crawl` (reached AND fresh content delivered). The leaderboard
slices each target by render / depth / site_type / freshness, so one board becomes
many — the "one winner is a lie" surface for crawling.
"""
from __future__ import annotations

from typing import Any, Iterable

from bench_core import Scorer as _Scorer, Task as _Task
from bench_schemas import ScorerOutput
from bench_schemas.models import Primitive

from eval_crawl.scoring import score_crawl
from eval_crawl.slicing import slice_keys


class Scorer(_Scorer):
    """page_coverage (paired binary): did the crawler reach the target AND return
    its current content (truth_token survives)?

    `raw` is the crawler's bench_adapters result for the target's SEED (`pages` =
    ``[{url, content}]``, `returned_urls`). Misses are decomposed (not_reached /
    stale / token_absent / blocked / empty) onto ScorerOutput.miss_reason; a crawl
    error is an uncharged non-attempt (`correct=None`). See scoring.score_crawl.
    """

    def score(self, item: dict[str, Any], raw: dict[str, Any]) -> ScorerOutput:
        return score_crawl(item, raw)


class Task(_Task):
    primitive = Primitive.CRAWL
    task_version = "crawl@1"
    dataset_version = "crawl-2026.06.controlled.v1"

    def __init__(self, rows: Iterable[dict[str, Any]] | None = None) -> None:
        """`rows` are golden TARGET rows (seed_url + target_url + truth_token +
        site_type/render/hops/stratum, and optionally an inline `site` graph for
        the controlled offline split).

        When omitted the Task carries no rows (the public DEV split is loaded by the
        harness via `eval_crawl.loader.load_rows`). Each yielded item is normalized
        to the shape the Scorer + runner expect.
        """
        self._rows = list(rows or [])

    def items(self) -> Iterable[dict[str, Any]]:
        for r in self._rows:
            site_type = str(r.get("site_type", "unknown"))
            render = str(r.get("render", "static_html"))
            hops = int(r.get("hops", 0))
            stratum = str(r.get("stratum", "coverage"))
            slices = r.get("slices") or slice_keys(site_type, render, hops, stratum)
            yield {
                "id": r.get("row_id") or r.get("id"),
                "seed_id": r.get("seed_id") or r.get("seed_url"),
                "seed_url": r.get("seed_url", ""),
                "target_url": r.get("target_url") or r.get("url", ""),
                "equivalence_members": list(r.get("equivalence_members") or []),
                "truth_token": r.get("truth_token", ""),
                "token_depth": int(r.get("token_depth", -1)),
                "site_type": site_type,
                "render": render,
                "hops": hops,
                "stratum": stratum,
                "slices": slices,
                "ground_truth_tier": r.get("ground_truth_tier", "sentinel_planted"),
                # The inline link graph rides on the row for the offline split; the
                # runner reads it once per seed. Absent for the live dev split.
                "site": r.get("site"),
                "max_pages": int(r.get("max_pages", 60)),
                "max_depth": int(r.get("max_depth", 4)),
            }

    def scorer(self) -> _Scorer:
        return Scorer()
