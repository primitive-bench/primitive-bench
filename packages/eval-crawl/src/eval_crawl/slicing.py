"""Deterministic slice assignment for crawl golden targets.

Slices are the "one winner is a lie" surface: a constraint that can *separate*
crawlers. Assigned from row facts only (no model, no fetch), so the golden-set
builder and the Task agree. Four axes:

  * `render`     — static_html vs js_rendered. THE primary separator: a page whose
    links/content only appear after JavaScript execution is invisible to a plain
    HTTP crawler and needs a headless browser. This is the central capability gap
    in crawling (the crawl analogue of extraction's anti-bot `government_registry`).
  * `depth`      — shallow (target within 2 hops of the seed) vs deep (>=3 hops).
    Separates breadth-limited crawlers from thorough ones; a shallow crawler that
    stops early simply never reaches deep pages.
  * `site_type`  — docs / blog / commerce / gov_registry. The content domain;
    different site shapes (paginated catalogs, JS docs, registries) stress
    discovery differently.
  * `freshness`  — time_varying vs stable target. On a time-varying page a crawler
    serving a stale/cached copy misses the current `truth_token`; on a stable page
    it cannot. Isolates the freshness axis (Cho & Garcia-Molina) from coverage.
"""
from __future__ import annotations

# A target >= this many hops from the seed is a "deep" discovery problem.
DEEP_HOPS = 3


def depth_bucket(hops: int) -> str:
    return "deep" if int(hops) >= DEEP_HOPS else "shallow"


def freshness_bucket(stratum: str) -> str:
    return "time_varying" if stratum == "freshness" else "stable"


def slice_keys(site_type: str, render: str, hops: int, stratum: str) -> list[str]:
    """Slice keys for a target: site_type, render, depth, freshness."""
    return [
        f"site_type:{site_type}",
        f"render:{render}",
        f"depth:{depth_bucket(hops)}",
        f"freshness:{freshness_bucket(stratum)}",
    ]
