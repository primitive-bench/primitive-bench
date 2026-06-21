"""Generate the COMMITTED controlled crawl example split (network-free, reproducible).

Real-site coverage ground truth is unknowable (you never know every page a site
has), so the public, reproducible portion of the crawl benchmark uses **controlled
sites with KNOWN structure** — a classic web-crawler evaluation design. Each site
is an inline link graph (`pages: {url: {content, links, js_links}}` + `sitemap`)
on RFC-2606 `.example` domains, so the keyless local crawlers can be scored offline
with EXACT coverage truth and anyone can reproduce the numbers with no network.

Four archetype sites (docs / blog / commerce / gov_registry), each built the same
parametric way:
  * seed -> s1,s2,s3 (hop 1, static) -> sX/p1 (hop 2) -> sX/p2 (hop 3) -> sX/p3 (hop 4)
  * sX/js   — reachable ONLY via a JS-rendered link (render=js_rendered; NOT in sitemap)
  * extra1/2 — present ONLY in sitemap.xml (discoverable only by a sitemap-aware crawl)
  * changelog + two more targets carry a time-varying `truth_token` (freshness stratum)

The hop distance of each target is computed by a static BFS from the seed, so the
`depth` slice is derived, not asserted. Targets span the four slice axes (render /
depth / site_type / freshness) with n>=10 on each primary axis. The four keyless
crawl strategies separate exactly on these axes:
  bfs-shallow reaches only hop-1; bfs-deep walks the static chain; sitemap-crawl
  adds the sitemap-only pages; render-crawl additionally follows JS links.

Run (writes the committed example split + the selection manifest):
    uv run python packages/eval-crawl/tools/build_controlled_set.py
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from eval_crawl._paths import CANARY, GOLDEN_CRAWL_DIR
from eval_crawl.slicing import slice_keys

MANIFEST = Path(__file__).resolve().parents[1] / "selection_manifest.json"

# A fixed "current release" marker stands in for the time-varying value a live
# freshness target would carry. In the controlled split it is a planted token; the
# live builder (build_crawl_goldenset.py) uses a real sitemap <lastmod>/version.
FRESH_TAG = "2026.06-rel"

# (site_id, base_url, site_type)
SITES = [
    ("docs", "https://docs.example", "docs"),
    ("blog", "https://blog.example", "blog"),
    ("shop", "https://shop.example", "commerce"),
    ("reg", "https://registry.example", "gov_registry"),
]


def _build_graph(site_id: str, base: str) -> tuple[str, dict[str, dict[str, Any]], list[str]]:
    """Build one archetype site's inline link graph + sitemap. Returns (seed, pages, sitemap)."""
    pages: dict[str, dict[str, Any]] = {}
    sitemap: list[str] = []

    def add(path: str, content: str, links: list[str] | None = None,
            js_links: list[str] | None = None, in_sitemap: bool = True) -> str:
        url = base + path
        pages[url] = {
            "content": content,
            "links": [base + ln for ln in (links or [])],
            "js_links": [base + ln for ln in (js_links or [])],
        }
        if in_sitemap:
            sitemap.append(url)
        return url

    seed = base + "/"
    pages[seed] = {
        "content": f"{site_id} home index",
        "links": [base + "/s1", base + "/s2", base + "/s3", base + "/changelog"],
        "js_links": [],
    }
    sitemap.append(seed)

    for s in (1, 2, 3):
        add(f"/s{s}", f"{site_id} section {s} overview",
            links=[f"/s{s}/p1"], js_links=[f"/s{s}/js"])
        add(f"/s{s}/p1", f"{site_id} s{s} page1 body TOK-{site_id}-s{s}-p1", links=[f"/s{s}/p2"])
        # s1/p3 carries the fresh tag (a deep freshness target).
        p2_fresh = f" current {site_id}-{FRESH_TAG}" if s == 1 else ""
        add(f"/s{s}/p2", f"{site_id} s{s} page2 body TOK-{site_id}-s{s}-p2", links=[f"/s{s}/p3"])
        add(f"/s{s}/p3", f"{site_id} s{s} page3 body TOK-{site_id}-s{s}-p3{p2_fresh}", links=[])
        # JS-only page: reachable solely via the section's js_links, and NOT listed
        # in the sitemap, so only a JS-following crawler reaches it.
        add(f"/s{s}/js", f"{site_id} s{s} dynamic TOK-{site_id}-s{s}-js", in_sitemap=False)

    add("/changelog", f"{site_id} changelog latest release {site_id}-{FRESH_TAG}", links=[])
    # Sitemap-only pages (nothing links to them): only a sitemap-aware crawl finds them.
    add("/extra1", f"{site_id} appendix TOK-{site_id}-extra1", links=[])
    add("/extra2", f"{site_id} status snapshot {site_id}-{FRESH_TAG}", links=[])
    return seed, pages, sitemap


def _static_hops(seed: str, pages: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Hop distance of each page from the seed over STATIC links only (BFS)."""
    hops = {seed: 0}
    q = deque([seed])
    while q:
        u = q.popleft()
        for ln in pages.get(u, {}).get("links", []):
            if ln not in hops:
                hops[ln] = hops[u] + 1
                q.append(ln)
    return hops


def _targets_for_site(site_id: str, base: str, seed: str, pages: dict[str, dict[str, Any]],
                      hops: dict[str, int]) -> list[dict[str, Any]]:
    """The 14 scored targets per site, spanning all four slice axes."""
    site_type = next(st for sid, _b, st in SITES if sid == site_id)

    def tok(url: str) -> str:
        # The truth_token a crawler must return for this page (a stable fingerprint,
        # or the fresh tag for freshness targets) — taken from the page's content.
        content = pages[url]["content"]
        for piece in content.split():
            if piece.startswith("TOK-") or piece.endswith(FRESH_TAG):
                return piece
        return content.split()[-1]

    # (path, render, stratum, hops_override)
    plan: list[tuple[str, str, str, int | None]] = [
        ("/s1", "static_html", "sentinel", None),
        ("/s2", "static_html", "sentinel", None),
        ("/s3", "static_html", "sentinel", None),
        ("/s1/p1", "static_html", "sentinel", None),
        ("/s2/p1", "static_html", "sentinel", None),
        ("/s1/p2", "static_html", "sentinel", None),     # deep
        ("/s2/p2", "static_html", "sentinel", None),     # deep
        ("/s1/p3", "static_html", "freshness", None),    # deep + fresh
        ("/s1/js", "js_rendered", "sentinel", 2),
        ("/s2/js", "js_rendered", "sentinel", 2),
        ("/s3/js", "js_rendered", "sentinel", 2),
        ("/changelog", "static_html", "freshness", None),
        ("/extra1", "static_html", "sentinel", 1),       # sitemap-only
        ("/extra2", "static_html", "freshness", 1),      # sitemap-only + fresh
    ]

    rows: list[dict[str, Any]] = []
    for i, (path, render, stratum, hops_override) in enumerate(plan, start=1):
        url = base + path
        hop = hops_override if hops_override is not None else hops.get(url, 1)
        rows.append({
            "row_id": f"{site_id}_{i:03d}",
            "seed_id": site_id,
            "seed_url": seed,
            "target_url": url,
            "truth_token": tok(url),
            "render": render,
            "hops": hop,
            "site_type": site_type,
            "stratum": stratum,
            "slices": slice_keys(site_type, render, hop, stratum),
            "ground_truth_tier": "sentinel_planted",
            "source": "controlled-known-structure",
        })
    return rows


def main() -> None:
    out_rows: list[dict[str, Any]] = []
    site_meta: list[dict[str, Any]] = []
    site_graphs: dict[str, dict[str, Any]] = {}

    for site_id, base, _st in SITES:
        seed, pages, sitemap = _build_graph(site_id, base)
        hops = _static_hops(seed, pages)
        rows = _targets_for_site(site_id, base, seed, pages, hops)
        # Attach the inline graph once per seed (on the first target row).
        rows[0]["site"] = {"pages": pages, "sitemap": sitemap}
        rows[0]["max_pages"] = 80
        rows[0]["max_depth"] = 6
        out_rows.extend(rows)
        site_graphs[site_id] = {"pages": len(pages), "sitemap": len(sitemap)}
        site_meta.append({"site_id": site_id, "base": base, "targets": len(rows),
                          "pages": len(pages), "sitemap": len(sitemap)})

    out_path = GOLDEN_CRAWL_DIR / "dev.example.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# {CANARY}  (do not train on this file; see ../CANARY)\n")
        f.write("# CONTROLLED crawl example split — known-structure .example sites with EXACT coverage\n")
        f.write("# ground truth, generated by packages/eval-crawl/tools/build_controlled_set.py.\n")
        f.write("# Network-free + reproducible: this is the split the public snapshot is generated from.\n")
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    # Slice census (for the manifest / audit).
    census: dict[str, int] = {}
    for r in out_rows:
        for sl in r["slices"]:
            census[sl] = census.get(sl, 0) + 1

    MANIFEST.write_text(json.dumps({
        "dataset_version": "crawl-2026.06.controlled.v1",
        "design": "controlled known-structure sites (.example); exact coverage ground truth",
        "total_targets": len(out_rows),
        "sites": site_meta,
        "slice_census": dict(sorted(census.items())),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(out_rows)} targets across {len(SITES)} sites -> {out_path}")
    print(f"wrote selection manifest -> {MANIFEST}")
    print("slice census:")
    for sl, n in sorted(census.items()):
        print(f"  {sl:>28} {n}")


if __name__ == "__main__":
    main()
