"""Runner + report tests driving the REAL keyless local crawlers over an inline
link graph (no network, fully deterministic).

The graph is built so the four crawl strategies genuinely separate, exercising the
whole path: run_sync -> per-seed crawl -> per-target score -> ItemResults ->
report.run/collect_counts -> counts.toml. This is the offline, reproducible core
the public snapshot is generated from.
"""
from __future__ import annotations

import tomllib

from eval_crawl import report
from eval_crawl.runner import run_sync

# One controlled site. seed->a->b->c (depth chain); a has a JS-only link to `js`;
# `d` exists only in the sitemap. The crawl strategies differ exactly on these.
SITE = {
    "pages": {
        "https://s.test/": {"content": "home", "links": ["https://s.test/a"]},
        "https://s.test/a": {"content": "alpha AAA", "links": ["https://s.test/b"],
                             "js_links": ["https://s.test/js"]},
        "https://s.test/b": {"content": "bravo BBB", "links": ["https://s.test/c"]},
        "https://s.test/c": {"content": "charlie CCC", "links": []},
        "https://s.test/js": {"content": "jsonly JJJ", "links": []},
        "https://s.test/d": {"content": "sitemaponly DDD", "links": []},
    },
    "sitemap": ["https://s.test/", "https://s.test/a", "https://s.test/b",
                "https://s.test/c", "https://s.test/d"],
}


def _target(rid, url, token, *, render="static_html", hops=1, site_type="docs", stratum="sentinel"):
    return {
        "row_id": rid, "seed_id": "s", "seed_url": "https://s.test/",
        "target_url": url, "truth_token": token,
        "render": render, "hops": hops, "site_type": site_type, "stratum": stratum,
        "ground_truth_tier": "sentinel_planted", "site": SITE, "max_pages": 50, "max_depth": 4,
    }


ROWS = [
    _target("t_a", "https://s.test/a", "AAA", hops=1),                       # shallow static
    _target("t_b", "https://s.test/b", "BBB", hops=2),                       # shallow static (depth 2)
    _target("t_c", "https://s.test/c", "CCC", hops=3),                       # deep static
    _target("t_js", "https://s.test/js", "JJJ", render="js_rendered", hops=1),  # JS-rendered
    _target("t_d", "https://s.test/d", "DDD", hops=1, site_type="gov_registry"),  # sitemap-only
]


def test_run_sync_strategies_separate(tmp_path):
    recs = run_sync(ROWS, "test-run", out_root=str(tmp_path), write_snapshot=False)
    by_adapter: dict[str, dict[str, bool]] = {}
    for r in recs:
        by_adapter.setdefault(r.adapter, {})[r.item_id] = r.output.correct

    # bfs-shallow (depth 1) only reaches the seed + its direct child `a`.
    assert by_adapter["bfs-shallow"]["t_a"] is True
    assert by_adapter["bfs-shallow"]["t_c"] is False   # deep target unreachable at depth 1
    assert by_adapter["bfs-shallow"]["t_js"] is False  # JS-only link

    # bfs-deep (depth 4) walks the static chain to the deep target.
    assert by_adapter["bfs-deep"]["t_c"] is True
    assert by_adapter["bfs-deep"]["t_js"] is False     # still no JS
    assert by_adapter["bfs-deep"]["t_d"] is False      # not statically linked

    # sitemap-crawl reaches the sitemap-only page; render-crawl reaches everything.
    assert by_adapter["sitemap-crawl"]["t_d"] is True
    assert by_adapter["sitemap-crawl"]["t_js"] is False
    assert all(by_adapter["render-crawl"].values()) is True

    for artifact in ("items.jsonl", "slices.jsonl", "manifest.json"):
        assert (tmp_path / "test-run" / artifact).exists()


def test_collect_counts_render_slice(tmp_path):
    recs = run_sync(ROWS, "test-run2", out_root=str(tmp_path), write_snapshot=False)
    counts = report.collect_counts(recs)
    # Only render-crawl covers the js_rendered target.
    js = dict((a, (k, n)) for a, k, n in counts["render:js_rendered"])
    assert js["render-crawl"] == (1, 1)
    assert js["bfs-shallow"] == (0, 1)
    # All four cover the shallow static target `t_a` -> saturated TIE band.
    overall = dict((a, (k, n)) for a, k, n in counts["all"])
    assert overall["render-crawl"] == (5, 5)
    assert overall["bfs-shallow"][0] == 1  # only t_a


def test_counts_toml_is_parseable(tmp_path):
    recs = run_sync(ROWS, "test-run3", out_root=str(tmp_path), write_snapshot=False)
    out = tmp_path / "crawl.counts.toml"
    report.write_counts_toml(recs, out, run_id="test")
    spec = tomllib.loads(out.read_text())
    assert spec["primitive"] == "crawl"
    assert spec["metric_name"] == "page_coverage"
    assert "render:js_rendered" in spec["slices"]
