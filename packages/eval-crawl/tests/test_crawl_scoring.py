"""Unit tests for crawl scoring (coverage + freshness + miss decomposition)."""
from __future__ import annotations

from eval_crawl.scoring import score_crawl

TARGET = {
    "truth_token": "v8.4.0",
    "target_url": "https://docs.test/changelog",
    "stratum": "coverage",
}


def _crawl(*pages: tuple[str, str]) -> dict:
    return {"pages": [{"url": u, "content": c} for u, c in pages],
            "returned_urls": [u for u, _ in pages]}


def test_reached_and_fresh_is_a_hit():
    out = score_crawl(TARGET, _crawl(("https://docs.test/changelog", "Release v8.4.0 shipped")))
    assert out.correct is True
    assert out.score == 1.0
    assert out.metrics["url_discovered"] == 1.0
    assert out.metrics["content_fresh"] == 1.0


def test_not_reached_is_the_coverage_gap():
    out = score_crawl(TARGET, _crawl(("https://docs.test/other", "unrelated")))
    assert out.correct is False
    assert out.miss_reason == "not_reached"
    assert out.metrics["url_discovered"] == 0.0


def test_reached_but_token_absent_on_coverage_target():
    out = score_crawl(TARGET, _crawl(("https://docs.test/changelog", "Release v8.3.0 (older)")))
    assert out.correct is False
    assert out.miss_reason == "token_absent"
    assert out.metrics["url_discovered"] == 1.0
    assert out.metrics["content_fresh"] == 0.0


def test_reached_but_stale_on_freshness_target():
    fresh_target = {**TARGET, "stratum": "freshness"}
    out = score_crawl(fresh_target, _crawl(("https://docs.test/changelog", "Release v8.3.0 (older)")))
    assert out.correct is False
    assert out.miss_reason == "stale"  # freshness target reached with a stale copy


def test_blocked_interstitial_is_classified_separately():
    out = score_crawl(TARGET, _crawl(("https://docs.test/changelog", "Just a moment... Cloudflare")))
    assert out.correct is False
    assert out.miss_reason == "blocked"


def test_discovered_without_content_is_empty():
    raw = {"pages": [], "returned_urls": ["https://docs.test/changelog"]}
    out = score_crawl(TARGET, raw)
    assert out.correct is False
    assert out.miss_reason == "empty"
    assert out.metrics["url_discovered"] == 1.0  # URL recalled, content not delivered


def test_crawl_error_is_uncharged():
    out = score_crawl(TARGET, {"error": "boom"})
    assert out.correct is None and out.miss_reason == "crawl_failed"


def test_equivalence_member_alias_matches():
    target = {**TARGET, "equivalence_members": ["https://docs.test/changelog/"]}
    # crawler returned the trailing-slash alias; normalization makes them equal
    out = score_crawl(target, _crawl(("https://docs.test/changelog/", "Now at v8.4.0")))
    assert out.correct is True
