"""Crawl scoring: a seed-crawl result, one golden TARGET page -> ScorerOutput.

The model: hand a crawler a SEED url; it discovers and fetches pages. For each
golden target page reachable under that seed, the cell is a HIT iff the crawler
**(a) reached the page** (coverage) **and (b) returned its current content** such
that the target's `truth_token` survives in the fetched page (freshness/fidelity).
That single binary — `page_coverage` — is the leaderboard's paired-binary
separability surface (the same stance as websearch hit_rate, extraction
token_survival, reranker hit@1): McNemar/Wilson gate it per slice.

Coverage and freshness are the two canonical axes of crawl quality (Olston &
Najork, "Web Crawling," Foundations and Trends in IR, 2010; freshness/age formally
defined by Cho & Garcia-Molina, SIGMOD 2000 / ACM TODS 2003). We keep BOTH legible
by decomposing every miss:

  * not_reached  — the crawler never returned this page at all. THE COVERAGE GAP:
                   discovery failed (depth budget too shallow, no JS render, page
                   not linked and not in the sitemap). This is the failure mode
                   extraction can never see, because extraction is handed the URL.
  * stale        — the page WAS reached, but the fetched content does not carry the
                   current `truth_token` on a freshness target (the crawler served
                   a stale/cached copy). THE FRESHNESS GAP.
  * token_absent — reached, real content returned, the token simply is not in it.
  * blocked      — the returned "content" is an anti-bot interstitial (a different
                   capability gap than freshness/coverage — never charged as one).
  * empty        — reached the URL but returned no usable content.

`metrics` carries the two axes separately — `url_discovered` (pure coverage /
URL-recall, 0/1) and `content_fresh` (0/1) — so an analyst can read the coverage
gap apart from the freshness gap. Non-attempts (crawl error) are uncharged
(`correct=None`), never failures.
"""
from __future__ import annotations

import re
from typing import Any

from bench_core.urls import EquivalenceClass
from bench_schemas import ScorerOutput

_WS = re.compile(r"\s+")

# Anti-bot / access-wall interstitials (shared vocabulary with the extraction
# vertical). A "miss" because the crawler got firewalled is a DIFFERENT capability
# gap than failing to reach or refresh a page, so it is classified separately.
_BLOCK_SIGNATURES = (
    "request access", "aggressive automated scraping", "just a moment",
    "attention required", "verify you are human", "are you a robot",
    "captcha", "access denied", "enable javascript", "cloudflare",
    "unusual traffic", "403 forbidden",
)


def _norm(text: str) -> str:
    return _WS.sub(" ", text or "")


def token_survives(token: str, text: str) -> bool:
    """Whitespace-normalized substring survival (FR doc numbers / CVE ids / version
    tags are exact strings). Tolerates internal-whitespace differences."""
    if not text or not token:
        return False
    norm = _norm(text)
    if token in norm:
        return True
    return _WS.sub("", token) in _WS.sub("", text)


def _matched_page(eq: EquivalenceClass, raw: dict[str, Any]) -> tuple[bool, str | None]:
    """(url_discovered, content) for the first crawled page in the target's class.

    `url_discovered` is true if any returned URL is in the equivalence class, even
    when no page body is attached (a map-only / URL-discovery crawler). `content`
    is the matched page body, or None when the URL was discovered without content.
    """
    for page in raw.get("pages") or []:
        url = str(page.get("url", ""))
        if url and eq.contains(url):
            return True, str(page.get("content", ""))
    for url in raw.get("returned_urls") or []:
        if url and eq.contains(str(url)):
            return True, None  # discovered, but no content body attached
    return False, None


def _classify_content_miss(content: str | None, *, freshness: bool) -> str:
    """Decompose a reached-but-token-missing page into a miss reason."""
    if content is None or not content.strip():
        return "empty"
    low = content.lower()
    if any(sig in low for sig in _BLOCK_SIGNATURES):
        return "blocked"
    # On a freshness target, reached-without-the-current-token means a stale fetch.
    return "stale" if freshness else "token_absent"


def score_crawl(item: dict[str, Any], raw: dict[str, Any]) -> ScorerOutput:
    """Score one (crawler, golden target) cell -> ScorerOutput.

    `item` carries at least `truth_token` and `target_url` (+ optional
    `equivalence_members`, and `stratum` to mark a freshness target). `raw` is the
    crawler's result dict for the target's SEED (`pages` = ``[{url, content}]``,
    `returned_urls`). A crawl-level error short-circuits to an uncharged non-attempt.
    """
    if raw.get("error"):
        return ScorerOutput(correct=None, miss_reason="crawl_failed",
                            rationale=str(raw["error"])[:200])

    token = str(item.get("truth_token", ""))
    target_url = str(item.get("target_url") or item.get("url") or "")
    members = [str(m) for m in (item.get("equivalence_members") or [])]
    freshness = str(item.get("stratum", "")) == "freshness"
    eq = EquivalenceClass(target_url, members) if target_url else EquivalenceClass(target_url)

    url_discovered, content = _matched_page(eq, raw)

    if not url_discovered:
        # Coverage failure: the crawler never surfaced this page from the seed.
        return ScorerOutput(
            correct=False, score=0.0, miss_reason="not_reached",
            metrics={"url_discovered": 0.0, "content_fresh": 0.0, "chars": 0.0},
        )

    fresh = token_survives(token, content or "")
    if fresh:
        return ScorerOutput(
            correct=True, score=1.0, miss_reason=None,
            metrics={"url_discovered": 1.0, "content_fresh": 1.0,
                     "chars": float(len(content or ""))},
        )

    miss = _classify_content_miss(content, freshness=freshness)
    return ScorerOutput(
        correct=False, score=0.0, miss_reason=miss,
        metrics={"url_discovered": 1.0, "content_fresh": 0.0,
                 "chars": float(len(content or "")) if content else 0.0},
    )
