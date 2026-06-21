"""bench-adapters / crawl — crawler systems-under-test.

(seed URL) -> a set of fetched ``{url, content}`` pages. Importing this subpackage
registers all crawl adapters via `@register` side effects:
  bfs-shallow, bfs-deep, sitemap-crawl, render-crawl (free local crawl strategies),
  firecrawl-crawl, tavily-crawl, spider-crawl, apify-crawl (hosted APIs).
"""

from bench_adapters.crawl.adapters import VendorUnavailable

from bench_adapters.crawl import adapters as adapters  # noqa: F401  (registration side effects)

__all__ = ["VendorUnavailable"]
