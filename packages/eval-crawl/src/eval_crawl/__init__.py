"""eval-crawl — Primitive Bench vertical (site coverage & freshness).

Hand a crawler a SEED url; it discovers and fetches the pages under it. For each
golden TARGET page the cell is a hit iff the crawler **reached** it (coverage) AND
returned its **current content** so the `truth_token` survives (freshness). That
binary — `page_coverage` — is the McNemar/Wilson separability surface, so per-slice
winners (or TIE bands), never one global ranking. Coverage and freshness are the
two canonical axes of crawl quality (Olston & Najork 2010; Cho & Garcia-Molina
2000/2003); every miss is decomposed (not_reached = coverage gap; stale = freshness
gap; plus blocked/token_absent/empty) so a low score stays legible.

Implements `bench_core.Task` + `bench_core.Scorer`. The public DEV split lives in
golden-sets-public/crawl/ (a controlled offline example committed here + a live
sitemap-sampled generator); the held-out test split lives only behind the private
eval server. Slices: slices.yaml (render / depth / site_type / freshness).
"""

from eval_crawl.task import Scorer, Task
from eval_crawl.scoring import score_crawl, token_survives
from eval_crawl.runner import run_sync
from eval_crawl import report

__all__ = ["Task", "Scorer", "score_crawl", "token_survives", "run_sync", "report"]
