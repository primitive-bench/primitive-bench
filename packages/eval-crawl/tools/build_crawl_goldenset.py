"""Generate the LIVE crawl dev split (dev.jsonl) from real seed sites (NOT committed).

The controlled split (build_controlled_set.py) gives reproducible, network-free
coverage truth; THIS builder produces the live split the hosted crawl vendors
(firecrawl-crawl, tavily-crawl, spider-crawl, apify-crawl) are scored on, sampled
from real sites with the **sitemap as the authoritative page registry** (the
webmaster's own declaration of what pages exist — ground_truth_tier
`authoritative_registry`).

For each seed in `seeds.toml`:
  1. Pull the gold target set from the seed's ``sitemap.xml`` when present, else
     from a bounded static BFS of the seed.
  2. Measure each target's hop distance from the seed by a bounded static BFS
     (so the `depth` slice is *measured*, not guessed); fall back to URL path depth.
  3. Pick a stable ``truth_token`` from the fetched page (a salient title token /
     the URL slug; for a freshness target, a detected version/date that changes
     over time) and VERIFY it via the liveness gate — the token must appear in the
     fetched main content, or the target is dropped (never scored on a stale gold).
  4. Stratify-sample to `--per-seed` targets balanced across depth, and tag slices
     deterministically (render comes from the seed's known mode in seeds.toml).

Determinism (D-14): a seeded sample over the verified, sitemap-derived target pool;
the chosen targets + seed are written to `selection_manifest.json` so the subset is
auditable, and the recipe is pinned in `Task.dataset_version`.

Run (needs network; writes a git-ignored dev.jsonl):
    uv run python packages/eval-crawl/tools/build_crawl_goldenset.py --per-seed 16 --seed 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import tomllib
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin

from bench_core.http import extract_main_text, get_with_retry, make_client, BROWSER_UA
from bench_core.urls import normalize_url, registrable_domain
from bench_core.verify import liveness_gate
from selectolax.parser import HTMLParser

from eval_crawl._paths import CANARY, GOLDEN_CRAWL_DIR
from eval_crawl.slicing import DEEP_HOPS, slice_keys

SEEDS_TOML = Path(__file__).resolve().parent / "seeds.toml"
MANIFEST = Path(__file__).resolve().parents[1] / "selection_manifest.live.json"

# A token that changes over time -> a freshness target (semantic version or ISO date).
_FRESH_RE = re.compile(r"\b(v?\d+\.\d+(?:\.\d+)?|20\d{2}-\d{2}-\d{2})\b")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{4,}")


async def _fetch(client, url: str) -> tuple[str, str]:
    """(raw_html, main_text) for a URL, or ('','') on failure."""
    try:
        r = await get_with_retry(client, url)
    except Exception:
        return "", ""
    if r.status_code >= 400:
        return "", ""
    html = r.text if "html" in r.headers.get("content-type", "") else ""
    return html, (extract_main_text(html) if html else r.text)


def _links(html: str, base_url: str, domain: str) -> list[str]:
    out: list[str] = []
    for a in HTMLParser(html).css("a[href]"):
        href = a.attributes.get("href")
        if not href or href.startswith(("mailto:", "javascript:", "#", "tel:")):
            continue
        u = urldefrag(urljoin(base_url, href))[0]
        if u.startswith(("http://", "https://")) and registrable_domain(u) == domain:
            out.append(u)
    return out


async def _sitemap_targets(client, seed_url: str, domain: str) -> list[str]:
    try:
        r = await get_with_retry(client, urljoin(seed_url, "/sitemap.xml"))
    except Exception:
        return []
    if r.status_code >= 400:
        return []
    urls = [loc.text(strip=True) for loc in HTMLParser(r.text).css("loc")]
    return [u for u in urls if u and registrable_domain(u) == domain]


async def _bfs_hops(client, seed_url: str, domain: str, max_pages: int, max_depth: int) -> dict[str, int]:
    """Measured static hop distance from the seed for reachable pages (bounded BFS)."""
    hops = {normalize_url(seed_url): 0}
    raw = {normalize_url(seed_url): seed_url}
    q: deque[tuple[str, int]] = deque([(seed_url, 0)])
    n = 0
    while q and n < max_pages:
        url, depth = q.popleft()
        n += 1
        html, _ = await _fetch(client, url)
        if not html or depth >= max_depth:
            continue
        for link in _links(html, url, domain):
            key = normalize_url(link)
            if key not in hops:
                hops[key] = depth + 1
                raw[key] = link
                q.append((link, depth + 1))
        await asyncio.sleep(0.2)
    return {raw[k]: v for k, v in hops.items()}


def _pick_token(url: str, main_text: str, freshness: bool) -> str | None:
    """A stable truth_token present in the fetched main content (or None)."""
    if freshness:
        m = _FRESH_RE.search(main_text)
        if m:
            return m.group(0)
    # URL slug, if it shows up verbatim in the content.
    slug = [p for p in url.rstrip("/").split("/") if p][-1:]
    norm = re.sub(r"\s+", " ", main_text)
    if slug and slug[0] in norm:
        return slug[0]
    # else the longest distinctive word in the first chunk of main content.
    words = sorted(set(_WORD_RE.findall(norm[:600])), key=len, reverse=True)
    return words[0] if words else None


def _stratified(pool: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in pool:
        groups["deep" if r["hops"] >= DEEP_HOPS else "shallow"].append(r)
    for g in groups.values():
        g.sort(key=lambda x: x["target_url"])
        rng.shuffle(g)
    keys = sorted(groups)
    idx: dict[str, int] = {k: 0 for k in groups}
    chosen: list[dict[str, Any]] = []
    while len(chosen) < n and any(idx[k] < len(groups[k]) for k in keys):
        for k in keys:
            if idx[k] < len(groups[k]):
                chosen.append(groups[k][idx[k]])
                idx[k] += 1
                if len(chosen) >= n:
                    break
    return chosen


async def _build_seed(client, seed: dict[str, Any], per_seed: int, seed_num: int) -> list[dict[str, Any]]:
    seed_url, site_type, render = seed["seed_url"], seed["site_type"], seed["render"]
    domain = registrable_domain(seed_url)
    max_pages, max_depth = int(seed.get("max_pages", 60)), int(seed.get("max_depth", 5))

    hops = await _bfs_hops(client, seed_url, domain, max_pages, max_depth)
    targets = await _sitemap_targets(client, seed_url, domain) or list(hops)
    targets = [t for t in targets if normalize_url(t) != normalize_url(seed_url)]

    pool: list[dict[str, Any]] = []
    for t in targets:
        _, main = await _fetch(client, t)
        if not main:
            continue
        freshness = bool(_FRESH_RE.search(main))
        token = _pick_token(t, main, freshness)
        if not token:
            continue
        lv = await liveness_gate({"golden_url": t, "truth_token": token, "source": domain})
        if not lv.live:
            continue
        hop = hops.get(t, len([p for p in t.rstrip("/").split("/") if p]) - 2)
        stratum = "freshness" if freshness else "coverage"
        pool.append({
            "seed_id": f"{domain}::{seed_url}",
            "seed_url": seed_url,
            "target_url": (lv.payload or {}).get("canonical", t),
            "equivalence_members": (lv.payload or {}).get("members", [t]),
            "truth_token": token,
            "token_depth": (lv.payload or {}).get("token_depth", -1),
            "render": render,
            "hops": max(0, int(hop)),
            "site_type": site_type,
            "stratum": stratum,
            "ground_truth_tier": "authoritative_registry",
            "source": domain,
            "max_pages": max_pages,
            "max_depth": max_depth,
        })
        await asyncio.sleep(0.1)

    chosen = _stratified(pool, per_seed, seed_num)
    for i, row in enumerate(chosen, start=1):
        row["row_id"] = f"{domain.replace('.', '_')}_{i:03d}"
        row["slices"] = slice_keys(row["site_type"], row["render"], row["hops"], row["stratum"])
        row["canary"] = CANARY
    print(f"  {domain}: {len(chosen)}/{len(pool)} verified targets sampled")
    return chosen


async def _main_async(args) -> None:
    seeds = tomllib.loads(SEEDS_TOML.read_text())["seed"]
    out_rows: list[dict[str, Any]] = []
    async with make_client({"User-Agent": BROWSER_UA}) as client:
        for i, seed in enumerate(seeds):
            print(f"building {seed['seed_url']} ...")
            out_rows.extend(await _build_seed(client, seed, args.per_seed, args.seed + i))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# {CANARY}  (do not train on this file; see ../CANARY)\n")
        f.write("# LIVE crawl dev split — sampled from real sitemaps; NOT redistributed (see README).\n")
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    census: dict[str, int] = defaultdict(int)
    for r in out_rows:
        for sl in r["slices"]:
            census[sl] += 1
    MANIFEST.write_text(json.dumps({
        "dataset_version": f"crawl-2026.06.live.n{len(out_rows)}.seed{args.seed}",
        "per_seed": args.per_seed, "seed": args.seed, "total_targets": len(out_rows),
        "seeds": [s["seed_url"] for s in seeds], "slice_census": dict(sorted(census.items())),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(out_rows)} live targets -> {out_path}")
    print(f"wrote live selection manifest -> {MANIFEST}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-seed", type=int, default=16, help="targets sampled per seed site")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(GOLDEN_CRAWL_DIR / "dev.jsonl"))
    args = ap.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
