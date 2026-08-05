"""URL canonicalization + equivalence-class helpers.

Pre-declared normalization rules ONLY (see plan, must-fix #7). A redirect or
declared canonical may join a row's equivalence class only when it (a) resolves
to the same registrable domain and (b) passes the truth-token check — both of
which are enforced in bench_core.verify, not here. This module only normalizes.
"""
from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import tldextract
from w3lib.url import canonicalize_url

# Tracking params stripped during normalization (closed list, not heuristic).
_TRACKING_PREFIXES = ("utm_", "mc_", "_hs")
_TRACKING_EXACT = {"gclid", "fbclid", "ref", "ref_src", "source", "spm", "amp"}

_extractor = tldextract.TLDExtract(suffix_list_urls=())  # offline; uses bundled snapshot


def registrable_domain(url: str) -> str:
    """eTLD+1, e.g. https://www.sec.gov/x -> 'sec.gov'."""
    ext = _extractor(url)
    return ext.top_domain_under_public_suffix.lower()


def _strip_tracking(query: str) -> str:
    kept = [
        (k, v)
        for k, v in parse_qsl(query, keep_blank_values=True)
        if k.lower() not in _TRACKING_EXACT
        and not any(k.lower().startswith(p) for p in _TRACKING_PREFIXES)
    ]
    return urlencode(kept)


def normalize(url: str) -> str:
    """Deterministic normalization used for set-membership scoring.

    Rules (and ONLY these): lowercase scheme+host, force https where a host
    serves it is NOT assumed (scheme preserved), drop a leading 'www.', strip
    a trailing slash on non-root paths, remove fragments, drop tracking params,
    sort remaining query params.
    """
    url = canonicalize_url(url, keep_fragments=False)
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = parts.hostname.lower() if parts.hostname else ""
    if host.startswith("www."):
        host = host[4:]
    # AMP: amp.example.com and example.com/amp/... are the same document.
    if host.startswith("amp."):
        host = host[4:]
    netloc = host
    if parts.port:
        netloc = f"{host}:{parts.port}"
    path = parts.path
    # strip an AMP path segment (/amp/ or trailing /amp) — same document.
    if path.endswith("/amp"):
        path = path[: -len("/amp")]
    path = path.replace("/amp/", "/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    # Stripping a bare '/amp' empties the path; restore the canonical root '/' so
    # an AMP-root URL keys identically to the plain root. Without this, normalize
    # is not idempotent (normalize('https://x/amp') -> 'https://x' but
    # normalize('https://x/') -> 'https://x/'), and an equivalence class built
    # from an AMP-root golden would never match the same root document.
    if not path:
        path = "/"
    query = _strip_tracking(parts.query)
    # canonicalize_url already sorts query params; re-sort after stripping.
    query = urlencode(sorted(parse_qsl(query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


# Public alias matching the INTERPACKAGE contract name (normalize_url).
normalize_url = normalize


def same_registrable_domain(a: str, b: str) -> bool:
    return registrable_domain(a) == registrable_domain(b) != ""


class EquivalenceClass:
    """A set of normalized URLs that all count as the one correct answer."""

    def __init__(self, canonical: str, members: list[str] | None = None) -> None:
        self.canonical_raw = canonical
        self.canonical = normalize(canonical)
        self._members: set[str] = {self.canonical}
        for m in members or []:
            self._members.add(normalize(m))

    @property
    def domain(self) -> str:
        return registrable_domain(self.canonical_raw)

    def add(self, url: str) -> None:
        self._members.add(normalize(url))

    def contains(self, url: str) -> bool:
        return normalize(url) in self._members

    @property
    def members(self) -> list[str]:
        return sorted(self._members)
