"""Tests for bench_core.urls normalization and equivalence classes.

normalize() is the set-membership key behind hit@k scoring, so its central
invariant is idempotence: normalize(normalize(u)) == normalize(u). If that fails,
two spellings of the same document hash to different keys and a correct vendor
answer is scored a miss. These are offline (tldextract uses the bundled snapshot).
"""

import pytest

from bench_core.urls import EquivalenceClass, normalize

# Spellings that should all reduce to the same canonical document.
_IDEMPOTENCE_CORPUS = [
    "https://example.com/amp",
    "https://example.com/foo/amp",
    "https://example.com/amp/page",
    "https://example.com/path/",
    "https://www.example.com/a?utm_source=x&b=2#frag",
    "https://amp.example.com/x",
    "https://example.com/",
    "https://example.com",
    "http://example.com/Y",
    "https://example.com/a/b/",
]


@pytest.mark.parametrize("url", _IDEMPOTENCE_CORPUS)
def test_normalize_is_idempotent(url):
    once = normalize(url)
    assert normalize(once) == once


def test_amp_root_keys_same_as_plain_root():
    # The regression this fixes: '/amp' stripped to an empty path must restore
    # the canonical root '/', so all three spellings collapse to one key.
    root = normalize("https://example.com/")
    assert normalize("https://example.com/amp") == root
    assert normalize("https://example.com") == root


def test_host_and_path_amp_collapse_to_same_document():
    canonical = normalize("https://example.com/path")
    assert normalize("https://www.example.com/path") == canonical
    assert normalize("https://amp.example.com/path") == canonical
    assert normalize("https://example.com/amp/path") == canonical


def test_trailing_slash_stripped_on_nonroot_but_kept_on_root():
    assert normalize("https://example.com/path/") == normalize("https://example.com/path")
    assert normalize("https://example.com/").endswith("/")


def test_fragment_is_dropped():
    assert "#" not in normalize("https://example.com/a#section")


def test_tracking_params_stripped_and_real_params_kept():
    out = normalize("https://example.com/x?utm_source=a&b=2&a=1&gclid=z")
    assert "utm_source" not in out and "gclid" not in out
    assert "a=1" in out and "b=2" in out


def test_equivalence_class_amp_root_matches_plain_root():
    eq = EquivalenceClass("https://site.com/amp")
    assert eq.contains("https://site.com/")
    assert eq.contains("https://site.com")
    assert eq.contains("https://www.site.com")


def test_equivalence_class_plain_root_matches_amp_root():
    eq = EquivalenceClass("https://site.com/")
    assert eq.contains("https://site.com/amp")


def test_equivalence_class_excludes_other_paths():
    eq = EquivalenceClass("https://site.com/a", ["https://site.com/b"])
    assert eq.contains("https://site.com/a")
    assert eq.contains("https://site.com/b")
    assert not eq.contains("https://site.com/c")
