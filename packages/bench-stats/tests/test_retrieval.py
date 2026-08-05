"""Tests for the BEIR/MTEB-standard IR metrics in `bench_stats.retrieval`.

These metrics back the search / retrieval / reranker verticals, so they are
checked against hand-computable and canonical reference values (the Wikipedia DCG
worked example for nDCG, TREC definitions for the set metrics) rather than against
their own implementation. The module is scipy-free, so nothing here is skipped.
"""

import pytest

from bench_stats import (
    hit_at_k,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
)


# --------------------------------------------------------------------------- #
# hit@k
# --------------------------------------------------------------------------- #
def test_hit_at_k_relevant_inside_k():
    assert hit_at_k([0, 0, 1, 0], 3) == 1.0


def test_hit_at_k_relevant_beyond_k_is_miss():
    # The only relevant item sits at rank 3, outside the top-2 window.
    assert hit_at_k([0, 0, 1, 0], 2) == 0.0


def test_hit_at_k_graded_relevance_counts_as_hit():
    assert hit_at_k([0, 2, 0], 2) == 1.0


def test_hit_at_k_empty_is_miss():
    assert hit_at_k([], 5) == 0.0


# --------------------------------------------------------------------------- #
# mrr@k
# --------------------------------------------------------------------------- #
def test_mrr_at_k_first_relevant_rank():
    assert mrr_at_k([0, 0, 1], 3) == pytest.approx(1 / 3)


def test_mrr_at_k_top_rank_is_one():
    assert mrr_at_k([1, 0, 0], 3) == 1.0


def test_mrr_at_k_relevant_beyond_k_is_zero():
    # Relevant item at rank 2, but k=1 only inspects the first result.
    assert mrr_at_k([0, 1], 1) == 0.0


def test_mrr_at_k_no_relevant_is_zero():
    assert mrr_at_k([0, 0, 0], 3) == 0.0


# --------------------------------------------------------------------------- #
# nDCG@k
# --------------------------------------------------------------------------- #
def test_ndcg_at_k_perfect_ranking_is_one():
    # Already in ideal (descending) order -> DCG == IDCG.
    assert ndcg_at_k([3, 2, 1], 3) == pytest.approx(1.0)


def test_ndcg_at_k_wikipedia_worked_example():
    # Canonical DCG example (Wikipedia "Discounted cumulative gain"):
    # relevances [3,2,3,0,1,2] over 6 ranks -> nDCG@6 ~= 0.9608.
    assert ndcg_at_k([3, 2, 3, 0, 1, 2], 6) == pytest.approx(0.9608, abs=1e-4)


def test_ndcg_at_k_all_irrelevant_is_zero():
    # IDCG == 0, so the metric is defined to 0.0 rather than dividing by zero.
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_at_k_rewards_better_ordering():
    good = ndcg_at_k([3, 1, 0], 3)
    bad = ndcg_at_k([0, 1, 3], 3)
    assert good > bad


# --------------------------------------------------------------------------- #
# MAP@k
# --------------------------------------------------------------------------- #
def test_map_at_k_known_value_with_corpus_total():
    # Relevant at ranks 1,3,5 with 3 relevant in the corpus:
    # AP = (1/1 + 2/3 + 3/5) / 3 = 0.75556.
    assert map_at_k([1, 0, 1, 0, 1], k=5, n_relevant=3) == pytest.approx(0.755556, abs=1e-5)


def test_map_at_k_perfect_ranking_is_one():
    assert map_at_k([1, 1, 1], k=3, n_relevant=3) == pytest.approx(1.0)


def test_map_at_k_no_relevant_is_zero():
    assert map_at_k([0, 0, 0], k=3, n_relevant=3) == 0.0


def test_map_at_k_default_denominator_uses_hits_found():
    # Without n_relevant the denominator is the number of relevant items found in
    # top-k, so [1,0,1] averages precisions at ranks 1 and 3: (1/1 + 2/3) / 2.
    assert map_at_k([1, 0, 1], k=3) == pytest.approx((1.0 + 2 / 3) / 2)


# --------------------------------------------------------------------------- #
# precision@k
# --------------------------------------------------------------------------- #
def test_precision_at_k_basic():
    assert precision_at_k([1, 0, 1, 0, 1], 5) == pytest.approx(0.6)
    assert precision_at_k([1, 0, 1, 0, 1], 3) == pytest.approx(2 / 3)


def test_precision_at_k_charges_for_empty_slots():
    # Only 2 items returned but k=5 -> denominator stays 5 (TREC convention).
    assert precision_at_k([1, 1], 5) == pytest.approx(0.4)


def test_precision_at_k_graded_relevance():
    assert precision_at_k([2, 0, 3], 3) == pytest.approx(2 / 3)


def test_precision_at_k_zero_k_is_zero():
    assert precision_at_k([1, 1, 1], 0) == 0.0


# --------------------------------------------------------------------------- #
# recall@k
# --------------------------------------------------------------------------- #
def test_recall_at_k_partial():
    # 3 relevant retrieved out of 4 in the corpus.
    assert recall_at_k([1, 0, 1, 0, 1], k=5, n_relevant=4) == pytest.approx(0.75)


def test_recall_at_k_window_limits_what_counts():
    # Same list, k=3: only ranks 1 and 3 are relevant -> 2 of 4.
    assert recall_at_k([1, 0, 1, 0, 1], k=3, n_relevant=4) == pytest.approx(0.5)


def test_recall_at_k_does_not_credit_unsurfaced_items():
    # All 2 returned are relevant, but the corpus has 5 relevant -> recall 0.4,
    # never 1.0. This is the honesty property: missed items still count against it.
    assert recall_at_k([1, 1], k=10, n_relevant=5) == pytest.approx(0.4)


def test_recall_at_k_no_relevant_corpus_is_zero():
    assert recall_at_k([0, 0], k=2, n_relevant=0) == 0.0


# --------------------------------------------------------------------------- #
# R-precision
# --------------------------------------------------------------------------- #
def test_r_precision_known_value():
    # R = 3; the top-3 [1,0,1] hold 2 relevant -> 2/3.
    assert r_precision([1, 0, 1, 0, 1], n_relevant=3) == pytest.approx(2 / 3)


def test_r_precision_coincides_with_precision_and_recall_at_r():
    rel = [1, 0, 1, 0, 1, 0]
    r = 3
    assert r_precision(rel, r) == pytest.approx(precision_at_k(rel, r))
    assert r_precision(rel, r) == pytest.approx(recall_at_k(rel, k=r, n_relevant=r))


def test_r_precision_perfect_when_all_relevant_lead():
    assert r_precision([1, 1, 1, 0, 0], n_relevant=3) == pytest.approx(1.0)


def test_r_precision_no_relevant_is_zero():
    assert r_precision([0, 0, 0], n_relevant=0) == 0.0
