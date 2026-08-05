"""Tests for the proportion trust-gate stats: Wilson intervals, McNemar's paired
test, and the separability decision the leaderboard depends on.

Wilson is checked against its textbook 95% interval, McNemar against symmetric vs
lopsided discordance, and separable against the alpha threshold. A guarded
cross-check confirms the dependency-light mcnemar() agrees with the
statsmodels-backed mcnemar_pair() on the continuity-corrected path (discordance
>= 25, where both use the corrected chi-square rather than the exact binomial).
"""

import pytest

from bench_stats.proportions import Separability, mcnemar, separable, wilson


def test_wilson_zero_n_is_full_interval():
    t = wilson(0, 0)
    assert t.method == "wilson"
    assert t.n == 0
    assert t.ci_low == 0.0 and t.ci_high == 1.0


def test_wilson_matches_textbook_interval():
    # 10/20 successes -> Wilson 95% CI ~ [0.299, 0.701], symmetric around 0.5.
    t = wilson(10, 20)
    assert t.statistic == pytest.approx(0.5)
    assert t.ci_low == pytest.approx(0.2993, abs=1e-3)
    assert t.ci_high == pytest.approx(0.7007, abs=1e-3)


def test_wilson_clamps_to_unit_interval():
    assert wilson(20, 20).ci_high == 1.0
    assert wilson(0, 20).ci_low == 0.0


def test_wilson_interval_brackets_point():
    t = wilson(7, 25)
    assert t.ci_low <= t.statistic <= t.ci_high


def test_mcnemar_no_discordance_is_nonsignificant():
    t = mcnemar(0, 0)
    assert t.method == "mcnemar"
    assert t.p_value == 1.0
    assert t.n == 0
    assert t.statistic == 0.0


def test_mcnemar_symmetric_is_nonsignificant():
    assert mcnemar(10, 10).p_value > 0.5


def test_mcnemar_lopsided_is_significant():
    assert mcnemar(20, 2).p_value < 0.05


def test_mcnemar_continuity_correction_lowers_chi2():
    # The correction subtracts 1 from |n01 - n10| before squaring, so the
    # corrected statistic is strictly smaller than the uncorrected one.
    corrected = mcnemar(20, 5, continuity=True).statistic
    uncorrected = mcnemar(20, 5, continuity=False).statistic
    assert corrected < uncorrected


def test_separable_when_lopsided():
    s = separable(20, 2)
    assert isinstance(s, Separability)
    assert s.separable is True
    assert s.test.p_value is not None and s.test.p_value < 0.05


def test_not_separable_when_symmetric():
    assert separable(10, 10).separable is False


def test_mcnemar_matches_statsmodels_on_corrected_path():
    pytest.importorskip("statsmodels")
    from bench_stats.reporting import mcnemar_pair

    # Only the corrected chi-square path is comparable; for b + c < 25
    # mcnemar_pair switches to the exact binomial, which mcnemar() does not use.
    for n01, n10 in [(20, 5), (30, 10), (40, 8)]:
        t = mcnemar(n01, n10)
        sm_stat, sm_p = mcnemar_pair(n01, n10)
        assert t.statistic == pytest.approx(sm_stat, rel=1e-6)
        assert t.p_value == pytest.approx(sm_p, rel=1e-6)
