import numpy as np
import pytest

from agentgate.stats import bootstrap_mean_ci, cohen_kappa, signflip_pvalue, wilson


def test_wilson_known_values():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.31  # 0/10 -> upper bound ~0.278
    lo, hi = wilson(5, 10)
    assert 0.23 < lo < 0.26 and 0.74 < hi < 0.77
    assert wilson(0, 0) == (0.0, 1.0)


def test_wilson_stays_inside_unit_interval():
    for k, n in [(0, 1), (1, 1), (50, 50), (0, 200)]:
        lo, hi = wilson(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_signflip_obvious_effect_is_significant():
    d = np.full(24, 0.2)
    assert signflip_pvalue(d, np.random.default_rng(0)) < 0.001


def test_signflip_no_effect_and_wrong_direction():
    rng = np.random.default_rng(0)
    assert signflip_pvalue(np.zeros(10), rng) == 1.0
    assert signflip_pvalue(np.full(24, -0.2), rng) > 0.99  # improvement is never a regression


def test_signflip_is_exact_for_tiny_samples():
    # n=3, all positive: only 1 of 8 sign patterns is as extreme -> p ~ 1/8
    p = signflip_pvalue(np.array([1.0, 1.0, 1.0]), np.random.default_rng(1), n_perm=20_000)
    assert 0.10 < p < 0.15


def test_signflip_controls_false_positive_rate_under_the_null():
    """Symmetric noise around zero (an A/A comparison): rejections at alpha=.05 must be ~5%."""
    rng = np.random.default_rng(42)
    hits = 0
    trials = 600
    for _ in range(trials):
        d = rng.normal(0, 0.1, size=24)
        d = d - 0  # symmetric around 0
        hits += signflip_pvalue(d, rng, n_perm=1000) < 0.05
    assert hits / trials < 0.08


def test_bootstrap_ci_contains_the_mean_and_shrinks_with_data():
    rng = np.random.default_rng(0)
    small = rng.normal(0.1, 0.1, 10)
    big = rng.normal(0.1, 0.1, 400)
    lo_s, hi_s = bootstrap_mean_ci(small, rng)
    lo_b, hi_b = bootstrap_mean_ci(big, rng)
    assert lo_s <= small.mean() <= hi_s
    assert (hi_b - lo_b) < (hi_s - lo_s)
    assert bootstrap_mean_ci(np.array([]), rng) == (0.0, 0.0)


def test_kappa():
    a = [True, True, False, False]
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, [not x for x in a]) == -1.0
    assert abs(cohen_kappa([True, False, True, False], [True, True, False, False])) < 1e-9  # chance level
    assert cohen_kappa([True] * 5, [True] * 5) == 1.0  # degenerate: both always agree
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_fisher_lower_pvalue():
    from agentgate.stats import fisher_lower_pvalue

    assert fisher_lower_pvalue(10, 10, 0, 10) < 1e-5  # 100% -> 0%
    assert fisher_lower_pvalue(9, 10, 3, 10) == pytest.approx(0.0099, abs=1e-3)
    assert fisher_lower_pvalue(9, 10, 9, 10) > 0.5  # no change
    assert fisher_lower_pvalue(3, 10, 9, 10) > 0.99  # an improvement is never "lower"
    assert 0.0 <= fisher_lower_pvalue(0, 5, 0, 5) <= 1.0  # degenerate: no passes anywhere


def test_holm_controls_family_wise_error():
    from agentgate.stats import holm_reject

    assert holm_reject([0.001, 0.02, 0.04], 0.05) == [True, True, True]
    assert holm_reject([0.02, 0.03, 0.04], 0.05) == [False, False, False]  # none clears alpha/3 = .0167
    assert holm_reject([0.004, 0.2, 0.3], 0.05) == [True, False, False]
    assert holm_reject([], 0.05) == []
