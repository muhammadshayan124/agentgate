"""Small, dependency-light statistics used by the gate and the judge calibration.

Design choice that matters: the unit of analysis is the *task*, not the trial. Trials of the
same task are correlated, so treating 240 trials as 240 independent observations would make
every confidence interval far too narrow. We reduce each task to one pass rate and compare
the *paired* per-task differences between two runs.
"""

from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (well behaved near 0 and 1)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def signflip_pvalue(d: np.ndarray, rng: np.random.Generator, n_perm: int = 10_000) -> float:
    """One-sided paired randomisation test of H0: the mean of d is <= 0, against mean(d) > 0.

    Under H0 with symmetric noise each per-task difference is as likely to be positive as
    negative, so randomly flipping signs gives the null distribution of the mean. The test is
    exact in size, with no normality assumption, which matters with only a few dozen tasks.
    """
    d = np.asarray(d, dtype=float)
    if d.size == 0 or not np.any(d):
        return 1.0
    obs = d.mean()
    signs = rng.integers(0, 2, size=(n_perm, d.size), dtype=np.int8) * 2 - 1
    null = (signs @ d) / d.size
    return float((1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1))


def bootstrap_mean_ci(
    d: np.ndarray, rng: np.random.Generator, alpha: float = 0.05, n_boot: int = 5_000
) -> tuple[float, float]:
    """Percentile bootstrap, two-sided (1 - alpha) interval for the mean difference."""
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return 0.0, 0.0
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Chance-corrected agreement between two binary raters."""
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError("need two equal-length, non-empty label lists")
    agree = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1.0:
        return 1.0 if agree == 1.0 else 0.0
    return (agree - expected) / (1 - expected)


def fisher_lower_pvalue(pass_a: int, n_a: int, pass_b: int, n_b: int) -> float:
    """One-sided exact (Fisher) p-value that arm B's pass probability is LOWER than arm A's.

    Hypergeometric tail: given the total number of passes, how likely is it that B got this few?
    Exact for the small counts (a handful of trials) we actually have, no normal approximation.
    """
    total, passes = n_a + n_b, pass_a + pass_b
    denom = math.comb(total, n_b)
    lo = max(0, n_b - (total - passes))
    return min(1.0, sum(math.comb(passes, x) * math.comb(total - passes, n_b - x) for x in range(lo, pass_b + 1)) / denom)


def holm_reject(pvalues: list[float], alpha: float) -> list[bool]:
    """Holm-Bonferroni step-down: controls the chance of ANY false rejection at alpha."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    reject = [False] * len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (len(pvalues) - rank):
            reject[i] = True
        else:
            break
    return reject
