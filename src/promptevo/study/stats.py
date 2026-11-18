"""Paired statistics over units: the mean difference, a bootstrap interval,
the sign-flip permutation test and Holm's correction."""

import bisect
import random
from typing import NamedTuple


class Paired(NamedTuple):
    mean: float  # of the differences
    low:  float  # 95% bootstrap interval
    high: float
    p:    float  # two-sided sign-flip p-value


def mean(values) -> float:
    """sum over count."""
    values = list(values)
    return sum(values) / len(values)


def paired(diffs: list[float], seed: int = 0) -> Paired:
    """mean of the paired differences, its interval and p-value. p is
    exact up to 40 units."""
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(
        mean(rng.choice(diffs) for _ in range(n)) for _ in range(10000)
    )
    return Paired(mean(diffs), means[250], means[9749], sign_flip_p(diffs))


def sign_flip_p(diffs: list[float]) -> float:
    """two-sided p of the sign-flip test, the share of sign flips at least
    as far from 0 as the observed sum. exact up to 40 units, sampled past
    that."""
    if len(diffs) > 40:
        return _sampled_sign_flip_p(diffs)
    edge = abs(sum(diffs)) - _SLACK
    if edge <= 0:
        return 1.0
    # all 2^n flips without listing them. each half's 2^(n/2) sums get
    # paired up instead, so 40 units takes seconds
    half = len(diffs) // 2
    left = _flip_sums(diffs[:half])
    right = sorted(_flip_sums(diffs[half:]))
    extreme = sum(
        len(right) - bisect.bisect_left(right, edge - s)
        + bisect.bisect_right(right, -edge - s)
        for s in left
    )
    return extreme / (len(left) * len(right))


def holm(pvalues: list[float]) -> list[float]:
    """holm-adjusted p-values, same order."""
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


# keeps exact ties that float rounding would split
_SLACK = 1e-9


# share of random sign flips at least as extreme. the +1 counts the observed
# one, so p is never 0
def _sampled_sign_flip_p(diffs, draws=200000, seed=0):
    rng = random.Random(seed)
    edge = abs(sum(diffs)) - _SLACK
    if edge <= 0:
        return 1.0
    extreme = 0
    for _ in range(draws):
        bits = rng.getrandbits(len(diffs))
        flipped = sum(
            d if bits >> i & 1 else -d for i, d in enumerate(diffs)
        )
        extreme += abs(flipped) >= edge
    return (extreme + 1) / (draws + 1)


def _flip_sums(values):
    sums = [0.0]
    for value in values:
        sums = [s + value for s in sums] + [s - value for s in sums]
    return sums
