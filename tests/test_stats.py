"""The paired statistics behind every reported p-value and interval."""

import itertools
import random

import pytest

from promptevo.study.stats import (
    _sampled_sign_flip_p,
    holm,
    paired,
    sign_flip_p,
)


# every sign flip listed out, the slow way the fast one has to match
def _brute_force_p(diffs):
    edge = abs(sum(diffs)) - 1e-9
    flips = list(itertools.product([1, -1], repeat=len(diffs)))
    extreme = sum(
        abs(sum(s * d for s, d in zip(signs, diffs))) >= edge
        for signs in flips
    )
    return extreme / len(flips)


@pytest.mark.parametrize("n", [1, 2, 5, 8, 11])
def test_sign_flip_matches_brute_force(n):
    rng = random.Random(n)
    diffs = [round(rng.uniform(-0.1, 0.15), 3) for _ in range(n)]
    assert sign_flip_p(diffs) == pytest.approx(_brute_force_p(diffs))


def test_sign_flip_with_ties_and_zeros():
    diffs = [0.05, -0.05, 0.0, 0.05, 0.1, 0.0]
    assert sign_flip_p(diffs) == pytest.approx(_brute_force_p(diffs))


def test_all_zero_diffs_give_p_one():
    assert sign_flip_p([0.0] * 6) == 1.0
    assert paired([0.0] * 6).p == 1.0


def test_unanimous_units_give_the_smallest_p():
    assert sign_flip_p([0.01 * (i + 1) for i in range(10)]) == 2 / 2**10


def test_sampled_p_is_close_to_exact():
    rng = random.Random(3)
    diffs = [rng.gauss(0.02, 0.04) for _ in range(14)]
    exact = sign_flip_p(diffs)
    assert _sampled_sign_flip_p(diffs, draws=50000) == pytest.approx(
        exact, abs=0.01
    )


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_paired_interval_brackets_the_mean():
    diffs = [0.03, 0.01, -0.02, 0.05, 0.04, 0.0, 0.02, 0.06]
    mean, low, high, _ = paired(diffs)
    assert mean == pytest.approx(sum(diffs) / len(diffs))
    assert low < mean < high
    # same seed, same interval
    assert paired(diffs) == paired(diffs)
