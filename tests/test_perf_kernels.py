"""The jitted kernels must agree with the straightforward implementations they replace."""

import numpy as np
import pytest

from multiply.align.algorithms import (
    PrimerDimerLike,
    _pairwise_scores_njit,
    encode_primers_for_pairwise,
)
from multiply.select.selectors import _greedy_search_njit


def _random_primers(n, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list("ACGT"), size=int(rng.integers(18, 28)))) for _ in range(n)]


def test_pairwise_score_kernel_matches_the_python_aligner():
    primers = _random_primers(24, seed=1)
    model = PrimerDimerLike()
    model.load_parameters()
    fwd, rev, lengths = encode_primers_for_pairwise(primers)
    fast = _pairwise_scores_njit(fwd, rev, lengths, model._nn_lut, model.end_length, model.end_bonus)

    assert fast.shape == (24, 24)
    assert np.allclose(fast, fast.T)
    for i in range(24):
        for j in range(i, 24):
            model.set_primers(primers[i], primers[j], f"p{i}", f"p{j}")
            model.align()
            assert fast[i, j] == pytest.approx(model.score), (primers[i], primers[j])


def test_greedy_search_costs_equal_the_cost_of_the_multiplex_it_returns():
    """The incremental cost update must equal the cost computed from scratch."""
    rng = np.random.default_rng(0)
    n_targets, per = 10, 4
    n = n_targets * per
    indv = rng.normal(size=n)
    pairwise = rng.normal(size=(n, n))
    pairwise = (pairwise + pairwise.T) / 2
    flat = np.arange(n, dtype=np.int64)
    starts = np.arange(0, n, per, dtype=np.int64)
    lens = np.full(n_targets, per, dtype=np.int64)

    chosen, costs = _greedy_search_njit(50, flat, starts, lens, indv, pairwise)

    for row, cost in zip(chosen, costs, strict=True):
        assert len(set(row)) == n_targets                      # one pair per target
        explicit = indv[row].sum() + pairwise[np.ix_(row, row)].sum()
        assert cost == pytest.approx(explicit)
