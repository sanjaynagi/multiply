import pytest
from multiply.select.multiplex import Multiplex

# Fixtures
m1 = Multiplex(cost=-1, primer_pairs=["A", "C", "B"])
m2 = Multiplex(cost=-2, primer_pairs=["D", "F", "E"])
m3 = Multiplex(cost=-3, primer_pairs=["A", "B", "D"])
m4 = Multiplex(cost=8, primer_pairs=["A", "C", "B"]) # same pairs as m1, but different cost
ms = [m1, m2, m3, m1, m1, m2, m4]

def test_multiplex_primer_sort():
    """
    Test that the class internally sorts
    primer names
    """
    assert m1.primer_pairs == ("A", "B", "C")
    assert m2.primer_pairs == ("D", "E", "F")

def test_multiplex_unique():
    """
    Test that the class correctly reduces
    to a unique set of multiplexes

    """
    uniq_ms = [m1, m2, m3]
    assert len(set(ms)) == len(uniq_ms)
    assert set(ms) == set(uniq_ms)

def test_multiplex_sort_and_unique():
    ms_sorted = sorted(set(ms))
    assert ms_sorted == [m1, m3, m2]

    

def _greedy(seed):
    import numpy as np

    from multiply.select.selectors import _greedy_search_njit

    rng = np.random.default_rng(0)
    n_targets, per = 12, 4
    n = n_targets * per
    indv = rng.normal(size=n)
    pair = rng.normal(size=(n, n))
    pair = (pair + pair.T) / 2
    flat = np.arange(n, dtype=np.int64)
    starts = np.arange(0, n, per, dtype=np.int64)
    lens = np.full(n_targets, per, dtype=np.int64)
    return _greedy_search_njit(200, seed, flat, starts, lens, indv, pair)


def test_seeded_greedy_search_is_reproducible():
    a_ix, a_cost = _greedy(11)
    b_ix, b_cost = _greedy(11)
    assert (a_ix == b_ix).all() and (a_cost == b_cost).all()


def test_different_seeds_explore_differently():
    assert not (_greedy(1)[0] == _greedy(2)[0]).all()
