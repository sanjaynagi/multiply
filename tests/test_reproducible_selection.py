import numpy as np
import pandas as pd

from multiply.select.selectors import GreedySearch, _greedy_search_njit


class _StubCostFunction:
    """The few attributes GreedySearch reads from a cost function."""

    def __init__(self, n_targets, per_target, seed=0):
        rng = np.random.default_rng(seed)
        self.pairs = [f"t{t}_u{i}" for t in range(n_targets) for i in range(per_target)]
        n = len(self.pairs)
        self._primer_pair_ix = {p: i for i, p in enumerate(self.pairs)}
        self.indv_combined_arr = rng.normal(size=n)
        m = rng.normal(size=(n, n))
        self.pairwise_combined_arr = (m + m.T) / 2


def _search(seed):
    cf = _StubCostFunction(n_targets=12, per_target=5)
    primer_df = pd.DataFrame(
        {"target_id": [p.split("_")[0] for p in cf.pairs], "pair_name": cf.pairs}
    )
    return GreedySearch(primer_df, cf).run(N=300, seed=seed)


def test_a_seeded_search_returns_the_same_multiplexes_every_time():
    a, b = _search(5), _search(5)
    assert (a.all_indices == b.all_indices).all()
    assert (a.all_costs == b.all_costs).all()


def test_different_seeds_explore_differently():
    assert not (_search(1).all_indices == _search(2).all_indices).all()


def test_seed_is_optional_and_a_negative_kernel_seed_means_unseeded():
    cf = _StubCostFunction(6, 3)
    n = len(cf.pairs)
    args = (30, np.arange(n, dtype=np.int64), np.arange(0, n, 3, dtype=np.int64),
            np.full(6, 3, dtype=np.int64), cf.indv_combined_arr, cf.pairwise_combined_arr)
    ix, cost = _greedy_search_njit(args[0], -1, *args[1:])
    assert ix.shape == (30, 6) and cost.shape == (30,)


def test_candidate_pairs_are_visited_in_sorted_order():
    """A bare set made the visit order, and so tie-breaks, depend on the hash seed."""
    cf = _StubCostFunction(3, 4)
    names = list(reversed(cf.pairs))            # deliberately scrambled input order
    primer_df = pd.DataFrame({"target_id": [p.split("_")[0] for p in names], "pair_name": names})
    a = GreedySearch(primer_df, cf).run(N=50, seed=3)
    b = GreedySearch(primer_df.sample(frac=1, random_state=0), cf).run(N=50, seed=3)
    assert (a.all_indices == b.all_indices).all()
