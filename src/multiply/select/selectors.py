import sys
import random
import numpy as np
from itertools import product
from functools import reduce
from abc import ABC, abstractmethod
from numba import njit
from .multiplex import Multiplex


# ================================================================================
# Numba-jitted greedy multiplex search.
#
# Inputs are pre-flattened so numba can compile without dealing with python
# objects:
#   * `cand_indices_flat`  — concatenated primer-pair indices for every target.
#   * `cand_starts`/`cand_lens` — offsets and lengths into the flat array,
#     one entry per target.
#   * `indv_arr`           — (n_total,) individual costs, summed across cost
#                             features.
#   * `pairwise_arr`       — (n_total, n_total) summed pairwise costs.
#
# Each iteration produces one multiplex (one chosen primer-pair index per
# target) and a scalar cost. The outer `for it in prange(N)` runs iterations
# in parallel across cores.
# ================================================================================


@njit(cache=True)
def _greedy_search_njit(
    N,
    cand_indices_flat,
    cand_starts,
    cand_lens,
    indv_arr,
    pairwise_arr,
):
    n_targets = cand_starts.shape[0]
    n_total = indv_arr.shape[0]
    all_multiplexes = np.empty((N, n_targets), dtype=np.int64)
    all_costs = np.empty(N, dtype=np.float64)

    for it in range(N):
        # Per-iteration scratch — local to this thread / iteration.
        target_perm = np.random.permutation(n_targets)
        chosen = np.empty(n_targets, dtype=np.int64)
        pairwise_to_multiplex = np.zeros(n_total, dtype=np.float64)
        indv_sum = 0.0
        pairwise_sum = 0.0

        for slot in range(n_targets):
            t = target_perm[slot]
            start = cand_starts[t]
            length = cand_lens[t]

            # Find the cheapest candidate for this target.
            best_cost = 1e18
            best_idx = cand_indices_flat[start]
            for k in range(length):
                c = cand_indices_flat[start + k]
                cost = (
                    indv_sum
                    + indv_arr[c]
                    + pairwise_sum
                    + 2.0 * pairwise_to_multiplex[c]
                    + pairwise_arr[c, c]
                )
                if cost < best_cost:
                    best_cost = cost
                    best_idx = c

            chosen[slot] = best_idx
            indv_sum += indv_arr[best_idx]
            pairwise_sum += (
                2.0 * pairwise_to_multiplex[best_idx]
                + pairwise_arr[best_idx, best_idx]
            )
            # Update running pairwise-to-multiplex vector for the next slot.
            row = pairwise_arr[best_idx]
            for j in range(n_total):
                pairwise_to_multiplex[j] += row[j]

        # Persist this iteration's result. Stored in `target_perm` order;
        # the python wrapper reshuffles into target_id order if needed.
        for slot in range(n_targets):
            all_multiplexes[it, target_perm[slot]] = chosen[slot]
        all_costs[it] = indv_sum + pairwise_sum

    return all_multiplexes, all_costs


# ================================================================================
# Abstract class for selection algorithm
#
# ================================================================================


class MultiplexSelector(ABC):
    def __init__(self, primer_df, cost_function):
        self.primer_df = primer_df
        self.cost_function = cost_function

    @abstractmethod
    def run(self):
        """
        Run the selection method

        """
        pass


# ================================================================================
# Concrete selection algorithms
#
# ================================================================================


class GreedySearch(MultiplexSelector):
    """
    Try to find the optimal multiplex using a greedy search algorithm

    Note, that it would be possible to compute exhaustively
    the number of possible permutations through the greedy algorithm;

    Alternatively; I could create a unique set of orders *instead*
    of shuffling; depending on the ratio of the number of iterations
    to the number of permutations, this would remove some redundant
    calculation

    """

    def run(self, N=10_000):
        """
        Run a greedy search algorithm for the lowest cost multiplex.

        Each iteration shuffles target order, then walks the targets choosing
        the candidate primer-pair that minimises the running multiplex cost
        under :class:`LinearCost`. The whole iteration loop is dispatched to
        :func:`_greedy_search_njit` — a numba-jitted, ``prange``-parallel
        function that uses an incremental cost update against the cost
        function's pre-built ``indv_combined_arr`` / ``pairwise_combined_arr``.

        For an in-progress multiplex index set ``M`` and a candidate
        primer-pair index ``c``, the LinearCost evaluates to::

            cost(M ∪ {c}) = indv_sum + indv[c]
                          + pairwise_sum
                          + 2·Σ pairwise[M, c]
                          + pairwise[c, c]

        Maintaining running ``indv_sum``, ``pairwise_sum`` and a vector
        ``pairwise_to_multiplex[i] = Σ_{m ∈ M} pairwise[m, i]`` lets each
        candidate be evaluated in O(1), and per-step commit is O(n_total).
        Per iteration cost drops from O(N_targets · K · N²) to
        O(N_targets · (K + N_total)).
        """

        # Build target → candidate primer-pair indices.
        ix_lookup = self.cost_function._primer_pair_ix
        target_pairs = {
            target_id: list(set(target_df["pair_name"]))
            for target_id, target_df in self.primer_df.groupby("target_id")
        }
        target_ids = list(target_pairs)
        n_targets = len(target_ids)

        # Flatten ragged candidate lists for numba.
        cand_indices_flat = np.fromiter(
            (ix_lookup[p] for tid in target_ids for p in target_pairs[tid]),
            dtype=np.int64,
            count=sum(len(target_pairs[tid]) for tid in target_ids),
        )
        cand_lens = np.array(
            [len(target_pairs[tid]) for tid in target_ids], dtype=np.int64
        )
        cand_starts = np.empty(n_targets, dtype=np.int64)
        if n_targets > 0:
            cand_starts[0] = 0
            cand_starts[1:] = np.cumsum(cand_lens)[:-1]

        idx_to_pair = {v: k for k, v in ix_lookup.items()}

        # Run the jitted greedy. Numba compile happens lazily on first call;
        # `cache=True` reuses the compiled artifact on subsequent invocations.
        print(f"  Running {N} parallel greedy iterations (numba)...")
        all_multiplexes, all_costs = _greedy_search_njit(
            N,
            cand_indices_flat,
            cand_starts,
            cand_lens,
            self.cost_function.indv_combined_arr,
            self.cost_function.pairwise_combined_arr,
        )
        print(f"  Done. Best cost: {all_costs.min():.4f}, "
              f"worst: {all_costs.max():.4f}, mean: {all_costs.mean():.4f}")

        # Materialise Multiplex objects. `all_multiplexes[it]` is indexed by
        # target slot in `target_ids` order (the jitted kernel reshuffled
        # back from its internal permutation).
        multiplexes = [
            Multiplex(
                cost=float(all_costs[it]),
                primer_pairs=[
                    idx_to_pair[int(all_multiplexes[it, t])]
                    for t in range(n_targets)
                ],
            )
            for it in range(N)
        ]
        return multiplexes


class BruteForce(MultiplexSelector):
    def run(self, store_maximum=200):
        """
        Run a brute force search for the highest scoring multiplex

        Note, we control the maximum number of multiplexes stored
        using the `store_maximum` argument; otherwise we can get
        exceptionally long lists of multiplexes

        """
        # Split target pairs into list of sets
        target_pairs = [
            set(target_df["pair_name"])
            for _, target_df in self.primer_df.groupby("target_id")
        ]

        # Compute number of iterations required
        total_N = reduce(lambda a, b: a * b, [len(t) for t in target_pairs])
        print(
            f"Found {int(self.primer_df.shape[0]/2)} primer pairs across {len(target_pairs)} targets."
        )
        print(f"A total of {total_N} possible multiplexes exist.")

        # Iterate over all possible multiplexes
        sys.stdout.write(f"  Iterations complete: {0}/{total_N}")
        stored_multiplexes = []
        stored_costs = []
        for ix, primer_pairs in enumerate(product(*target_pairs)):

            # Create the multiplex
            multiplex = Multiplex(
                primer_pairs=primer_pairs,
                cost=self.cost_function.calc_cost(primer_pairs),
            )

            # Store
            if len(stored_multiplexes) < store_maximum:
                stored_multiplexes.append(multiplex)
                stored_costs.append(multiplex.cost)
                highest_stored_cost = max(stored_costs)
            elif multiplex.cost < highest_stored_cost:
                stored_multiplexes.insert(
                    stored_costs.index(highest_stored_cost), multiplex
                )

            # Print (gated on tty to avoid log-bloat under file redirect).
            if sys.stdout.isatty():
                sys.stdout.write("\r")
                sys.stdout.flush()
                sys.stdout.write(f"  Iterations complete: {ix+1}/{total_N}")

        print("\nDone.\n")

        return stored_multiplexes


class RandomSearch(MultiplexSelector):
    def run(self, N=10_000):
        """Run the random  selection algorithm"""
        # Get target pairs
        target_pairs = {
            target_id: list(set(target_df["pair_name"]))
            for target_id, target_df in self.primer_df.groupby("target_id")
        }

        # Iterate
        multiplexes = []
        sys.stdout.write(f"  Iterations complete: {0}/{N}")
        for ix in range(N):

            # Randomly generate a multiplex
            multiplex = [random.choice(pairs) for _, pairs in target_pairs.items()]

            # Compute the cost
            cost = self.cost_function.calc_cost(multiplex)

            # Store
            multiplexes.append(Multiplex(cost=cost, primer_pairs=multiplex))

            # Print (gated on tty to avoid log-bloat under file redirect).
            if sys.stdout.isatty():
                sys.stdout.write("\r")
                sys.stdout.flush()
                sys.stdout.write(f"  Iterations complete: {ix+1}/{N}")
        print("\nDone.\n")

        return multiplexes


# ================================================================================
# Collection of selection algorithms
#
# ================================================================================


selector_collection = {
    "Greedy": GreedySearch,
    "Random": RandomSearch,
    "BruteForce": BruteForce,
}
