from dataclasses import dataclass, field
from itertools import product
from typing import Tuple, Sequence

import numpy as np


@dataclass(unsafe_hash=True, order=True)
class Multiplex:
    """
    Encapsulate information about a multiplex in a manner
    that allows for hashing, such that a unique set of 
    multiplexes can be found.

    To make hashable, the `.primer_pairs` attribute is
    co-erced into a tuple during `__post_init__`.

    NB:
    - Call to sorted() must use `key` to sort by cost.


    """

    cost: float = field(compare=False)
    primer_pairs: Tuple[str]
    method: str = ""

    def __post_init__(self):
        if not isinstance(self.primer_pairs, Tuple):
            self.primer_pairs.sort()
            self.primer_pairs = tuple(self.primer_pairs)

    def get_primer_names(self):
        directions = ["F", "R"]
        return [f"{p}_{d}" for p, d in product(self.primer_pairs, directions)]


@dataclass
class MultiplexResults:
    """Lazy container for the output of a jitted selector kernel.

    Holds the raw ``(N, n_targets)`` integer-indexed multiplex array and the
    ``(N,)`` cost array directly, plus the ``int → pair_name`` reverse map.
    Conversion to :class:`Multiplex` instances is deferred and done per-row
    via :py:meth:`materialise` so that callers needing only costs (e.g. the
    Greedy-vs-Random diagnostics plot) or only the top-N picks never pay
    the cost of constructing ~10K Python objects.

    Parameters
    ----------
    all_indices : np.ndarray, shape (N, n_targets), dtype int64
        Per-iteration chosen primer-pair indices into the cost arrays.
    all_costs : np.ndarray, shape (N,), dtype float64
        Per-iteration LinearCost evaluation.
    idx_to_pair : dict[int, str]
        Reverse map from primer-pair index back to pair_name.
    n_targets : int
        Number of targets (columns in ``all_indices``).
    """

    all_indices: np.ndarray
    all_costs: np.ndarray
    idx_to_pair: dict
    n_targets: int

    def __len__(self) -> int:
        return self.all_indices.shape[0]

    def materialise(self, iter_indices: Sequence[int]) -> list:
        """Build :class:`Multiplex` objects for the given iteration indices."""
        out = []
        for it in iter_indices:
            it = int(it)
            pairs = [
                self.idx_to_pair[int(self.all_indices[it, t])]
                for t in range(self.n_targets)
            ]
            out.append(
                Multiplex(cost=float(self.all_costs[it]), primer_pairs=pairs)
            )
        return out
