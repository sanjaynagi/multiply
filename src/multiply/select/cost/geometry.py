"""Geometry-based pairwise cost: amplicons in one pool must not overlap.

Two amplicons that overlap in the same pool put a forward primer of one on the
same bases as a reverse primer of the other, so those primers are
complementary over the overlap and can dimer or form short cross-products.
Neighbouring amplicons are normally split between two pools for exactly this
reason; this cost keeps the *chosen* primer pairs from overlapping when the
pool assignment alone cannot guarantee it.
"""

import numpy as np
import pandas as pd

from .features import PairwiseCosts


def amplicon_overlap_matrix(candidates):
    """
    Build a square matrix over candidate primer pairs: 1 where the two pairs'
    amplicons overlap on the same chromosome, otherwise 0

    Pairs of the same target are never both chosen, so they are left at 0.

    params
        candidates: pandas DataFrame
            `table.candidate_primers.csv`: `pair_name`, `target_id`,
            `direction`, `chrom` and `start` (for a reverse primer, its
            inclusive 5' end).

    returns
        pandas DataFrame
            Indexed and columned by pair name, sorted.

    """
    fwd = candidates[candidates["direction"] == "F"].drop_duplicates("pair_name").set_index("pair_name")
    rev = candidates[candidates["direction"] == "R"].drop_duplicates("pair_name").set_index("pair_name")
    pairs = pd.DataFrame(
        {
            "chrom": fwd["chrom"],
            "target": fwd["target_id"],
            "lo": fwd["start"],
            "hi": rev["start"].reindex(fwd.index) + 1,
        }
    ).sort_index()
    names = list(pairs.index)
    matrix = np.zeros((len(names), len(names)), dtype=np.float64)
    position = {name: i for i, name in enumerate(names)}
    for _, block in pairs.groupby("chrom"):
        idx = np.array([position[n] for n in block.index])
        lo, hi = block["lo"].to_numpy(), block["hi"].to_numpy()
        target = block["target"].to_numpy()
        overlap = (lo[:, None] < hi[None, :]) & (lo[None, :] < hi[:, None])
        overlap &= target[:, None] != target[None, :]
        matrix[np.ix_(idx, idx)] = overlap
    return pd.DataFrame(matrix, index=names, columns=names)


class AmpliconOverlapCost(PairwiseCosts):
    """An overlap penalty applied as given: not collapsed, not z-scored."""

    def collapse_to_per_pair(self, collapse_func=sum):
        # Already per pair.
        self.primer_pair_values = self.primer_values
        return self

    def normalise_costs(self):
        # A z-score of a rare 0/1 indicator would depend on how many candidate
        # combinations happen to overlap; the weight is the cost of one overlap.
        self.primer_pair_costs = self.weight * self.primer_pair_values
        return self

    @staticmethod
    def _check_primer_values(primer_values):
        assert primer_values.shape[0] == primer_values.shape[1], "overlap matrix must be square"
