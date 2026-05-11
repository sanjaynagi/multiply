import json
import numpy as np
from numba import njit, prange
from numba.typed import Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from .nn_model import create_nn_score_dt
from multiply.util.definitions import ROOT_DIR


# ================================================================================
# Numba-jitted primer-dimer alignment hot loop.
#
# The original implementation iterated alignment positions in pure Python and
# looked up nearest-neighbour Gibbs scores via a `dict[str, float]` keyed on
# strings like `"AT/CG"`. At ~720k primer pairs × ~400 inner iterations per
# pair, the per-iteration string formatting + dict lookup dominates wall time.
#
# The optimised path encodes each primer as an `int8` array (A=0, T=1, C=2,
# G=3, ambiguous→0) and uses a flat 256-element float lookup table indexed by
# `(l_dinuc_id, s_dinuc_id) = (a*4+b)*16 + (c*4+d)`. Dinucleotide computation
# and the end-extension-bonus calculation are wrapped in a single `@njit`
# function returning just `(best_score, best_start)`. The matching-state
# vector required for the ASCII alignment diagram is recomputed lazily and
# only for the (small) set of kept alignments.
# ================================================================================


_BASE_TO_INT = {"A": 0, "T": 1, "C": 2, "G": 3}
# Reverse-complement table indexed by encoded base int.
_RC_INT = np.array([1, 0, 3, 2], dtype=np.int8)


def _encode_seq(seq: str) -> np.ndarray:
    """Encode a primer sequence as an int8 array; ambiguous bases map to 0."""
    return np.array(
        [_BASE_TO_INT.get(c, 0) for c in seq.upper()], dtype=np.int8
    )


def _build_nn_lut(nn_score_dt: dict) -> np.ndarray:
    """Convert the string-keyed nn-score dict into a 256-entry float64 lookup.

    Layout: ``lut[(l1*4+l2)*16 + (s1*4+s2)]``.
    """
    int_to_base = "ATCG"
    lut = np.zeros(256, dtype=np.float64)
    for l1, b1 in enumerate(int_to_base):
        for l2, b2 in enumerate(int_to_base):
            for s1, c1 in enumerate(int_to_base):
                for s2, c2 in enumerate(int_to_base):
                    key = f"{b1}{b2}/{c1}{c2}"
                    lut[(l1 * 4 + l2) * 16 + (s1 * 4 + s2)] = nn_score_dt[key]
    return lut


@njit(cache=True)
def _align_njit(l_arr, s_arr, nn_lut, end_length, end_bonus):
    """Find the best (lowest-Gibbs) ungapped 5'-overhang alignment.

    Mirrors :py:meth:`PrimerDimerLike.align` exactly, returning
    ``(best_score, best_start)``.
    """
    nL = l_arr.shape[0]
    nS = s_arr.shape[0]
    best_start = 0
    best_score = 10.0

    for i in range(nL - 1):
        current_score = 0.0
        # match_len tracks how many positions of `s` we actually iterated.
        match_len = 0
        for j in range(nS - 1):
            l1 = l_arr[i + j]
            l2 = l_arr[i + j + 1]
            s1 = s_arr[j]
            s2 = s_arr[j + 1]
            current_score += nn_lut[(l1 * 4 + l2) * 16 + (s1 * 4 + s2)]
            match_len = j + 1
            if i + j == nL - 2:
                break

        # Add the trailing matching state for the last nucleotide pair.
        match_len += 1

        # End-extension bonus: count contiguous matches at each end (only when
        # an overhang exists in that direction so extension would be possible).
        overhang_left = i > 0
        overhang_right = match_len < nS

        left_end = 0
        if overhang_left:
            for k in range(end_length):
                if k >= match_len:
                    break
                idx = k
                pos_l = i + idx
                pos_s = idx
                if pos_s < nS and pos_l < nL:
                    if _RC_INT[s_arr[pos_s]] != l_arr[pos_l]:
                        break
                    left_end += 1
                else:
                    break

        right_end = 0
        if overhang_right:
            for k in range(end_length):
                if k >= match_len:
                    break
                idx = match_len - 1 - k
                pos_l = i + idx
                pos_s = idx
                if pos_s >= 0 and pos_l >= 0 and pos_s < nS and pos_l < nL:
                    if _RC_INT[s_arr[pos_s]] != l_arr[pos_l]:
                        break
                    right_end += 1
                else:
                    break

        current_score += (left_end + right_end) * end_bonus

        if current_score <= best_score:
            best_score = current_score
            best_start = i

    return best_score, best_start


def encode_primers_for_pairwise(seqs):
    """Pack a list of primer sequences for the parallel pairwise kernel.

    Returns three arrays: forward-encoded (``N, max_len``), reverse-encoded
    (same shape, zero-padded right), and integer lengths. The reverse array
    mirrors the ``s = s[::-1]`` step in :py:meth:`PrimerDimerLike.align`:
    the shorter primer is reversed before pairwise alignment so 3'-end
    overhangs are evaluated correctly.

    Parameters
    ----------
    seqs : Sequence[str]
        Primer sequences in original order.

    Returns
    -------
    fwd : np.ndarray, shape (N, max_len), dtype int8
    rev : np.ndarray, shape (N, max_len), dtype int8
        Reverse of ``fwd`` per row, left-aligned within the row's ``length``.
    lengths : np.ndarray, shape (N,), dtype int32
    """
    n = len(seqs)
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    max_len = int(lengths.max()) if n else 0
    fwd = np.zeros((n, max_len), dtype=np.int8)
    rev = np.zeros((n, max_len), dtype=np.int8)
    for i, seq in enumerate(seqs):
        enc = _encode_seq(seq)
        L = enc.shape[0]
        fwd[i, :L] = enc
        rev[i, :L] = enc[::-1]
    return fwd, rev, lengths


@njit(cache=True)
def _score_pair_njit(l_arr, s_arr, nL, nS, nn_lut, end_length, end_bonus):
    """Score one primer pair from pre-encoded slices. Helper for the prange
    kernel; mirrors :func:`_align_njit` but reads the longer/shorter
    decision through int64 indices rather than (l_idx, s_idx) tuples that
    confuse numba's parfor type inference.
    """
    best_score = 10.0

    for k_off in range(nL - 1):
        current_score = 0.0
        match_len = 0
        for jj in range(nS - 1):
            l1 = l_arr[k_off + jj]
            l2 = l_arr[k_off + jj + 1]
            s1 = s_arr[jj]
            s2 = s_arr[jj + 1]
            current_score += nn_lut[(l1 * 4 + l2) * 16 + (s1 * 4 + s2)]
            match_len = jj + 1
            if k_off + jj == nL - 2:
                break
        match_len += 1

        overhang_left = k_off > 0
        overhang_right = match_len < nS

        left_end = 0
        if overhang_left:
            for kk in range(end_length):
                if kk >= match_len:
                    break
                pos_l = k_off + kk
                pos_s = kk
                if pos_s < nS and pos_l < nL:
                    if _RC_INT[s_arr[pos_s]] != l_arr[pos_l]:
                        break
                    left_end += 1
                else:
                    break

        right_end = 0
        if overhang_right:
            for kk in range(end_length):
                if kk >= match_len:
                    break
                idx_ = match_len - 1 - kk
                pos_l = k_off + idx_
                pos_s = idx_
                if pos_s >= 0 and pos_l >= 0 and pos_s < nS and pos_l < nL:
                    if _RC_INT[s_arr[pos_s]] != l_arr[pos_l]:
                        break
                    right_end += 1
                else:
                    break

        current_score += (left_end + right_end) * end_bonus

        if current_score <= best_score:
            best_score = current_score

    return best_score


@njit(parallel=True, cache=True)
def _pairwise_scores_njit(fwd, rev, lengths, nn_lut, end_length, end_bonus):
    """Fill the (N, N) primer-dimer score matrix in parallel via ``prange``.

    Each thread iterates over disjoint rows of ``i`` (the outer ``prange``)
    and writes the upper-triangle cells of row ``i`` plus the mirrored
    ``scores[j, i]`` cell. No two threads write the same (i, j), so the
    parfor is contention-free without atomics.

    Mirrors :py:meth:`PrimerDimerLike.align` for every (i, j) with ``j >= i``,
    deciding longer-vs-shorter the same way ``set_primers`` does (sort by
    length descending; on ties, primer1 wins — here that maps to taking the
    row with the lower index).

    Parameters
    ----------
    fwd, rev : np.ndarray, shape (N, max_len), dtype int8
        Output of :func:`encode_primers_for_pairwise`.
    lengths : np.ndarray, shape (N,), dtype int64
    nn_lut : np.ndarray, shape (256,), dtype float32
        Flat dinucleotide nearest-neighbour table.
    end_length : int
    end_bonus : float

    Returns
    -------
    scores : np.ndarray, shape (N, N), dtype float64
        Symmetric primer-dimer score matrix.
    """
    n = fwd.shape[0]
    scores = np.zeros((n, n), dtype=np.float64)

    for i in prange(n):
        Li = lengths[i]
        for j in range(i, n):
            Lj = lengths[j]
            # Pass the longer primer's row as l_arr, shorter as s_arr.
            # On ties, l = i (primer1 by index — matches set_primers' stable
            # sort).
            if Li >= Lj:
                score = _score_pair_njit(
                    fwd[i], rev[j], Li, Lj, nn_lut, end_length, end_bonus
                )
            else:
                score = _score_pair_njit(
                    fwd[j], rev[i], Lj, Li, nn_lut, end_length, end_bonus
                )
            scores[i, j] = score
            scores[j, i] = score

    return scores


# ================================================================================
# Define an alignment between two primers
#
# ================================================================================


@dataclass(order=True)
class PrimerAlignment:
    """
    Represent the alignment of two primers
    """

    primer1_name: str = field(compare=False)
    primer2_name: str = field(compare=False)
    primer1: str = field(compare=False)
    primer2: str = field(compare=False)
    score: float = field(compare=True)
    alignment: str = field(compare=False, repr=False)
    # index: int=field(compare=False, repr=False)


# ================================================================================
# Abstract base class for various primer alignment algorithms
#
# ================================================================================


class AlignmentAlgorithm(ABC):
    """
    Alignment algorithm for a pair of primers

    """

    rc_map = {"A": "T", "T": "A", "C": "G", "G": "C"}

    def __init__(self):
        pass

    def set_primers(self, primer1, primer2, primer1_name, primer2_name):
        """
        Set a pair for primers to align

        """
        self.primer1 = primer1
        self.primer2 = primer2
        self.primer1_name = primer1_name
        self.primer2_name = primer2_name

        self.score = None  # reset

    @abstractmethod
    def load_parameters():
        pass

    @abstractmethod
    def align():
        """
        Align the primers

        """
        pass

    @abstractmethod
    def get_alignment_string():
        """
        Create an ASCII string representing the aligned primers

        """
        pass

    def print_alignment(self):
        """
        Print an ASCII view of the aligned primers

        """
        print(self.get_alignment_string())

    def get_primer_alignment(self):
        """
        Return an alignment object

        """
        return PrimerAlignment(
            primer1=self.primer1,
            primer2=self.primer2,
            primer1_name=self.primer1_name,
            primer2_name=self.primer2_name,
            score=self.score,
            alignment=self.get_alignment_string(),
        )


# ================================================================================
# Concrete primer alignment algorithms
#
# ================================================================================


class PrimerDimerLike(AlignmentAlgorithm):
    """
    Align two primers using an algorithm like the one described
    by Johnston et al. (2019) Sci Reports
    
    Primary idea is to only allow:
    - Ungapped alignments
    - With 5' overhangs (i.e. extensible)
    
    And then to add a bonus if either 3' end is complementary in
    the highest scoring alignment
    
    
    """
    
    param_path = f"{ROOT_DIR}/settings/alignment/primer_dimer/parameters.json"
    
    def load_parameters(self):
        """
        Load parameters necessary for Primer Dimer algorithm,
        and set as attributes

        """
        # Load parameter JSON
        params = json.load(open(self.param_path, "r"))

        # Load nearest neighbour model (kept for reference / debugging).
        self.nn_scores = create_nn_score_dt(
            match_json=f"{ROOT_DIR}/{params['match_scores']}",  # this is a path
            single_mismatch_json=f"{ROOT_DIR}/{params['single_mismatch_scores']}",  # this is a path
            double_mismatch_score=params['double_mismatch_score']  # this is a float
        )
        # Flat numpy lookup used by the @njit alignment kernel.
        self._nn_lut = _build_nn_lut(self.nn_scores)

        # Load penalties
        self.end_length = params["end_length"]
        #self.end_penalty = params["end_penalty"]
        self.end_bonus = params["end_bonus"]
        
    @staticmethod
    def _calc_linear_extension_bonus(matching, 
                                     overhang_left, 
                                     overhang_right, 
                                     end_length, 
                                     end_bonus):
        """
        Calculate a bonus score for primer-dimer alignments that would allow for *extension*

        params
            matching : list[bool]
                List of booleans indicating whether or not bases matched
                for this alignment position
            overhang_left : bool
                Is there an overhang on the left side of the matches;
                i.e. would extension be possible?
            overhang_right : bool
                As above, but right side.
            end_length : int
                Number of bases to consider for end bonus.
            end_bonus : float
                Bonus to add per aligned, extendible base.

        returns
            _ : float
                Bonus score in [-2*end_length*end_bonus, 0].

        """

        # Left end
        left_end = 0
        for match in matching[:end_length]:
            if not overhang_left:
                break
            if not match:
                break
            left_end += 1

        # Right end
        right_end = 0
        for match in matching[::-1][:end_length]:
            if not overhang_right:
                break
            if not match:
                break
            right_end += 1

        return (left_end + right_end) * end_bonus
        
    def align(self):
        """
        Align primers; finding highest score and its associated start position.

        Delegates the inner hot loop to a numba-jitted helper. The matching-
        state vector required for ``get_alignment_string`` is recomputed lazily
        in :py:meth:`_recompute_matching` since only kept alignments need it.
        """

        # Identify longer and shorter primer
        primers = [self.primer1, self.primer2]
        primers.sort(key=len, reverse=True)
        l, s = primers
        s = s[::-1]

        l_arr = _encode_seq(l)
        s_arr = _encode_seq(s)

        best_score, best_start = _align_njit(
            l_arr, s_arr, self._nn_lut, self.end_length, self.end_bonus
        )

        self.score = float(best_score)
        self.best_start = int(best_start)

        # Stash inputs needed for lazy matching/diagram materialisation.
        self.s = s
        self.l = l
        self._l_arr = l_arr
        self._s_arr = s_arr
        self.best_matching = None  # computed on demand

    def _recompute_matching(self) -> list:
        """Re-derive the per-position match boolean list at ``self.best_start``."""
        l_arr = self._l_arr
        s_arr = self._s_arr
        nL, nS = l_arr.shape[0], s_arr.shape[0]
        i = self.best_start
        matching = []
        for j in range(nS - 1):
            matching.append(bool(_RC_INT[s_arr[j]] == l_arr[i + j]))
            if i + j == nL - 2:
                break
        # Last position
        last_j = len(matching)
        if last_j < nS and (i + last_j) < nL:
            matching.append(bool(_RC_INT[s_arr[last_j]] == l_arr[i + last_j]))
        return matching
        
    def get_alignment_string(self):
        """
        Return a string representing the best alignment
        between the two primers

        """

        # Recover names, kinda ugly
        if self.l == self.primer1:
            lname = self.primer1_name
            sname = self.primer2_name
        else:
            lname = self.primer2_name
            sname = self.primer1_name

        # Create space, regardless of length of primer names
        name_max = max([len(self.primer1_name), len(self.primer2_name)])
        str_template = "{:>%d}    {}\n" % name_max

        # Lazily materialise the matching state for diagram rendering.
        if self.best_matching is None:
            self.best_matching = self._recompute_matching()

        # Create individual strings
        gap = " "*self.best_start
        lstr = f"5-{self.l}-3"
        mstr = f"{gap}  {''.join(['|' if m else ' ' for m in self.best_matching])}"
        sstr = f"{gap}3-{self.s}-5"

        align_str = f"Dimer Score: {self.score:.03f}\n"
        align_str += str_template.format(lname, lstr)
        align_str += str_template.format("", mstr)
        align_str += str_template.format(sname, sstr)
        align_str += "\n"

        return align_str