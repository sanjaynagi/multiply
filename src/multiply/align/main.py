import heapq
import os
import sys
import pandas as pd
import numpy as np

from multiply.util.printing import print_header, print_footer
from multiply.util.dirs import produce_dir
from .algorithms import (
    PrimerDimerLike,
    _pairwise_scores_njit,
    encode_primers_for_pairwise,
)


def align(primer_csv):
    """
    Run a pairwise alignment between all primers in `primer_csv` to identify
    potential primer dimers

    """

    # Cap the number of full alignment objects retained for the
    # `alignment_diagrams.txt` / `table.alignment_scores.csv` reports.
    # The full ASCII string of every (i, j) pair dominates RSS at panel scale;
    # the lowest-(i.e. worst-) scoring K is all that's actually consulted later.
    SAVE_TOP = 1000

    # PARSE CLI
    t0 = print_header("MULTIPLY: Perform pairwise alignment to identify possible primer dimers")
    input_dir = os.path.dirname(primer_csv)
    output_dir = produce_dir(input_dir, "align")
    print("Parsing inputs...")
    print(f"  Primer CSV: {primer_csv}")
    print(f"  Output directory: {output_dir}")
    print("Done.\n")

    # LOAD DATA
    print("Loading primers...")
    primer_df = pd.read_csv(primer_csv)
    primer_df.reset_index(inplace=True, drop=True)  # precaution, to ensure ordering
    n_primers = primer_df.shape[0]
    print(f"  Found {n_primers} primers.")
    print("Done.\n")

    # SET MODEL
    model = PrimerDimerLike()
    model.load_parameters()

    # COMPUTE PAIRWISE — fully-jitted, prange across i.
    # Two passes:
    #   (1) Parallel score-only kernel: fills the symmetric (N, N) score
    #       matrix in one shot. ~N²/2 alignments scored, but threads share
    #       no rows so writes don't contend.
    #   (2) Single-threaded materialisation of full PrimerAlignment objects
    #       only for the top-K (lowest-score, most dimer-like) pairs. K is
    #       small (SAVE_TOP=1000) so the slow Python path stays cheap.
    seqs = primer_df["seq"].to_numpy()
    names = primer_df["primer_name"].to_numpy()

    print("Computing pairwise alignments (parallel score kernel)...")
    fwd, rev, lengths = encode_primers_for_pairwise(seqs)
    pairwise_scores = _pairwise_scores_njit(
        fwd, rev, lengths, model._nn_lut, model.end_length, model.end_bonus,
    )
    print("  Score matrix complete.")

    print(f"Materialising top-{SAVE_TOP} worst-scoring alignments...")
    # Pull the lower triangle off (i <= j) to deduplicate the symmetric pairs.
    iu, ju = np.triu_indices(n_primers)
    flat_scores = pairwise_scores[iu, ju]
    # argpartition picks the SAVE_TOP smallest (lowest-score, most dimer-like)
    # in O(N²) without sorting all pairs.
    k = min(SAVE_TOP, flat_scores.shape[0])
    worst_flat_ix = np.argpartition(flat_scores, k - 1)[:k]
    worst_i = iu[worst_flat_ix]
    worst_j = ju[worst_flat_ix]

    alignments = []
    for i, j in zip(worst_i.tolist(), worst_j.tolist()):
        model.set_primers(seqs[i], seqs[j], names[i], names[j])
        model.align()
        alignments.append(model.get_primer_alignment())
    alignments.sort(key=lambda a: a.score)
    print("  Done.\n")

    # SAVE AS CSV
    print("Saving outputs...")
    matrix_output_csv = f"{output_dir}/matrix.pairwise_scores.csv"
    pairwise_df = pd.DataFrame(
        pairwise_scores,
        index=primer_df["primer_name"],
        columns=primer_df["primer_name"],
    )
    pairwise_df.to_csv(matrix_output_csv)
    print(f"  Pairwise interaction matrix: {matrix_output_csv}")

    # ADDITIONAL OUTPUTS FOR HIGH-SCORING ALIGNMENTS
    n_kept = len(alignments)
    alignment_df = pd.DataFrame(alignments)
    alignment_df.insert(0, "rank", range(n_kept))
    alignment_df.to_csv(f"{output_dir}/table.alignment_scores.csv", index=False)
    with open(f"{output_dir}/alignment_diagrams.txt", "w") as fn:
        for ix, a in enumerate(alignments):
            fn.write(f"Alignment Rank: {ix:05d}\n")
            fn.write(f"{a.alignment}\n\n")
    print("Done.\n")

    print_footer(t0)
