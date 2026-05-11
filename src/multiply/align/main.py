import heapq
import os
import sys
import pandas as pd
import numpy as np

from multiply.util.printing import print_header, print_footer
from multiply.util.dirs import produce_dir
from .algorithms import PrimerDimerLike


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

    # COMPUTE PAIRWISE
    # NB: Essential to keep track of ordering here.
    print("Computing pairwise alignments...")
    # Hoist columns out of the inner loop — pandas iloc is ~50 us per call.
    seqs = primer_df["seq"].to_numpy()
    names = primer_df["primer_name"].to_numpy()
    # Bounded min-heap of the worst (lowest-score) alignments. We push
    # `(-score, sequence_no, alignment)` so heapq's min-heap semantics give
    # us the highest score on top, allowing efficient eviction when full.
    # `sequence_no` is a tie-breaker so PrimerAlignment objects don't need
    # ordering comparison once two have the same score.
    alignments_heap: list[tuple] = []
    seq_no = 0
    pairwise_scores = np.zeros((n_primers, n_primers))
    fmt_str = "  {:<4} {:<14} {:4>}/{:<4}"
    print("  {:<4} {:<14} {:<}".format("#", "Primer", "Completed"))
    for i in range(n_primers):

        primer1_seq, primer1_name = seqs[i], names[i]

        for j in range(i, n_primers):

            primer2_seq, primer2_name = seqs[j], names[j]

            # Align
            model.set_primers(primer1_seq, primer2_seq, primer1_name, primer2_name)
            model.align()

            score = model.score
            pairwise_scores[i, j] = score
            pairwise_scores[j, i] = score

            # Push into bounded heap, materialise alignment object only when retained.
            if len(alignments_heap) < SAVE_TOP:
                heapq.heappush(
                    alignments_heap, (-score, seq_no, model.get_primer_alignment())
                )
                seq_no += 1
            elif -score > alignments_heap[0][0]:
                # Current alignment is *worse* (lower score, more dimer-like) than the
                # best of the kept-set — evict the kept-best (least-bad) and add this.
                heapq.heapreplace(
                    alignments_heap, (-score, seq_no, model.get_primer_alignment())
                )
                seq_no += 1

            # Inner-progress prints suppressed when stdout isn't a tty.
            # At ~10K primers the per-step CR-overwrite balloons file logs to
            # GBs even though only a handful of frames are user-visible.
            if sys.stdout.isatty():
                sys.stdout.write("\r")
                sys.stdout.flush()
                sys.stdout.write(fmt_str.format(i + 1, primer1_name, j + 1, n_primers))
        if sys.stdout.isatty():
            sys.stdout.write("\r")
            sys.stdout.flush()
    print("\nDone.\n")

    # Materialise the kept alignments in ascending-score order (lowest score == worst).
    alignments = sorted(
        (a for _, _, a in alignments_heap), key=lambda a: a.score
    )

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
