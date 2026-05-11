import os
import json

import pandas as pd
from functools import partial, reduce

from multiply.util.printing import print_header, print_footer
from multiply.util.dirs import produce_dir
from multiply.util.io import write_primers_to_bed
from multiply.download.collection import genome_collection
from .bedtools import bedtools_intersect


def snpcheck(primer_csv, genome_name, backend_config=None):
    """Annotate candidate primers with cohort SNP load.

    Dispatches to one of two backends:

    - ``bedtools`` (default): intersects primer BED with VCF tracks listed in
      ``genome.include_variation``. Used when no [SNPCheck] section is set.
    - ``malariagen``: queries ``malariagen_data.snp_calls()`` for the design
      region with a configurable cohort and emits per-primer cost features
      directly. Used when ``[SNPCheck] backend = malariagen`` in the design.

    Both backends emit ``snpcheck/table.candidate_primers.snp_counts.csv``
    with a ``primer_name`` column; downstream ``select`` consumes either
    schema via ``IndividualCostFactory`` against the configured cost columns.

    Parameters
    ----------
    primer_csv : str
        Path to ``table.candidate_primers.csv`` from ``multiply generate``.
    genome_name : str
        Genome registry key (used for the bedtools backend's variation files).
    backend_config : dict, optional
        Output of ``add_snpcheck`` from the design parser. ``None`` means
        bedtools.
    """
    backend = (backend_config or {"backend": "bedtools"}).get("backend", "bedtools")
    if backend == "malariagen":
        return _snpcheck_malariagen(
            primer_csv=primer_csv, backend_config=backend_config or {},
        )
    return _snpcheck_bedtools(primer_csv=primer_csv, genome_name=genome_name)


def _snpcheck_bedtools(primer_csv, genome_name):
    """Legacy bedtools-based snpcheck: intersect primer BED with registered VCFs."""
    t0 = print_header("MULTIPLY: Identify SNPs in primers (bedtools backend)")
    input_dir = os.path.dirname(primer_csv)
    output_dir = produce_dir(input_dir, "snpcheck")
    genome = genome_collection[genome_name]

    primer_df = pd.read_csv(primer_csv)

    bed_path = f"{output_dir}/candidate_primers.bed"
    write_primers_to_bed(primer_df, bed_path)

    print("Gathering variation data...")
    if not genome.include_variation:
        print(f"No variation data is available for {genome_name}. Exiting.")
        return
    print(f"  Found data at: {genome.include_variation}")
    variation_dt = json.load(open(genome.include_variation, "r"))
    print(f"  Includes: {', '.join(variation_dt)}")
    print("Done.\n")

    dfs = []
    for pop, pop_fn in variation_dt.items():
        print(f"Looking for intersections with: {pop}")
        print(f"  Variation file: {pop_fn}")
        print("  Intersecting...")
        pop_output_path = f"{output_dir}/primers.snp_counts.{pop}.bed"
        bedtools_intersect(
            a=bed_path, b=pop_fn, flags=["-c"], output_path=pop_output_path
        )
        print(f"  Output: {pop_output_path}")
        df = pd.read_csv(
            pop_output_path,
            names=["chrom", "primer_start", "primer_end", "primer_name", pop],
            sep="\t",
        )
        print(f"  Total No. primers with SNPs: {(df[pop] > 0).sum()}")
        print(f"  Total No. SNPs found in primers: {df[pop].sum()}")
        dfs.append(df)
        print("Done.\n")

    print("Merging and writing summary CSV...")
    output_path = f"{output_dir}/table.candidate_primers.snp_counts.csv"
    print(f"  to: {output_path}")
    merge_func = partial(
        pd.merge, on=["chrom", "primer_start", "primer_end", "primer_name"]
    )
    merged_df = reduce(merge_func, dfs)
    merged_df.to_csv(output_path, index=False)
    print("Done.\n")
    print_footer(t0)


def _snpcheck_malariagen(primer_csv, backend_config):
    """Cohort-aware snpcheck via malariagen_data.snp_calls."""
    from agam_tiled_ampseq.malariagen_snpcheck import (
        score_primers_by_cohort,
        write_snp_counts_csv,
    )

    t0 = print_header("MULTIPLY: Identify SNPs in primers (malariagen backend)")
    input_dir = os.path.dirname(primer_csv)
    output_dir = produce_dir(input_dir, "snpcheck")

    primer_df = pd.read_csv(primer_csv)

    species = backend_config.get("species", "gambiae_sl")
    sample_sets = backend_config.get("sample_sets", "3.0")
    sample_query = backend_config.get("sample_query")
    site_mask = backend_config.get("site_mask", "gamb_colu_arab")
    maf_threshold = float(backend_config.get("maf_threshold", 0.05))
    three_prime_bp = int(backend_config.get("three_prime_bp", 5))
    region_pad_bp = int(backend_config.get("region_pad_bp", 0))

    print("Cohort:")
    print(f"  species       = {species}")
    print(f"  sample_sets   = {sample_sets}")
    print(f"  sample_query  = {sample_query}")
    print(f"  site_mask     = {site_mask}")
    print(f"  maf_threshold = {maf_threshold}")
    print(f"  three_prime_bp = {three_prime_bp}")
    print(f"  region_pad_bp = {region_pad_bp}")
    print(f"  primers       = {len(primer_df)}")

    print("Fetching SNP calls + scoring primers...")
    scores_df = score_primers_by_cohort(
        primer_df=primer_df,
        species=species,
        sample_sets=sample_sets,
        sample_query=sample_query,
        site_mask=site_mask,
        maf_threshold=maf_threshold,
        three_prime_bp=three_prime_bp,
        region_pad_bp=region_pad_bp,
    )

    output_path = f"{output_dir}/table.candidate_primers.snp_counts.csv"
    write_snp_counts_csv(
        primer_df=primer_df, scores_df=scores_df, output_path=output_path,
    )
    print(f"  Wrote: {output_path}")
    print("Per-primer feature summary (over all candidates):")
    print(scores_df.describe().round(4).to_string())
    print("Done.\n")
    print_footer(t0)
