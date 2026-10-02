import os
import concurrent.futures
import pandas as pd

from multiply.util.printing import print_header, print_footer, print_parameters
from multiply.util.parsing import parse_parameters
from multiply.util.exceptions import NoPrimersFoundException
from multiply.util.dirs import produce_dir, check_output_dir_overwrite
from multiply.util.io import load_bed_as_dataframe
from multiply.download.collection import genome_collection
from multiply.generate.targets import Target, TargetSet
from multiply.generate.primer3 import Primer3Runner
from multiply.generate.primers import load_primer_pairs_from_primer3_output


def _primer3_one(args):
    """Run primer3 for a single (setting, target) pair; designed for thread pool.

    Each worker constructs its own ``Primer3Runner`` to avoid sharing the
    runner's mutable state across threads. ``primer3_core`` is a subprocess
    so the GIL doesn't bottleneck — threads (rather than processes) are
    enough and avoid fork overhead.
    """
    setting_name, target, output_dir, min_size_bp, max_size_bp = args
    runner = Primer3Runner()
    runner.load_primer3_settings(setting_name)
    runner.set_amplicon_size_ranges(
        min_size_bp=min_size_bp, max_size_bp=max_size_bp
    )
    runner.set_target(
        ID=target.ID,
        seq=target.seq,
        start=target.clear_start,
        pad_start=target.pad_start,
        length=target.clear_length,
    )
    runner.run(output_dir=output_dir)
    primer_pairs = load_primer_pairs_from_primer3_output(
        runner.output_path, add_target=target
    )
    return target.ID, primer_pairs


def generate(design):
    """
    Generate a pool of candidate primers for a given `design` using
    primer3

    Visit the `settings/primer3` directory to observe or change primer3
    settings.

    """
    # PARSE CLI
    t0 = print_header(
        "MULTIPLY: Generate candidate primers for all targets using Primer3"
    )
    params = parse_parameters(design)
    genome = genome_collection[params["genome"]]
    # Create output directory
    check_output_dir_overwrite(params["output_dir"])
    _ = produce_dir(params["output_dir"])
    # Print to user
    print_parameters(design, params)

    # EXTRACT GENE INFORMATION
    print("Preparing targets...")
    genes = []
    if params["from_genes"]:
        gene_df = pd.read_csv(genome.gff_path)
        target_ids = params["target_ids"]
        gene_df.query("ID in @target_ids", inplace=True)
        genes = [Target.from_series(row) for _, row in gene_df.iterrows()]

        # Pretty ugly, would be nice to encapsulate
        for gene in genes:
            gene.name = params["target_id_to_name"][gene.ID]
    print(f"  Found {len(genes)} gene(s).")

    # EXTRACT REGION INFORMATION
    regions = []
    if params["from_regions"]:
        region_df = load_bed_as_dataframe(params["region_bed"])
        regions = [Target.from_series(row) for _, row in region_df.iterrows()]
    print(f"  Found {len(regions)} region(s).")

    # MERGE
    print("  Merging genes and regions...")
    targets = genes + regions
    clearance_bp = params["target_clearance_bp"]
    target_set = TargetSet(targets)
    if params["max_target_bp"] is not None:
        target_set.split_long_targets(params["max_target_bp"])
        print(f"  {len(target_set.targets)} target(s) after splitting at {params['max_target_bp']}bp.")
    target_set = (
        target_set
        .check_size_compatible(params["max_size_bp"], clearance_bp=clearance_bp)
        .calc_pads(
            clearance_bp=clearance_bp, adjust_overlaps=params["adjust_overlapping_pads"]
        )
        .extract_seqs(genome.fasta_path, include_pads=True)
        .to_csv(f"{params['output_dir']}/table.targets_overview.csv")
        .to_fasta(f"{params['output_dir']}/targets_sequence.fasta")
    )
    print("Done.\n")

    # RUN PRIMER3 — parallel across (setting, target) pairs.
    print("Running primer3...")
    primer3_output_dir = produce_dir(params["output_dir"], "primer3")

    primer_pair_dt = {target.ID: [] for target in target_set.targets}

    jobs = [
        (
            primer3_setting,
            target,
            primer3_output_dir,
            params["min_size_bp"],
            params["max_size_bp"],
        )
        for primer3_setting in params["primer3_settings"]
        for target in target_set.targets
    ]

    n_workers = os.cpu_count() or 4
    print(
        f"  Dispatching {len(jobs)} primer3 jobs ({len(params['primer3_settings'])} settings × {len(target_set.targets)} targets) "
        f"across {n_workers} threads..."
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        for target_id, primer_pairs in executor.map(_primer3_one, jobs):
            primer_pair_dt[target_id].extend(primer_pairs)
    print("Done.\n")

    # REDUCE TO UNIQUE PAIRS
    print("Primer pairs discovered by primer3:")
    print(f"  {'Target':<15} {'Total':<10} {'Unique':<10}")
    for target_id, all_primer_pairs in primer_pair_dt.items():

        # Reduce to unique primer pairs, and give names
        # this definitely can be cleaned
        # Sorted, not a bare set: set order depends on the process hash seed, so
        # pair names (`..._u3`) would differ between runs of the same design.
        uniq_primer_pairs = sorted(set(all_primer_pairs), key=lambda pair: pair.pair_id)
        for ix, pair in enumerate(uniq_primer_pairs):
            pair.give_primers_names(primer_code=params["primer_code"], primer_ix=ix)

        # Print summary
        print(
            f"  {target_id:<15} {len(all_primer_pairs):<10} {len(uniq_primer_pairs):<10}"
        )

        # Store
        primer_pair_dt[target_id] = uniq_primer_pairs
    print("Done.\n")

    # Tiled-geometry support: skip-and-continue on per-target primer3 failure.
    # At ~2,100-target panel scale a few rare misses are inevitable; abort would
    # waste the run. Drop the failing targets and emit a warning + summary.
    failed_target_ids = [tid for tid, pairs in primer_pair_dt.items() if len(pairs) == 0]
    if failed_target_ids:
        print(
            f"WARNING: {len(failed_target_ids)} target(s) had no primer3 hits and "
            f"will be dropped from the design: {', '.join(failed_target_ids)}"
        )
        for tid in failed_target_ids:
            del primer_pair_dt[tid]
    if not primer_pair_dt:
        raise NoPrimersFoundException(
            "No primer pairs were found for ANY target. "
            "Consider changing primer3 settings (found in 'settings/primer3'), "
            "`max_size_bp` in [Amplicons] in your design file, or your targets."
        )

    # Add tails, if provided
    if params["include_tails"]:
        print("Adding tails to discovered primers...")
        for target_id, primer_pairs in primer_pair_dt.items():
            for primer_pair in primer_pairs:
                primer_pair.F.add_tail(params["F_tail"])
                primer_pair.R.add_tail(params["R_tail"])
        print("Done.\n")

    # WRITE
    print("Writing output table...")
    primer_df = pd.DataFrame(
        [
            pair.get_primer_as_dict(direction)
            for _, primer_pairs in primer_pair_dt.items()
            for pair in primer_pairs
            for direction in ["F", "R"]
        ]
    )
    primer_df = primer_df[
        [
            "target_id",
            "target_name",
            "pair_name",
            "primer_name",
            "direction",
            "seq",
            "length",
            "tm",
            "gc",
            "chrom",
            "start",
            "product_bp",
            "pair_penalty",
        ]
    ]
    output_csv = f"{params['output_dir']}/table.candidate_primers.csv"
    primer_df.to_csv(output_csv, index=False)
    print(f"  to: {output_csv}")
    print("Done.\n")

    print_footer(t0)
