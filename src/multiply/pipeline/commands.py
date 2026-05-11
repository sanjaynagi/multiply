import click


@click.command(short_help="Run the full multiply pipeline.")
@click.option(
    "-d",
    "--design",
    type=str,
    required=True,
    help="Path to MULTIPLY design file (e.g. 'designs/pf-default.ini').",
)
def pipeline(design):
    """
    Run the complete `multiply` pipeline

    """
    # MAIN IMPORTS
    from multiply.util.parsing import parse_parameters
    from multiply.generate.main import generate
    from multiply.view.main import view
    from multiply.align.main import align
    from multiply.blast.main import blast
    from multiply.snpcheck.main import snpcheck
    from multiply.select.main import select

    # PARSE INPUT PARAMETERS
    params = parse_parameters(design)
    primer_csv = f"{params['output_dir']}/table.candidate_primers.csv"

    # Pipeline
    generate(design=design)
    # `view` is plotting-only; large tiled panels can defeat matplotlib's
    # GridSpec sizing. Treat as best-effort so the rest of the pipeline still
    # produces the design tables we actually consume downstream.
    try:
        view(result_dir=params["output_dir"], genome_name=params["genome"])
    except Exception as exc:
        print(f"WARNING: `multiply view` failed ({type(exc).__name__}: {exc}); continuing.")
    snpcheck(
        primer_csv=primer_csv,
        genome_name=params["genome"],
        backend_config=params.get("snpcheck"),
    )
    align(primer_csv=primer_csv)
    blast(primer_csv=primer_csv, genome_name=params["genome"])
    select(result_dir=params["output_dir"], algorithm="Greedy")

    # ADD HERE -- write a README.txt
