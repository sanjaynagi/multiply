import matplotlib

# Force a non-interactive backend before any sub-module imports pyplot.
# Some downstream packages (e.g. malariagen_data progress bars) pull in
# IPython, which then makes matplotlib try to attach a GUI backend to a
# non-existent IPython kernel and crashes plotting steps.
matplotlib.use("Agg", force=True)

import click
from multiply.download.commands import download
from multiply.generate.commands import generate
from multiply.view.commands import view
from multiply.snpcheck.commands import snpcheck
from multiply.align.commands import align
from multiply.blast.commands import blast
from multiply.select.commands import select
from multiply.pipeline.commands import pipeline


# ================================================================
# Entry point for all sub-commands
#
# ================================================================


@click.group()
def cli():
    """
    Multiplex PCR design, in silico

    """

    pass


# ================================================================
# Individual sub-commands
#
# ================================================================


cli.add_command(download)
cli.add_command(generate)
cli.add_command(view)
cli.add_command(snpcheck)
cli.add_command(align)
cli.add_command(blast)
cli.add_command(select)
cli.add_command(pipeline)


if __name__ == "__main__":
    cli()
