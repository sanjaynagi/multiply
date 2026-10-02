import pandas as pd
from collections import namedtuple


class BlastResultsAnnotator:
    def __init__(self, blast_df):
        """
        Annotate tabular results produced by running BLAST with
        `-outfmt 6`.

        params
            blast_df: DataFrame (n_hits, n_columns)
                Each row contains a BLAST hit.

        """
        self.blast_df = blast_df

    def build_annotation_dict(self, length_threshold=12, evalue_threshold=4):
        """
        Build a dictionary specifying conditions for annotation

        """
        self.annotations = {
            "from_3prime": lambda row: row["qend"] == row["length"],
            "length_pass_3prime": lambda row: row["length"] >= length_threshold
            and row["pident"] == 100
            and row["from_3prime"],
            "evalue_pass_3prime": lambda row: row["evalue"] < evalue_threshold
            and row["from_3prime"],
            "predicted_bound": lambda row: row["length_pass_3prime"]
            or row["evalue_pass_3prime"],
        }

    def add_annotations(self):
        """
        Add annotations to `blast_df`

        """
        for annot_name, annot_func in self.annotations.items():
            self.blast_df.insert(
                self.blast_df.shape[1],
                annot_name,
                self.blast_df.apply(func=annot_func, axis=1),
            )

    def get_predicted_bound(self):
        """
        Get all BLAST matches where we predict that the query
        sequence will bind

        """
        if "predicted_bound" not in self.blast_df:
            raise ValueError(
                "`blast_df` must contain a column `predicted_bound`. Try running `.add_annotations()` first."
            )

        return self.blast_df.query("predicted_bound")

    def summarise_by_primer(self, output_path=None):
        """
        Summarise the BLAST results on a per-primer
        basis

        In essence, convert from a dataframe where each row
        is a BLAST hit, to a dataframe where each row
        is a primer, and columns give the total number of hits
        for that primer.

        """

        # Define structure to hold per-primer summaries
        PrimerBlastRecord = namedtuple(
            "PrimerBlastRecord",
            ["primer_name", "primer_pair_name", "target_name", "total_alignments"]
            + list(self.annotations),
        )

        # Collect summaries for every primer
        primer_blast_records = [
            PrimerBlastRecord(
                primer_name=qseqid,
                primer_pair_name=qseqid[:-2],
                target_name=qseqid.split("_")[0],
                total_alignments=qseqid_df.shape[0],
                **qseqid_df[list(self.annotations.keys())].sum().to_dict(),  # Use keys as column names
            )
            for qseqid, qseqid_df in self.blast_df.groupby("qseqid")
        ]

        # Convert to data frame
        self.blast_primer_df = pd.DataFrame(primer_blast_records).sort_values(
            "predicted_bound", ascending=False
        )

        # Optionally write
        if output_path is not None:
            self.blast_primer_df.to_csv(output_path, index=False)


def complete_summary(summary_df, primer_names):
    """
    Give every candidate primer a row in a per-primer BLAST summary.

    BLAST can return no hit at all for a primer (it should at least find its
    own site, but short low-complexity primers are filtered out). Such a primer
    is absent from the summary, and a missing value reads as "no off-target
    sites", which makes the primer look better than every primer that was
    actually vetted. Charge it the worst value observed instead, so a primer
    that could not be checked is never preferred.

    params
        summary_df: pandas DataFrame
            Per-primer summary with a `primer_name` column and numeric counts.
        primer_names: iterable of str
            Every candidate primer.

    returns
        pandas DataFrame
            One row per primer in `primer_names`, in that order.

    """
    import pandas as pd

    summary_df = summary_df.set_index("primer_name")
    numeric = summary_df.select_dtypes("number").columns
    worst = summary_df[numeric].max()
    out = summary_df.reindex(list(primer_names))
    missing = out[numeric[0]].isna()
    for column in numeric:
        out.loc[missing, column] = worst[column]
    out.loc[missing, "primer_pair_name"] = [n[:-2] for n in out.index[missing]]
    out.loc[missing, "target_name"] = [n.split("_")[0] for n in out.index[missing]]
    out[numeric] = out[numeric].astype(summary_df[numeric].dtypes.to_dict())
    return out.reset_index(names="primer_name")
