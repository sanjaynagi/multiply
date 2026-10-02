from multiply.generate.primers import load_primer_pairs_from_primer3_output


def test_output_without_a_pair_count_means_no_primers(tmp_path):
    """primer3 rejects some inputs (e.g. a target past the contig end) without a count."""
    path = tmp_path / "x.output"
    path.write_text("PRIMER_ERROR=Missing SEQUENCE tag\n=\nSEQUENCE_ID=t\nPRIMER_ERROR=Target beyond end of sequence\n=\n")
    assert load_primer_pairs_from_primer3_output(str(path)) == []


def test_zero_pairs_returned_means_no_primers(tmp_path):
    path = tmp_path / "x.output"
    path.write_text("PRIMER_PAIR_NUM_RETURNED=0\n=\n")
    assert load_primer_pairs_from_primer3_output(str(path)) == []
