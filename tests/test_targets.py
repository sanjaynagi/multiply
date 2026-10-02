import pytest

from multiply.generate.targets import Target, TargetSet
from multiply.util.exceptions import TargetPositionError


def _target(start, end, ID="t"):
    return Target(chrom="2L", start=start, end=end, ID=ID, name=ID)


def test_pad_is_clamped_at_the_contig_start():
    t = _target(500, 600).calc_pads(max_size_bp=2500)
    assert t.pad_start == 0
    assert t.pad_end == 600 + 1250


def test_a_target_near_the_start_still_counts_as_padded(tmp_path):
    """pad_start == 0 is valid; it must not be mistaken for 'calc_pads not run'."""
    import pysam

    fasta = tmp_path / "ref.fa"
    fasta.write_text(">2L\n" + "A" * 5000 + "\n")
    pysam.faidx(str(fasta))
    t = _target(500, 600).calc_pads(max_size_bp=2500)
    t.extract_seq(str(fasta), include_pads=True)
    assert len(t.seq) == 600 + 1250


def test_extracting_before_calc_pads_is_still_an_error(tmp_path):
    with pytest.raises(ValueError, match="calc_pads"):
        _target(500, 600).extract_seq("unused.fa", include_pads=True)


def test_squeezed_out_target_fails_loudly_not_with_an_empty_sequence():
    t = _target(10_000, 10_500)
    t.pad_start, t.pad_end = 10_600, 10_600
    with pytest.raises(TargetPositionError):
        t.extract_seq("unused.fa")


def test_close_targets_still_share_their_pads():
    a, b = _target(10_000, 10_500, "a"), _target(11_000, 11_500, "b")
    TargetSet([a, b]).check_size_compatible(2500).calc_pads()
    assert a.pad_end < b.pad_start
