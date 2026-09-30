import pytest
from multiply.generate.targets import Target, TargetSet
from multiply.generate.primer3 import Primer3Runner
from multiply.util.exceptions import TargetSizeError


def _target(start, end, ID="t"):
    return Target(chrom="2L", start=start, end=end, ID=ID, name=ID)


def test_short_target_is_not_split():
    ts = TargetSet([_target(10_000, 11_000)]).split_long_targets(1500)
    assert [(t.start, t.end, t.ID) for t in ts.targets] == [(10_000, 11_000, "t")]


@pytest.mark.parametrize("length", [1501, 3000, 4999, 12_345])
def test_split_targets_abut_and_cover_exactly(length):
    ts = TargetSet([_target(10_000, 10_000 + length)]).split_long_targets(1500)
    parts = ts.targets
    assert parts[0].start == 10_000 and parts[-1].end == 10_000 + length
    assert all(a.end == b.start for a, b in zip(parts, parts[1:]))
    assert all(0 < t.length <= 1500 for t in parts)
    assert len({t.ID for t in parts}) == len(parts)


def test_split_parts_are_near_equal():
    ts = TargetSet([_target(0, 3001)]).split_long_targets(1500)
    lengths = [t.length for t in ts.targets]
    assert max(lengths) - min(lengths) <= 1


def test_split_rejects_nonpositive_limit():
    with pytest.raises(ValueError):
        TargetSet([_target(0, 10)]).split_long_targets(0)


def test_pads_are_measured_from_the_cleared_interval():
    ts = TargetSet([_target(10_000, 10_001)]).check_size_compatible(2500, clearance_bp=150).calc_pads(clearance_bp=150)
    (t,) = ts.targets
    assert t.pad_start == 10_000 - 150 - 1250
    assert t.pad_end == 10_001 + 150 + 1250


def test_clearance_counts_towards_the_size_check():
    with pytest.raises(TargetSizeError):
        TargetSet([_target(0, 2300)]).check_size_compatible(2500, clearance_bp=150)
    TargetSet([_target(0, 2200)]).check_size_compatible(2500, clearance_bp=150)


def test_primer3_target_includes_the_clearance():
    """primer3 must keep primers off [start - clearance, end + clearance)."""
    t = _target(10_000, 10_001)
    t.calc_pads(2500, clearance_bp=150)
    runner = Primer3Runner()
    runner.settings = {}
    runner.set_target(ID=t.ID, seq="N" * 10, pad_start=t.pad_start, start=t.clear_start, length=t.clear_length)
    assert runner.target_specific_settings["SEQUENCE_TARGET"] == f"1250,{1 + 300}"  # cleared interval starts 1250 bp into the pad


def test_zero_clearance_is_the_previous_behaviour():
    t = _target(10_000, 10_500)
    t.calc_pads(2500)
    assert (t.clear_start, t.clear_length) == (10_000, 500)
    assert (t.pad_start, t.pad_end) == (8750, 11_750)
