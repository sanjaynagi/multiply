import numpy as np

from multiply.util.statistics import get_array_encoding, get_homopolymer_runs


def test_homopolymer_run_longer_than_127_does_not_overflow():
    """The run array was int8; primers with long tails overflow it."""
    runs = get_homopolymer_runs("A" * 200)
    assert (runs == 200).all()


def test_ambiguous_bases_encode_as_unknown_not_a_key_error():
    a = get_array_encoding("ACNGT")
    assert a.shape[1] == 5
    assert a[:, 2].sum() == 0            # the N column is all zeros
    assert a[:, 0].sum() == 1


def test_lowercase_bases_are_encoded():
    assert (get_array_encoding("acgt") == get_array_encoding("ACGT")).all()
