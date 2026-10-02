import pandas as pd

from multiply.blast.annotator import complete_summary


def _summary():
    return pd.DataFrame({
        "primer_name": ["a_u0_F", "a_u0_R"],
        "primer_pair_name": ["a_u0", "a_u0"],
        "target_name": ["a", "a"],
        "total_alignments": [5, 9],
        "predicted_bound": [2, 7],
    })


def test_a_primer_with_no_blast_hit_is_charged_the_worst_value_not_zero():
    out = complete_summary(_summary(), ["a_u0_F", "a_u0_R", "b_u1_F"]).set_index("primer_name")
    assert out.loc["b_u1_F", "predicted_bound"] == 7
    assert out.loc["b_u1_F", "total_alignments"] == 9
    assert out.loc["b_u1_F", "primer_pair_name"] == "b_u1"
    assert out.loc["b_u1_F", "target_name"] == "b"


def test_vetted_primers_keep_their_values_and_order():
    out = complete_summary(_summary(), ["a_u0_R", "a_u0_F"])
    assert list(out["primer_name"]) == ["a_u0_R", "a_u0_F"]
    assert list(out["predicted_bound"]) == [7, 2]
