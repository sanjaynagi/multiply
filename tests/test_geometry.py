import pandas as pd

from multiply.select.cost.geometry import AmpliconOverlapCost, amplicon_overlap_matrix


def _cands(rows):
    out = []
    for pair, target, chrom, f, r in rows:
        for direction, start in (("F", f), ("R", r)):
            out.append(dict(pair_name=pair, target_id=target, direction=direction, chrom=chrom, start=start))
    return pd.DataFrame(out)


def test_overlapping_amplicons_on_one_chromosome_are_flagged():
    m = amplicon_overlap_matrix(_cands([("a", "ta", "2L", 100, 1899), ("b", "tb", "2L", 1000, 2799)]))
    assert m.loc["a", "b"] == m.loc["b", "a"] == 1


def test_touching_and_separate_amplicons_are_not_overlaps():
    m = amplicon_overlap_matrix(_cands([("a", "ta", "2L", 0, 999), ("b", "tb", "2L", 1000, 1999), ("c", "tc", "2L", 5000, 6999)]))
    assert m.loc["a", "b"] == 0          # [0,1000) and [1000,2000) abut
    assert m.loc["a", "c"] == 0


def test_the_same_coordinates_on_different_chromosomes_do_not_overlap():
    m = amplicon_overlap_matrix(_cands([("a", "ta", "2L", 0, 1999), ("b", "tb", "3R", 0, 1999)]))
    assert m.loc["a", "b"] == 0


def test_candidates_of_one_target_are_never_penalised_against_each_other():
    m = amplicon_overlap_matrix(_cands([("a1", "t", "2L", 0, 1999), ("a2", "t", "2L", 100, 2099)]))
    assert m.loc["a1", "a2"] == 0


def test_matrix_is_sorted_square_and_symmetric():
    m = amplicon_overlap_matrix(_cands([("b", "tb", "2L", 1000, 2799), ("a", "ta", "2L", 100, 1899)]))
    assert list(m.index) == list(m.columns) == ["a", "b"]
    assert (m.values == m.values.T).all()


def test_cost_is_the_weight_times_the_indicator_without_normalisation():
    m = amplicon_overlap_matrix(_cands([("a", "ta", "2L", 100, 1899), ("b", "tb", "2L", 1000, 2799), ("c", "tc", "2L", 9000, 10799)]))
    cost = AmpliconOverlapCost(cost_name="o", primer_values=m, weight=100.0).collapse_to_per_pair().normalise_costs()
    assert cost.primer_pair_costs.loc["a", "b"] == 100.0
    assert cost.primer_pair_costs.loc["a", "c"] == 0.0
