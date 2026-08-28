from scripts.win_bet.compare_substrate_sentinel import compare


def _summary(delta=0.001, sha="same", n=10):
    return {"params_sha256": sha, "n_crops": n,
            "delta": {"score": delta, "adj_edge_jaccard": delta,
                      "division_jaccard": 0.0}}


def test_equivalence_is_about_paired_delta_not_absolute_level():
    old = _summary(0.001)
    new = _summary(0.001004)
    assert compare(old, new, 1e-5)["equivalent"] is True
    assert compare(old, new, 1e-6)["equivalent"] is False


def test_mismatched_params_or_crop_population_fail_closed():
    import pytest
    with pytest.raises(ValueError, match="parameter"):
        compare(_summary(), _summary(sha="other"), 1e-5)
    with pytest.raises(ValueError, match="population"):
        compare(_summary(), _summary(n=11), 1e-5)
