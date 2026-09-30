r"""Software contracts for PKT-0048's mask/HOCT groundwork.

SCOPE, per CLAUDE.md rule 4: these enforce SOFTWARE contracts only. No promotion decision is
made here - the scientific verdicts live in the packet's ``result`` and in the payloads under
``_evidence/maskassoc/``.

The contracts under test are the ones whose failure would be SILENT:
  * a zero-filled feature slot must be REFUSED, before and after standardisation;
  * the unit convention must be stated, never defaulted;
  * the FOCUS-3D loader must fail closed on absent weights (FACT-0425);
  * the compiled tau must be re-derived from the artifact, not restated from the paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

from hoct_feature_contract import (  # noqa: E402
    PUBLISHER_MEAN,
    PUBLISHER_STD,
    SLOTS,
    FeatureRefusal,
    audit_matrix,
    build_node_feats,
    mutation_demo,
    prove_slot_map,
)


# ---------------------------------------------------------------------------------------------
# The slot map
# ---------------------------------------------------------------------------------------------
def test_slot_map_is_proved_against_the_publishers_own_constants():
    proof = prove_slot_map()
    failed = [c["check"] for c in proof["checks"] if not c["passes"]]
    assert proof["all_passed"], f"slot-map checks failed: {failed}"


def test_there_are_nineteen_slots_and_fourteen_need_masks():
    assert len(SLOTS) == 19
    blocked = [s for s in SLOTS if not s.computable_today]
    assert len(blocked) == 14, "FACT-0454 records 15; read at source it is 14 - border_dist is "
    assert "border_dist" not in {s.name for s in blocked}


def test_mean_and_std_widths_match_the_slot_count():
    assert len(PUBLISHER_MEAN) == len(PUBLISHER_STD) == len(SLOTS) == 19


# ---------------------------------------------------------------------------------------------
# The refusal, which is the packet's falsifier
# ---------------------------------------------------------------------------------------------
def _varying(n: int = 32) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, 19)) * 10.0 + np.arange(19)[None, :]


def test_clean_matrix_passes():
    assert audit_matrix(_varying())["verdict"] == "PASS"


@pytest.mark.parametrize("index", [4, 5, 9, 17])
def test_raw_zero_fill_is_refused_and_named(index: int):
    arr = _varying()
    arr[:, index] = 0.0
    out = audit_matrix(arr)
    assert out["verdict"] == "REFUSE"
    assert out["offending_slots"] == [SLOTS[index].name]


@pytest.mark.parametrize("index", [4, 9, 13])
def test_standardized_zero_fill_is_refused_even_though_it_is_not_zero(index: int):
    """The post-Standardize form of a zero fill is the constant ``-mean/std``.

    A shape check, a NaN check and a 'contains no zeros' check all pass on this matrix. Only a
    degeneracy check catches it, which is why the audit is a degeneracy check.
    """
    arr = (_varying() - np.asarray(PUBLISHER_MEAN)) / np.asarray(PUBLISHER_STD)
    fill = -PUBLISHER_MEAN[index] / PUBLISHER_STD[index]
    arr[:, index] = fill
    assert abs(fill) > 1e-3, "this test is vacuous unless the fill is genuinely nonzero"
    out = audit_matrix(arr, standardized=True)
    assert out["verdict"] == "REFUSE"
    assert out["offending_slots"] == [SLOTS[index].name]


def test_builder_refuses_points_only_columns_naming_all_fourteen():
    cols = {k: np.arange(8.0) for k in ("t", "z", "y", "x", "border_dist")}
    with pytest.raises(FeatureRefusal) as exc:
        build_node_feats(cols, 8)
    assert len(exc.value.slots) == 14


def test_audit_refuses_a_single_row_rather_than_guessing():
    out = audit_matrix(np.ones((1, 19)))
    assert out["verdict"] == "REFUSE"


def test_audit_refuses_a_wrong_width():
    assert audit_matrix(np.ones((10, 18)))["verdict"] == "REFUSE"


def test_the_committed_mutation_demo_is_correct_on_every_arm():
    demo = mutation_demo()
    wrong = [a["arm"] for a in demo["arms"] if not a["correct"]]
    assert demo["all_correct"], f"mutation arms wrong: {wrong}"


# ---------------------------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------------------------
def test_adapter_refuses_an_unstated_unit_convention():
    from mask_node_adapter import instances_to_nodes

    lab = np.zeros((4, 8, 8), dtype=np.int32)
    lab[1:3, 2:5, 2:5] = 1
    with pytest.raises(FeatureRefusal):
        instances_to_nodes(lab, t=0, convention="whatever_is_default")


def test_focus3d_loader_fails_closed_on_absent_weights():
    from mask_node_adapter import LoaderRefusal, load_focus3d_strict

    with pytest.raises(LoaderRefusal) as exc:
        load_focus3d_strict(object(), Path("C:/temp/definitely_not_here.pth"))
    assert "weights_absent" in str(exc.value)


def test_identity_downsample_guard_holds():
    """(1.625, 0.40625, 0.40625) * (1, 4, 4) is isotropic; full resolution is not."""
    from mask_node_adapter import assert_not_the_identity_downsample

    g = assert_not_the_identity_downsample()
    assert g["detector_grid_is_isotropic"] is True
    assert g["full_res_grid_is_isotropic"] is False
    assert g["passes"] is True


def test_adapter_produces_a_full_nineteen_vector_from_masks():
    """The whole point of the mask route: 14 refused slots become computed ones."""
    from mask_node_adapter import adapter_selftest

    r = adapter_selftest()
    assert r["with_intensity"]["audit_verdict"] == "PASS"
    assert r["without_intensity"]["passes"], r["without_intensity"]
    assert r["anisotropy_witness"]["passes"], r["anisotropy_witness"]


# ---------------------------------------------------------------------------------------------
# The scale gate
# ---------------------------------------------------------------------------------------------
def test_every_convention_declares_its_own_micron_per_unit():
    from hoct_scale_gate import CONVENTIONS

    for name, cfg in CONVENTIONS.items():
        assert "factors_zyx" in cfg and "um_per_unit" in cfg, name
        assert len(cfg["factors_zyx"]) == 3, name


def test_compiled_tau_is_re_derived_from_the_artifact_not_the_paper():
    from hoct_scale_gate import CHECKPOINT, read_compiled_tau

    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint absent at {CHECKPOINT}")
    info = read_compiled_tau()
    assert info["found"], info
    assert info["squared_cutoff"] == 90000.0
    assert info["tau_in_node_pos_units"] == pytest.approx(300.0)
    # FACT-0453, re-derived without executing the graph.
    assert info["orphan_output_is_new_zeros"] is True
    # 300 is also the RoPE positional normaliser, which is what makes it the coordinate unit.
    assert info["tau_appears_twice"] is True
    assert info["rope_constants"]["c2_double"] == pytest.approx(3.0 ** 0.5)


def test_scale_gate_refuses_when_the_artifact_is_missing(tmp_path):
    from hoct_scale_gate import read_compiled_tau

    out = read_compiled_tau(tmp_path / "nope.pt")
    assert out["found"] is False
