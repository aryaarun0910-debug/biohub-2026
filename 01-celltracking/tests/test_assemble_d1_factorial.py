"""The fetch-to-probe merge: four kernel outputs -> two bases d1f_probe can read.

These lock structure only -- layout, provenance, joins, determinism, and the basis
invariant from correction C1 that a head fitted in one checkpoint's 32-D basis may never
be applied in the other. No test here encodes a scientific conclusion or a score
threshold; the assembler has no opinion about what the numbers mean.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import assemble_d1_factorial as A  # noqa: E402
import assemble_p3_d1_smoke_spec as ASM  # noqa: E402
import d1f_probe as D  # noqa: E402

VIEWS = ["identity", "rot90_k1", "rot90_k2", "rot90_k3", "flipY", "transpose",
         "flipX", "flipX"]
# stage -> (split, family, role, crops). The corrected smoke: each split carries both
# families, so each basis has a source to fit on and a target to measure.
LAYOUT = {
    1: (1, "44b6", "source", ["44b6_0113de3b"]),
    2: (1, "6bba", "target", ["6bba_57b7cc1e", "6bba_6feb10f0"]),
    3: (0, "6bba", "source", ["6bba_57b7cc1e", "6bba_6feb10f0"]),
    4: (0, "44b6", "target", ["44b6_0113de3b"]),
}
N_ROWS = 12


def _write_cell(root: Path, stage: int, *, role=None, family=None, complete=True,
                scalars=None, observed_ckpt=None, crops=None, split=None) -> Path:
    """One fetched kernel output. Overrides exist so a defect can be expressed."""
    d_split, d_family, d_role, d_crops = LAYOUT[stage]
    split = d_split if split is None else split
    family = d_family if family is None else family
    role = d_role if role is None else role
    crops = d_crops if crops is None else crops
    ckpt = ASM.CKPT_SHA[split]
    shard = f"smoke__s{stage}__split_{split}__{family}__{role}__0"

    d = root / f"cell_s{stage}" / "d1_audit"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(stage)
    crop_recs = {}
    for c in crops:
        pl.DataFrame({"dataset": [c] * N_ROWS, "kind": ["gt_centre"] * N_ROWS,
                      "t": list(range(N_ROWS)), "logit": [0.5] * N_ROWS}
                     ).write_parquet(d / f"{c}__rows.parquet")
        for suffix in D.V6_REQUIRED_FEATURE_FILES:
            np.save(d / f"{c}{suffix}",
                    rng.normal(size=(N_ROWS, D.FEAT_DIM)).astype(np.float32))
        crop_recs[c] = {"status": "complete", "grid_zyx": [16, 64, 64],
                        "n_frames": N_ROWS, "n_uniform_per_frame": 64,
                        "estimated_number_of_nodes": 100, "checkpoint_sha256": ckpt,
                        "split": split}
    man = {"fold": split, "checkpoint_sha256": ckpt, "schema_version": "d1_v6",
           "match_um": 7.0, "search_um": 15.0, "tta_view_set": VIEWS,
           "n_encode_calls": 8, "n_distinct_views": 7,
           "crops": crop_recs, "problems": [] if complete else ["synthetic defect"],
           "COMPLETE": complete}
    man.update(scalars or {})
    (d / "d1_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    (d / "d1_cell.json").write_text(json.dumps({
        "kind": "d1_factorial_cell", "tier": "smoke", "shard_id": shard, "stage": stage,
        "split": split, "fold": split, "family": family, "role": role, "crops": crops,
        "n_crops": len(crops), "checkpoint_sha256": ckpt,
        "checkpoint_sha256_observed": observed_ckpt or ckpt,
        "checkpoint_trained_on_family": ASM.SPLIT_SOURCE_FAMILY[split],
        "checkpoint_heldout_family": ASM.SPLIT_HELDOUT_FAMILY[split],
        "family_rederived": family, "role_rederived": role, "crops_mounted": sorted(crops),
    }, indent=1), encoding="utf-8")
    return d.parent


def _all_cells(root: Path, over: dict | None = None) -> list[Path]:
    over = over or {}
    return [_write_cell(root, s, **over.get(s, {})) for s in (1, 2, 3, 4)]


# ==================================================================================
# 1. THE HAPPY PATH IS THE ONE THE PROBE CAN ACTUALLY READ
# ==================================================================================
def test_four_cells_assemble_into_two_bases_the_probe_loads(tmp_path):
    cells = _all_cells(tmp_path)
    rep = A.assemble(cells, tmp_path / "out", tier="smoke")
    assert [b["split"] for b in rep["bases"]] == [0, 1]

    corpus = D.load_factorial(tmp_path / "out", expect_encode_calls=8)
    assert corpus.bases() == [0, 1]
    # Each basis holds BOTH families -- that is the whole point of the 2x2.
    for split in (0, 1):
        fams = set(corpus.family[corpus.basis_mask(split)])
        assert fams == {"44b6", "6bba"}, f"basis {split} carries only {fams}"
    assert corpus.n == 2 * 3 * N_ROWS


def test_every_crop_carries_the_split_that_encoded_it_not_the_one_that_owns_it(tmp_path):
    """A crop appears in BOTH bases under different checkpoints. encoder_split must name
    the checkpoint that produced the row, or C1's same-basis guarantee is unenforceable."""
    A.assemble(_all_cells(tmp_path), tmp_path / "out", tier="smoke")
    for split in (0, 1):
        man = json.loads((tmp_path / "out" / f"basis_{split}" / "d1_manifest.json")
                         .read_text(encoding="utf-8"))
        assert set(man["crops"]) == {"44b6_0113de3b", "6bba_57b7cc1e", "6bba_6feb10f0"}
        assert {r["encoder_split"] for r in man["crops"].values()} == {split}


def test_the_cell_identity_reaches_the_crop_record(tmp_path):
    """The aggregator records fold and checkpoint but not WHICH factorial cell produced a
    row. Without the stamp, a merged basis cannot say where any row came from."""
    A.assemble(_all_cells(tmp_path), tmp_path / "out", tier="smoke")
    man = json.loads((tmp_path / "out" / "basis_1" / "d1_manifest.json")
                     .read_text(encoding="utf-8"))
    assert man["crops"]["44b6_0113de3b"]["role"] == "source"
    assert man["crops"]["6bba_57b7cc1e"]["role"] == "target"
    assert man["crops"]["44b6_0113de3b"]["shard_id"].endswith("44b6__source__0")


def test_the_merged_manifest_never_carries_the_retired_view_field(tmp_path):
    A.assemble(_all_cells(tmp_path), tmp_path / "out", tier="smoke")
    for split in (0, 1):
        man = json.loads((tmp_path / "out" / f"basis_{split}" / "d1_manifest.json")
                         .read_text(encoding="utf-8"))
        assert "n_views" not in man
        assert man["n_encode_calls"] == 8 and man["n_distinct_views"] == 7


def test_assembly_is_deterministic(tmp_path):
    cells = _all_cells(tmp_path)
    a = A.assemble(cells, tmp_path / "a", tier="smoke")
    b = A.assemble(cells, tmp_path / "b", tier="smoke")
    for split in (0, 1):
        pa = (tmp_path / "a" / f"basis_{split}" / "d1_manifest.json")
        pb = (tmp_path / "b" / f"basis_{split}" / "d1_manifest.json")
        assert pa.read_text(encoding="utf-8").replace(str(tmp_path / "a"), "") == \
               pb.read_text(encoding="utf-8").replace(str(tmp_path / "b"), "")
    assert [x["files"] for x in a["bases"]] == [x["files"] for x in b["bases"]]


def test_an_existing_basis_is_not_silently_overwritten(tmp_path):
    cells = _all_cells(tmp_path)
    A.assemble(cells, tmp_path / "out", tier="smoke")
    with pytest.raises(A.AssemblyError, match="--force"):
        A.assemble(cells, tmp_path / "out", tier="smoke")
    A.assemble(cells, tmp_path / "out", tier="smoke", force=True)


# ==================================================================================
# 2. THE DEFECTS THAT MUST BLOCK
# ==================================================================================
def test_a_routed_only_fetch_is_refused_and_says_why(tmp_path):
    """Every cell a TARGET is the original defect: each family under the checkpoint that
    held it out. The probe already refuses it; the assembler must refuse it earlier."""
    cells = [_write_cell(tmp_path, 1, split=0, family="44b6", role="target",
                         crops=["44b6_0113de3b"]),
             _write_cell(tmp_path, 2, split=1, family="6bba", role="target",
                         crops=["6bba_57b7cc1e"])]
    with pytest.raises(A.AssemblyError, match="one source and one target"):
        A.assemble(cells, tmp_path / "out")


def test_a_relabelled_role_is_refused(tmp_path):
    cells = _all_cells(tmp_path, {1: {"role": "target"}})
    with pytest.raises(A.AssemblyError, match="ROLE INVERSION"):
        A.assemble(cells, tmp_path / "out")


def test_an_incomplete_cell_is_refused(tmp_path):
    cells = _all_cells(tmp_path, {3: {"complete": False}})
    with pytest.raises(A.AssemblyError, match="COMPLETE"):
        A.assemble(cells, tmp_path / "out")


def test_weights_that_differ_from_the_pin_are_refused(tmp_path):
    cells = _all_cells(tmp_path, {2: {"observed_ckpt": "f" * 64}})
    with pytest.raises(A.AssemblyError, match="loaded weights"):
        A.assemble(cells, tmp_path / "out")


def test_cells_of_one_basis_that_disagree_on_a_scalar_are_refused(tmp_path):
    """One basis is one measurement. Two match radii inside it are not mergeable."""
    cells = _all_cells(tmp_path, {2: {"scalars": {"match_um": 5.0}}})
    with pytest.raises(A.AssemblyError, match="disagree on match_um"):
        A.assemble(cells, tmp_path / "out")


def test_a_view_count_disagreement_inside_a_basis_is_refused(tmp_path):
    cells = _all_cells(tmp_path, {4: {"scalars": {"n_encode_calls": 4}}})
    with pytest.raises(A.AssemblyError, match="disagree on n_encode_calls"):
        A.assemble(cells, tmp_path / "out")


def test_a_partial_factorial_is_refused_when_a_tier_is_named(tmp_path):
    cells = [_write_cell(tmp_path, s) for s in (1, 2, 3)]
    with pytest.raises(A.AssemblyError, match="A partial factorial is not a factorial"):
        A.assemble(cells, tmp_path / "out", tier="smoke")


def test_a_single_split_is_refused(tmp_path):
    cells = [_write_cell(tmp_path, s) for s in (1, 2)]
    with pytest.raises(A.AssemblyError, match="the factorial needs both"):
        A.assemble(cells, tmp_path / "out")


def test_two_cells_claiming_the_same_crop_collide_loudly(tmp_path):
    """Within a basis the families are disjoint, so a filename collision means the cells
    are not what they say. Silently overwriting would mix two encodings into one file."""
    cells = _all_cells(tmp_path, {2: {"family": "6bba", "crops": ["6bba_57b7cc1e"]},
                                    1: {"family": "6bba", "role": "target",
                                        "crops": ["6bba_57b7cc1e"]}})
    with pytest.raises(A.AssemblyError):
        A.assemble(cells, tmp_path / "out")


def test_an_output_that_predates_the_identity_block_is_refused(tmp_path):
    cells = _all_cells(tmp_path)
    (Path(cells[0]) / "d1_audit" / "d1_cell.json").unlink()
    with pytest.raises(A.AssemblyError, match="no d1_cell.json"):
        A.assemble(cells, tmp_path / "out")


def test_a_duplicate_shard_is_refused(tmp_path):
    cells = _all_cells(tmp_path)
    with pytest.raises(A.AssemblyError, match="supplied twice"):
        A.assemble(cells + [cells[0]], tmp_path / "out")


def test_a_cell_missing_a_promoted_scalar_is_refused_before_the_probe_sees_it(tmp_path):
    """Assembling around a missing field would only move the failure to load time, after
    the GPU has been paid for."""
    cells = _all_cells(tmp_path)
    for idx in (0, 1):                       # both halves of basis 1
        d = Path(cells[idx]) / "d1_audit" / "d1_manifest.json"
        man = json.loads(d.read_text(encoding="utf-8"))
        del man["tta_view_set"]
        d.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(A.AssemblyError, match="neither cell records tta_view_set"):
        A.assemble(cells, tmp_path / "out")


def test_one_half_of_a_basis_losing_a_scalar_is_a_disagreement_not_a_default(tmp_path):
    """Falling back to the half that still declares it would silently attribute one
    cell's radii to the other."""
    cells = _all_cells(tmp_path)
    d = Path(cells[1]) / "d1_audit" / "d1_manifest.json"
    man = json.loads(d.read_text(encoding="utf-8"))
    del man["search_um"]
    d.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(A.AssemblyError, match="disagree on search_um"):
        A.assemble(cells, tmp_path / "out")
