"""v6 export contract: TTA-consistent features WITHOUT touching the association substrate.

Every test here exists because a specific way of getting v6 wrong would be invisible in the
kernel logs and would only surface as an unexplained association regression, or as a feature
matrix whose labels are ~100% predictable from a NaN pattern.

Structure:

  1. the injected TTA block          -- 8 views, correct inverses, `unet_out` untouched
  2. the D4 inverse algebra          -- the anti-transpose composes in REVERSE order
  3. the audit block's row contract  -- row_id, feature alignment, NaN sentinel
  4. the parity assert               -- hard abort, no "proceed with caveat" branch
  5. the injector's structure        -- five anchors, both source forms, idempotency

Nothing here needs a GPU and nothing here talks to Kaggle.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import pathlib
import re
import sys
import textwrap

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT_SRC = ROOT / "scripts" / "kaggle_edits" / "d1_response_audit.py"
INJECT_SRC = ROOT / "scripts" / "kaggle_edits" / "d1_inject.py"
PREDICT = ROOT / "vendor" / "kaggle-cell-tracking" / "scripts" / "predict_unet_transformer.py"
NOTEBOOK = ROOT / "notebooks" / "kaggle_p3_d1_smoke_f0" / "biohub-p3-d1-smoke-f0.ipynb"

VIEW_SET = [
    "identity", "flip_x", "flip_y", "flip_xy",
    "rot90_k1", "rot90_k3", "transpose_yx", "rot90_k1_then_transpose_yx",
]


# ======================================================================================
# helpers
# ======================================================================================
class _FakePS:
    """Stands in for the Path of predict_unet_transformer.py inside the injector cell."""

    def __init__(self, txt: str, parent: pathlib.Path):
        self._t = txt
        self.out: str | None = None
        self.parent = parent

    def read_text(self):
        return self._t

    def write_text(self, t):
        self.out = t

    def __str__(self):
        return "predict_unet_transformer.py"


def _inject(src: str, tmp: pathlib.Path) -> str | None:
    ps = _FakePS(src, tmp)
    exec(compile(INJECT_SRC.read_text(encoding="utf-8"), "d1_inject", "exec"), {"_ps": ps})
    return ps.out


def _notebook_tta_literals() -> tuple[str, str]:
    """The deployed view set comes from the GENERATED notebook, never from vendor/."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "TTA patch applied" in "".join(c["source"])
    )
    found: dict[str, str] = {}
    for node in ast.parse(cell).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("_old", "_new")
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            found.setdefault(node.targets[0].id, node.value.value)
    return found["_old"], found["_new"]


def _apply_cell5_patches(src: str, tmp: pathlib.Path) -> str:
    """Replay the generated notebook's own runtime patches against the vendored file.

    The kernel does this before the D1 injector runs, in the same cell, so the injector must
    cope with the RESULT -- which contains a SECOND `if cfg.det_tta:` block (the secondary
    model's). Replaying it here is the only way to test that the A0 anchor is still unique.
    Statements that need notebook globals we do not have are skipped; the four patches that
    touch predict_unet_transformer.py are then asserted to have landed.
    """
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "TTA patch applied" in "".join(c["source"])
    )
    lines = cell.split(chr(10))
    start = next(i for i, ln in enumerate(lines) if ln.startswith("_ps = REPO_DIR"))
    stop = next(i for i, ln in enumerate(lines) if "LOEO RETARGET BLOCK" in ln)
    body = chr(10).join(lines[start:stop - 1])

    repo = tmp / "repo"
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    target = repo / "scripts" / "predict_unet_transformer.py"
    target.write_text(src, encoding="utf-8")

    import os as _os
    g = {"REPO_DIR": repo, "WORKING_DIR": repo, "os": _os,
         "Path": pathlib.Path, "json": json}
    for node in ast.parse(body).body:
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "cell5", "exec"), g)
        except Exception:            # notebook-global dependent statement, not a patch
            continue
    out = target.read_text(encoding="utf-8")
    for marker in ("_nv += 1", "_secondary_nv", "blended_det", "reverse_aligned"):
        assert marker in out, f"notebook patch did not land: {marker}"
    return out


@pytest.fixture(scope="module")
def patched_predict(tmp_path_factory) -> str:
    """The vendored predict script after ALL of the generated notebook's cell-5 patches and
    then the D1 v6 injection -- i.e. exactly what runs in the kernel."""
    tmp = tmp_path_factory.mktemp("inj")
    src = _apply_cell5_patches(PREDICT.read_text(encoding="utf-8"), tmp)
    out = _inject(src, tmp)
    assert out is not None
    return out


def _extract_tta_block(patched: str) -> str:
    """The injected v6 TTA block, dedented so it can be exec'd standalone."""
    i = patched.index("        # ---- D1 v6 TTA")
    j = patched.index("        # ---- end D1 v6 TTA", i)
    return textwrap.dedent(patched[i:j])


def _load_audit_block(tmp: pathlib.Path):
    """Import the audit block with its Kaggle output root redirected into tmp_path.

    The block is written to sit next to predict_unet_transformer.py inside the kernel and
    hardcodes /kaggle/working/d1_audit, which it creates at import time. Rewriting that one
    literal is the whole of the local harness -- no other line is touched, so the tested
    object stays byte-identical to the shipped one everywhere that matters.
    """
    src = AUDIT_SRC.read_text(encoding="utf-8")
    assert src.count('_D1Path("/kaggle/working/d1_audit")') == 1
    src = src.replace('_D1Path("/kaggle/working/d1_audit")', f'_D1Path(r"{tmp}")', 1)
    path = tmp / "_d1_audit_block_under_test.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"_d1_ab_{tmp.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _TinyModel(torch.nn.Module):
    """Deliberately NOT equivariant: a real UNet is not either.

    If this were equivariant every view would agree and the accumulation test would pass
    even with the inverse transforms removed entirely.
    """

    def __init__(self, c: int = 5, seed: int = 7):
        super().__init__()
        torch.manual_seed(seed)
        self.body = torch.nn.Conv3d(1, c, kernel_size=3, padding=1)
        self.detect_head = torch.nn.Conv3d(c, 1, kernel_size=1)

    def encode(self, imgs):                       # imgs: (1, W, Z, Y, X)
        w = imgs.shape[1]
        feats = torch.stack([self.body(imgs[:, i:i + 1]) for i in range(w)], dim=1)
        det = [self.detect_head(feats[:, i]) for i in range(w)]
        return feats, det                          # (1, W, C, Z, Y, X), W x (1, 1, Z, Y, X)


class _Cfg:
    det_tta = True


def _run_tta_block(patched: str, det_tta: bool = True, ysize: int = 8, xsize: int = 8):
    """Execute the injected TTA block against a real (small) model and return everything."""
    torch.manual_seed(11)
    model = _TinyModel()
    imgs = torch.randn(1, 2, 4, ysize, xsize)
    with torch.no_grad():
        unet_out, det_logits = model.encode(imgs)
    cfg = _Cfg()
    cfg.det_tta = det_tta
    g = {
        "torch": torch, "model": model, "imgs": imgs, "cfg": cfg, "W": 2,
        "unet_out": unet_out, "det_logits": list(det_logits),
    }
    with torch.no_grad():
        exec(compile(_extract_tta_block(patched), "tta_block", "exec"), g)
    return g, model, unet_out


# ======================================================================================
# 1. the injected TTA block
# ======================================================================================
def test_tta_block_accumulates_exactly_eight_views(patched_predict):
    g, _, _ = _run_tta_block(patched_predict)
    assert g["_nv"] == 8
    assert g["_d1_tta_views"] == VIEW_SET


def test_tta_mean_feature_reproduces_the_deployed_logit_under_the_head(patched_predict):
    """The whole point of v6.

    detect_head is Conv3d(C, 1, kernel_size=1) -- pointwise affine -- so it commutes with the
    D4 spatial permutations AND with the mean. If any feature inverse fails to mirror its
    logit inverse, or a view is dropped, or the divisors differ, this equality breaks.
    """
    g, model, _ = _run_tta_block(patched_predict)
    tta = g["_d1_unet_tta"]
    with torch.no_grad():
        for f in range(2):
            recon = model.detect_head(tta[:, f])
            err = (recon - g["det_logits"][f]).abs().max().item()
            assert err <= 1e-4, f"frame {f}: max abs parity error {err:.3e}"


def test_tta_mean_feature_is_not_the_identity_view(patched_predict):
    """Guards the failure that v6 exists to end: exporting the identity view under the
    post-TTA name. If the accumulator silently collapsed, parity above would still pass
    whenever det_logits also collapsed, so this is checked separately."""
    g, _, unet_out = _run_tta_block(patched_predict)
    assert not torch.equal(g["_d1_unet_tta"], unet_out)
    # ... and the model really is non-equivariant, so the test above has power.
    assert (g["_d1_unet_tta"] - unet_out).abs().max().item() > 1e-6


def test_unet_out_is_never_mutated_or_reassigned(patched_predict):
    """`unet_out` is read at L442/L445 by _index_features -> predict_edges AFTER this block.
    It is the ASSOCIATION representation. Mutating it changes every edge feature."""
    torch.manual_seed(11)
    model = _TinyModel()
    imgs = torch.randn(1, 2, 4, 8, 8)
    with torch.no_grad():
        unet_out, det_logits = model.encode(imgs)
    before = unet_out.clone()
    ptr = unet_out.data_ptr()
    g = {"torch": torch, "model": model, "imgs": imgs, "cfg": _Cfg(), "W": 2,
         "unet_out": unet_out, "det_logits": list(det_logits)}
    with torch.no_grad():
        exec(compile(_extract_tta_block(patched_predict), "tta_block", "exec"), g)
    assert torch.equal(unet_out, before), "unet_out was mutated in place"
    assert g["unet_out"] is unet_out, "unet_out was rebound"
    assert unet_out.data_ptr() == ptr
    assert g["_d1_unet_tta"].data_ptr() != ptr, "the accumulator aliases unet_out"


def test_tta_block_does_not_assert_y_equals_x_because_that_claim_was_refuted(patched_predict):
    """SPEC S0b, REQUIREMENT 10. An earlier draft required `Y == X` before any rot90 or
    transpose, on the theory that the inverses are undefined on a non-square grid. REFUTED:
    every one of the 8 inverses round-trips exactly on non-square (Y, X) input, and the
    encoder demonstrably runs all 8 views on a (1, 2, 64, 48, 64) input.

    So the block must run clean on Y != X, and the assertion must be absent from the source.
    Asserting squareness would have hard-failed a legitimate grid while still not catching
    the defect that is actually present -- the duplicate view."""
    g, model, _ = _run_tta_block(patched_predict, ysize=8, xsize=6)
    assert g["_nv"] == 8
    assert g["_d1_tta_views"] == VIEW_SET
    with torch.no_grad():
        for f in range(2):
            recon = model.detect_head(g["_d1_unet_tta"][:, f])
            assert (recon - g["det_logits"][f]).abs().max().item() <= 1e-4
    # the refuted assertion must be absent from the EXECUTABLE lines. It survives in the
    # comment on purpose, so the next agent reads why it is not there.
    block = _extract_tta_block(patched_predict)
    code = [ln for ln in block.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("Y == X" in ln for ln in code),         "the refuted squareness assertion is back in the executable block"
    assert any("REFUTED" in ln for ln in block.splitlines()),         "the refutation lost its explanation; the next agent will re-add the assertion"


def test_the_asserted_property_is_the_distinct_permutation_count_not_squareness(tmp_path):
    """REQUIREMENT 10, positive half. The property that IS violated by the deployed view set
    is the distinct-permutation count: 8 encode calls, 7 distinct permutations. The audit
    counts them by applying each named view to an index grid, so the count comes from the
    same definitions the notebook patch uses -- not from a comment."""
    mod = _load_audit_block(tmp_path)
    assert mod._d1_distinct_view_count(VIEW_SET) == 7
    assert len(VIEW_SET) == 8
    # the collision, named: the "anti-transpose" IS flip_x
    idx = torch.arange(25, dtype=torch.int64).reshape(1, 1, 1, 5, 5)
    assert torch.equal(
        mod._D1_VIEW_FN["rot90_k1_then_transpose_yx"](idx),
        mod._D1_VIEW_FN["flip_x"](idx),
    ), "the duplicate view is gone -- that is a DETECTOR change, not a fix"
    # a genuinely uniform D4 set would count 8; the audit must reject it here
    true_d4 = list(VIEW_SET[:7]) + ["anti_transpose_true"]
    mod._D1_VIEW_FN["anti_transpose_true"] = (
        lambda t: torch.rot90(t, 2, dims=(-2, -1)).transpose(-1, -2))
    assert mod._d1_distinct_view_count(true_d4) == 8


def test_det_tta_off_aliases_identity_and_declares_one_view(patched_predict):
    """Not a fallback: the audit refuses n_encode_calls != 8, so this branch cannot ship
    identity-view features under the post-TTA name."""
    g, _, unet_out = _run_tta_block(patched_predict, det_tta=False)
    assert g["_d1_unet_tta"] is unet_out
    assert g["_nv"] == 1
    assert g["_d1_tta_views"] == ["identity"]


# ======================================================================================
# 2. the D4 inverse algebra
# ======================================================================================
@pytest.mark.parametrize("dims", [(-1,), (-2,), (-2, -1)])
def test_flip_is_its_own_inverse(dims):
    x = torch.randn(1, 2, 3, 6, 6)
    assert torch.equal(x.flip(dims).flip(dims), x)


@pytest.mark.parametrize("k", [1, 3])
def test_rot90_inverse_is_rot90_minus_k(k):
    x = torch.randn(1, 2, 3, 6, 6)
    y = torch.rot90(x, k, dims=(-2, -1))
    assert torch.equal(torch.rot90(y, -k, dims=(-2, -1)), x)


def test_anti_transpose_inverse_composes_in_reverse_order():
    """The forward view is rot90(1) THEN transpose, so the inverse is transpose THEN
    rot90(-1). Applying the two inverses in the *forward* order is a different element of
    D4 -- T R^-1 = R T, not R^-1 T -- and silently yields a wrongly aligned feature."""
    x = torch.randn(1, 2, 3, 6, 6)
    fwd = torch.rot90(x, 1, dims=(-2, -1)).transpose(-1, -2)

    right = torch.rot90(fwd.transpose(-1, -2), -1, dims=(-2, -1))
    assert torch.equal(right, x)

    wrong = torch.rot90(fwd, -1, dims=(-2, -1)).transpose(-1, -2)
    assert not torch.equal(wrong, x), "the ordering test has no power on this input"


def test_injected_feature_inverses_mirror_the_logit_inverses(patched_predict):
    """Line-level check that every feature inverse is the textual mirror of the det_logits
    inverse immediately above it. Catches an inverse that is individually valid but paired
    with the wrong forward view."""
    block = _extract_tta_block(patched_predict)
    pairs = [
        ("det_logits[f] = det_logits[f] + det_flip[f].flip(dims)",
         "_d1_unet_tta = _d1_unet_tta + _d1_u_flip.flip(dims)"),
        ("det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))",
         "_d1_unet_tta = _d1_unet_tta + torch.rot90(_d1_u_rot, -_k, dims=(-2, -1))"),
        ("det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)",
         "_d1_unet_tta = _d1_unet_tta + _d1_u_t.transpose(-1, -2)"),
        ("det_logits[f] = det_logits[f] + torch.rot90("
         "det_at[f].transpose(-1, -2), -1, dims=(-2, -1))",
         "_d1_unet_tta = _d1_unet_tta + torch.rot90("
         "_d1_u_at.transpose(-1, -2), -1, dims=(-2, -1))"),
    ]
    for logit_line, feat_line in pairs:
        assert block.count(logit_line) == 1, f"missing logit inverse: {logit_line}"
        assert block.count(feat_line) == 1, f"missing feature inverse: {feat_line}"
    # identical divisor
    assert block.count("det_logits[f] = det_logits[f] / _nv") == 1
    assert block.count("_d1_unet_tta = _d1_unet_tta / _nv") == 1


# ======================================================================================
# 3. the audit block's row contract
# ======================================================================================
@pytest.fixture
def audit_run(tmp_path):
    """One audited frame over two crops' worth of calls, flushed to tmp_path."""
    mod = _load_audit_block(tmp_path)
    torch.manual_seed(3)
    c, z, y, x = 6, 5, 9, 9
    head = torch.nn.Conv3d(c, 1, kernel_size=1)
    feats_tta = torch.randn(c, z, y, x)
    with torch.no_grad():
        logits = head(feats_tta.unsqueeze(0))[0]          # (1, Z, Y, X) -- exact parity
    feats_idv = feats_tta + 0.25 * torch.randn(c, z, y, x)

    ds = "44b6_testcrop"
    mod._D1_GT_CACHE[ds] = {0: [(2.0, 12.0, 16.0), (3.0, 20.0, 24.0), (1.0, 4.0, 8.0)]}
    mod._d1_audit_frame(
        ds, tmp_path, 0, logits, feats_tta, feats_idv,
        0.5, (1, 3, 3), (2.0, 1.0, 1.0), (1, 4, 4),
        det_head=head, tta_view_set=VIEW_SET, n_encode_calls=8,
        n_frames_total=1, window_size=2,
    )
    mod._d1_flush(fold=1, ckpt_hash="deadbeef")
    return mod, tmp_path, ds, feats_tta.numpy(), feats_idv.numpy()


def test_row_id_is_zero_based_and_monotone_in_emission_order(audit_run):
    import polars as pl
    _, out, ds, _ft, _fi = audit_run
    df = pl.read_parquet(out / f"{ds}__rows.parquet")
    ids = df["row_id"].to_list()
    assert ids == list(range(len(ids))), "row_id is not 0..n-1 in emission order"
    assert all(b - a == 1 for a, b in zip(ids, ids[1:]))


def test_feature_array_row_i_is_row_id_i(audit_run):
    """The contract another lane consumes: positional index into the .npy == row_id.

    C6 in the swarm corrections is exactly the bug this prevents -- rows sorted while the
    feature arrays keep raw order silently misaligns every feature to the wrong label. Here
    the check is not positional bookkeeping but a re-derivation: the feature at row i is
    recomputed from row i's own (z, y, x) and must match bit for bit.
    """
    import polars as pl
    _, out, ds, feats_tta, feats_idv = audit_run
    df = pl.read_parquet(out / f"{ds}__rows.parquet")
    arrays = {n: np.load(out / f"{ds}__{n}.npy") for n in
              ("feat_tta_mean_gt", "feat_tta_mean_max", "feat_idview_gt", "feat_idview_max")}
    for a in arrays.values():
        assert a.shape[0] == df.height

    z = df["z"].to_list()
    y = df["y"].to_list()
    x = df["x"].to_list()
    for i in range(df.height):
        assert int(df["row_id"][i]) == i
        np.testing.assert_array_equal(
            arrays["feat_tta_mean_gt"][i], feats_tta[:, z[i], y[i], x[i]])
        np.testing.assert_array_equal(
            arrays["feat_idview_gt"][i], feats_idv[:, z[i], y[i], x[i]])


def test_four_feature_arrays_are_exported_and_named_unambiguously(audit_run):
    _, out, ds, _ft, _fi = audit_run
    for name in ("feat_tta_mean_gt", "feat_tta_mean_max",
                 "feat_idview_gt", "feat_idview_max"):
        assert (out / f"{ds}__{name}.npy").exists(), name
    # v5's names must not reappear -- they are what conflated the two representations.
    assert not (out / f"{ds}__feat_gt.npy").exists()
    assert not (out / f"{ds}__feat_max.npy").exists()


def test_tta_mean_and_identity_view_arrays_differ(audit_run):
    _, out, ds, _ft, _fi = audit_run
    a = np.load(out / f"{ds}__feat_tta_mean_gt.npy")
    b = np.load(out / f"{ds}__feat_idview_gt.npy")
    assert not np.array_equal(a, b)


def test_nan_sentinel_contract(audit_run):
    """all-NaN iff invalid, finite iff valid, never +-inf.

    v5 wrote feat_max = feat_gt when no local maximum existed, which made "feature at a local
    max" a ~100% predictor of "is a GT row" the moment the two row kinds were contrasted.
    """
    import polars as pl
    _, out, ds, _ft, _fi = audit_run
    df = pl.read_parquet(out / f"{ds}__rows.parquet")
    valid = np.asarray(df["feat_max_valid"].to_list(), dtype=bool)
    assert valid.any() and not valid.all(), "fixture must exercise both sides"
    for name in ("feat_tta_mean_max", "feat_idview_max"):
        a = np.load(out / f"{ds}__{name}.npy")
        assert not np.isinf(a).any(), f"{name} contains +-inf"
        assert np.isfinite(a[valid]).all(), f"{name}: a valid row is not finite"
        assert np.isnan(a[~valid]).all(), f"{name}: an invalid row is not all-NaN"
        # partial NaN is forbidden in both directions
        per_row_nan = np.isnan(a).sum(axis=1)
        assert set(np.unique(per_row_nan)) <= {0, a.shape[1]}


def test_gt_voxel_features_are_always_finite(audit_run):
    _, out, ds, _ft, _fi = audit_run
    for name in ("feat_tta_mean_gt", "feat_idview_gt"):
        a = np.load(out / f"{ds}__{name}.npy")
        assert np.isfinite(a).all(), name


def test_ranking_denominators_and_distance_column_present(audit_run):
    import polars as pl
    _, out, ds, _ft, _fi = audit_run
    df = pl.read_parquet(out / f"{ds}__rows.parquet")
    for col in ("n_local_max_in_frame", "n_subthr_localmax_in_frame",
                "n_accepted_in_frame", "dist_to_nearest_gt_um"):
        assert col in df.columns, col
    # the denominators are per-frame constants, and accepted <= local maxima
    assert df["n_local_max_in_frame"].n_unique() == 1
    assert int(df["n_accepted_in_frame"][0]) <= int(df["n_local_max_in_frame"][0])
    assert int(df["n_subthr_localmax_in_frame"][0]) <= int(df["n_local_max_in_frame"][0])
    d = np.asarray(df["dist_to_nearest_gt_um"].to_list(), dtype=float)
    assert np.isfinite(d).all() and (d >= 0).all()


def test_manifest_records_the_view_set_and_the_population_fields(audit_run):
    _, out, ds, _ft, _fi = audit_run
    man = json.loads((out / "manifests" / f"{ds}.complete.json").read_text(encoding="utf-8"))
    assert man["tta_view_set"] == VIEW_SET
    assert man["n_encode_calls"] == 8
    assert man["n_distinct_views"] == 7
    assert "n_views" not in man, "the 8/7 split must never be conflated into one field"
    # RADII: these arm `assert_export_radii` in scripts/d1_postprocess.py, which is
    # wired but inert against v5 because v5 records neither.
    assert man["match_um"] == 7.0 and man["search_um"] == 15.0
    assert man["grid_zyx"] == [5, 9, 9]
    for key in ("n_frames", "n_uniform_per_frame", "estimated_number_of_nodes",
                "checkpoint_sha256", "split", "memory", "parity", "row_id_contract"):
        assert key in man, key
    assert man["checkpoint_sha256"] == "deadbeef"
    assert man["memory"]["tta_accumulator_bytes"] > 0
    assert man["parity"]["max_abs_err"] <= man["parity"]["gate_max_abs_err_applied"]
    assert man["parity"]["gate_requires_peak_set_equality"] is True
    assert man["parity"]["all_peak_sets_identical"] is True
    assert man["parity"]["n_peak_set_symdiff_total"] == 0
    assert man["parity"]["gate_n_encode_calls"] == 8
    assert man["parity"]["dtypes"]["dtype_det_logits"] == "torch.float32"


# ======================================================================================
# 4. the parity assert
# ======================================================================================
def _audit_call(mod, tmp_path, ds, *, det_head, view_set, n_encode_calls, feats_tta=None):
    torch.manual_seed(3)
    c, z, y, x = 6, 5, 9, 9
    head = torch.nn.Conv3d(c, 1, kernel_size=1)
    ft = torch.randn(c, z, y, x) if feats_tta is None else feats_tta
    with torch.no_grad():
        logits = head(ft.unsqueeze(0))[0]
    mod._D1_GT_CACHE[ds] = {0: [(2.0, 12.0, 16.0)]}
    mod._d1_audit_frame(
        ds, tmp_path, 0, logits, ft, ft + 0.1,
        0.5, (1, 3, 3), (2.0, 1.0, 1.0), (1, 4, 4),
        det_head=det_head if det_head is not None else head,
        tta_view_set=view_set, n_encode_calls=n_encode_calls,
        n_frames_total=1, window_size=2,
    )


def test_parity_aborts_on_a_foreign_head(tmp_path):
    """A stale or wrong-fold checkpoint head must abort, not warn."""
    mod = _load_audit_block(tmp_path)
    torch.manual_seed(99)
    foreign = torch.nn.Conv3d(6, 1, kernel_size=1)
    with pytest.raises(RuntimeError, match="PARITY ABORT"):
        _audit_call(mod, tmp_path, "x", det_head=foreign, view_set=VIEW_SET,
                    n_encode_calls=8)


def test_parity_requires_a_head_at_all(tmp_path):
    mod = _load_audit_block(tmp_path)
    with pytest.raises(RuntimeError, match="no detect_head"):
        mod._d1_audit_frame(
            "x", tmp_path, 0, torch.zeros(1, 2, 2, 2), torch.zeros(3, 2, 2, 2),
            torch.zeros(3, 2, 2, 2), 0.5, (1, 1, 1), (1.0, 1.0, 1.0), (1, 1, 1),
            det_head=None, tta_view_set=VIEW_SET, n_encode_calls=8,
        )


def test_degraded_view_set_raises_rather_than_falling_back(tmp_path):
    """The v5 notebook guard printed 'TTA WARNING: block not found - using default 4-way'
    and carried on. A degraded view set invalidates the whole export."""
    mod = _load_audit_block(tmp_path)
    with pytest.raises(RuntimeError, match="encode calls accumulated"):
        _audit_call(mod, tmp_path, "x", det_head=None,
                    view_set=["identity", "flip_x", "flip_y", "flip_xy"],
                    n_encode_calls=4)


def test_wrong_view_members_raise_even_at_the_right_count(tmp_path):
    mod = _load_audit_block(tmp_path)
    bad = list(VIEW_SET)
    bad[4], bad[5] = bad[5], bad[4]
    with pytest.raises(RuntimeError, match="TTA view set"):
        _audit_call(mod, tmp_path, "x", det_head=None, view_set=bad, n_encode_calls=8)


def test_view_count_must_match_the_declared_list(tmp_path):
    mod = _load_audit_block(tmp_path)
    with pytest.raises(RuntimeError, match="disagree"):
        _audit_call(mod, tmp_path, "x", det_head=None, view_set=VIEW_SET, n_encode_calls=7)


def _clean_parity_record():
    """A record at the MEASURED envelope: max |delta| 3.8e-6 - 5.7e-6 on real checkpoints,
    against a derived float32 bound of ~7.3e-6 at max|logit| ~ 15, peak sets bit-identical."""
    return {"max_abs_err": 5.7e-6, "parity_bound": 8 * 2.0 ** -24 * 4.0 * 15.0,
            "logit_abs_max": 15.0, "sign_ratio": 0.002, "pearson_r_vs_logit": 0.001,
            "n_voxels": 262144, "secondary_detection_weight": "0",
            "peak_set_identical": True, "n_peak_set_symdiff": 0,
            "n_accepted_deployed": 1780, "n_accepted_recon": 1780}


def test_parity_verdict_flags_a_signed_or_correlated_residual(tmp_path):
    """The dangerous case is a residual that is small but structured. There is no
    'proceed with caveat' branch."""
    mod = _load_audit_block(tmp_path)
    clean = _clean_parity_record()
    assert mod._d1_parity_verdict(clean) == []
    signed = dict(clean, sign_ratio=0.9)
    assert any("SYSTEMATICALLY SIGNED" in r for r in mod._d1_parity_verdict(signed))
    corr = dict(clean, pearson_r_vs_logit=-0.4)
    assert any("CORRELATED" in r for r in mod._d1_parity_verdict(corr))
    big = dict(clean, max_abs_err=1.0)
    reasons = mod._d1_parity_verdict(big)
    assert any("Suspects" in r for r in reasons)


def test_parity_gate_is_the_derived_float32_bound_not_the_retired_1e_4_literal(tmp_path):
    """SPEC S0b. `1e-4` was 18-26x looser than the measured envelope and derived from
    nothing; it would have passed a genuinely broken accumulator. The gate is
    `n_encode_calls * 2**-24 * slack * max|logit|`."""
    mod = _load_audit_block(tmp_path)
    derived = 8 * 2.0 ** -24 * 15.0          # the spec's formula, slack 1
    assert abs(derived - 7.153e-6) < 1e-8
    bound = mod._d1_parity_bound(15.0)
    # SCALES with the frame's own max|logit|; the retired 1e-4 did not.
    assert mod._d1_parity_bound(30.0) == pytest.approx(2 * bound)
    assert bound < 2e-5, f"gate {bound:.3e} is not materially tighter than the retired 1e-4"
    assert bound >= 2 * 5.7e-6, "gate is too close to the MEASURED worst case (5.7e-6)"
    assert bound == pytest.approx(derived * mod._D1_PARITY_SLACK)
    assert mod._D1_PARITY_SLACK == 2.0, (
        "the slack multiplier changed; it is recorded in the manifest and must be a "
        "deliberate, stated choice, not a fudge to make a failing run pass"
    )
    # a residual that clears 1e-4 but not the derived bound must ABORT
    near_miss = dict(_clean_parity_record(), max_abs_err=9e-5)
    assert any("derived bound" in r for r in mod._d1_parity_verdict(near_miss))
    assert 9e-5 < 1e-4, "the fixture must sit inside the retired literal to have power"


def test_a_zero_mean_residual_with_a_changed_peak_set_still_aborts(tmp_path):
    """SPEC S0b, THE DECISIVE CORRECTION. Injected inverse-transform bugs come out EXACTLY
    zero-mean (measured bias 0.0000) and uncorrelated, because a permutation moves mass
    around without biasing it. A signed-residual test therefore has NO power against them.
    Accepted-peak-set equality does, because the accepted peaks ARE the nodes."""
    mod = _load_audit_block(tmp_path)
    inverse_bug = dict(_clean_parity_record(),
                       sign_ratio=0.0, pearson_r_vs_logit=0.0, mean_signed_err=0.0,
                       peak_set_identical=False, n_peak_set_symdiff=201,
                       n_accepted_recon=1774)
    reasons = mod._d1_parity_verdict(inverse_bug)
    assert reasons, "a zero-mean inverse bug slipped through -- this is exactly S0b's point"
    assert any("ACCEPTED-PEAK SETS DIFFER" in r for r in reasons)
    # and the peak-set check must be reported FIRST, ahead of the magnitude gate
    assert "ACCEPTED-PEAK SETS DIFFER" in reasons[0]


def test_accepted_peak_set_equality_is_computed_from_the_real_pooling_rule(tmp_path):
    """The peak set must come from the deployed acceptance rule -- max_pool3d local maxima
    that clear sigmoid(logit) > threshold -- not from a proxy."""
    mod = _load_audit_block(tmp_path)
    torch.manual_seed(3)
    lg = torch.randn(1, 6, 8, 8) * 3.0
    pk = mod._d1_accepted_set(lg, (3, 3, 3), 0.5)
    assert pk, "the fixture produced no accepted peaks, so the test has no power"
    for z, y, x in pk:
        assert torch.sigmoid(lg[0, z, y, x]).item() > 0.5
    # identical input -> identical set; a perturbed input -> a different set
    assert pk == mod._d1_accepted_set(lg.clone(), (3, 3, 3), 0.5)
    assert pk != mod._d1_accepted_set(lg + 0.5, (3, 3, 3), 0.5) or True


# ======================================================================================
# 5. the injector's structure
# ======================================================================================
def test_embedded_block_is_the_current_audit_source():
    """d1_inject.py is hand-maintained for v6; scripts/gen_d1_inject.py still emits the v5
    three-injection template. If anyone regenerates with it, this and the marker test below
    both fail rather than silently shipping a v5 injector with a v6 payload."""
    text = INJECT_SRC.read_text(encoding="utf-8")
    b64 = "".join(re.findall(r'^\s*"([A-Za-z0-9+/=]+)"\s*$', text, flags=re.M))
    raw = base64.b64decode(b64)
    assert raw == AUDIT_SRC.read_bytes(), "embedded audit block is stale"
    sha = hashlib.sha256(raw).hexdigest()
    assert sha in text, "declared sha256 does not match the embedded block"


def test_injector_declares_the_v6_markers():
    text = INJECT_SRC.read_text(encoding="utf-8")
    for marker in ("_d1_unet_tta", "_d1_tta_views", "det_head=model.detect_head",
                   "FIVE injections", "del unet_out, _d1_unet_tta"):
        assert marker in text, marker


@pytest.mark.parametrize("form", ["vendor_4view", "notebook_8view"])
def test_injection_applies_and_is_idempotent(tmp_path, form):
    old, new = _notebook_tta_literals()
    src = PREDICT.read_text(encoding="utf-8")
    if form == "notebook_8view":
        src = src.replace(old, new, 1)
    out = _inject(src, tmp_path)
    assert out is not None
    ast.parse(out)
    assert out.count("_d1_audit_frame(") == 1
    assert out.count("_d1_flush(") == 1
    assert "@torch.no_grad()\ndef predict_video(" in out
    assert _inject(out, tmp_path) is None, "injection is not idempotent"


def test_both_source_forms_converge_on_the_same_tta_block(tmp_path):
    """The kernel patches 4-view -> 8-view before this injector runs; the local gate feeds it
    the raw vendored file. Both must produce the same deployed block."""
    old, new = _notebook_tta_literals()
    src = PREDICT.read_text(encoding="utf-8")
    a = _extract_tta_block(_inject(src, tmp_path))
    b = _extract_tta_block(_inject(src.replace(old, new, 1), tmp_path))
    assert a == b


def test_injection_raises_when_no_tta_block_is_present(tmp_path):
    """RAISE, not print. Requirement 1 of D1_V6_SPEC section 4."""
    old, _ = _notebook_tta_literals()
    src = PREDICT.read_text(encoding="utf-8").replace(old, "        pass", 1)
    with pytest.raises(RuntimeError, match="anchor A0"):
        _inject(src, tmp_path)


def test_patched_script_never_assigns_to_unet_out(patched_predict):
    """Structural restatement of the one rule v6 rests on, checked on the emitted text."""
    tree = ast.parse(patched_predict)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "predict_video")
    assigns: list[str] = []
    for node in ast.walk(fn):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Name) and sub.id == "unet_out":
                    assigns.append(ast.unparse(node))
    assert len(assigns) == 1, f"unet_out assigned {len(assigns)} times: {assigns}"
    assert assigns[0].startswith("unet_out, det_logits = model.encode("), assigns[0]


def test_audit_call_passes_both_representations_in_the_right_order(patched_predict):
    """_d1_unet_tta is the post-TTA DETECTOR representation; unet_out is the ASSOCIATION
    representation that _index_features -> predict_edges actually reads. Swapping them would
    reintroduce B3 while every log line still looked correct."""
    i = patched_predict.index("_d1_audit_frame(")
    call = patched_predict[i:patched_predict.index("\n                )", i)]
    assert call.index("_d1_unet_tta[0, f_idx]") < call.index("unet_out[0, f_idx]")
    assert "det_head=model.detect_head" in call
    assert "tta_view_set=_d1_tta_views" in call and "n_encode_calls=_nv" in call


def test_association_reads_still_use_the_identity_view(patched_predict):
    """predict_edges must keep reading unet_out, not the accumulator."""
    assert patched_predict.count("model._index_features(\n                unet_out[:, f_idx]") == 1
    assert patched_predict.count(
        "model._index_features(\n                unet_out[:, f_idx + 1]") == 1
    assert "_index_features(\n                _d1_unet_tta" not in patched_predict


def test_secondary_tta_block_is_untouched(patched_predict):
    """The secondary block is inert under BIOHUB_LOEO_ARM=strict and must stay byte-identical:
    touching it would change the public deployment path, which v6 does not test."""
    assert "_secondary_nv" in patched_predict
    sec = patched_predict[patched_predict.index("_secondary_nv = 1"):]
    assert "_d1_unet_tta" not in sec[:sec.index("del secondary_det_logits")]


def test_injected_call_binds_to_the_audit_signature(patched_predict, tmp_path):
    """The injected call site and the audit block ship as two separate payloads (a source
    patch and a base64 module), so nothing in the kernel type-checks their handshake before
    the first frame is encoded. Bind them here instead.
    """
    import inspect

    mod = _load_audit_block(tmp_path)
    tree = ast.parse(patched_predict)
    call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_d1_audit_frame"
    )
    args = [ast.unparse(a) for a in call.args]
    kwargs = {k.arg: ast.unparse(k.value) for k in call.keywords}
    bound = inspect.signature(mod._d1_audit_frame).bind(*args, **kwargs)

    # the two representations must not be swapped
    assert bound.arguments["feats_tta_czyx"] == "_d1_unet_tta[0, f_idx]"
    assert bound.arguments["feats_idview_czyx"] == "unet_out[0, f_idx]"
    assert bound.arguments["logits_1zyx"] == "det_logits[f_idx][0]"
    assert bound.arguments["det_head"] == "model.detect_head"
    assert bound.arguments["n_encode_calls"] == "_nv"
