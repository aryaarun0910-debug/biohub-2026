"""Locks for the D1 CPU postprocessor and for the TTA feature/logit defect it exposed.

Two independent things are locked here:

  1. the M/C/T/L/D derivation rules, which are now CPU-side because the kernel exports no
     `d1_class` and no `matched` column;
  2. the fact that v5 features are IDENTITY-VIEW while v5 logits are POST-TTA. That
     mismatch is why H0 parity is mathematically unavailable from a v5 export, and it must
     not be re-broken silently.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import d1_postprocess as d1pp  # noqa: E402
from d1_postprocess import (  # noqa: E402
    ASSOCIATION,
    DERIVED_SCHEMA_VERSION,
    DETECTOR,
    FEAT_MAX_VALID,
    FEATURE_ARRAYS_BY_SCHEMA,
    MATCH_UM,
    REPRESENTATIONS,
    REQUIRED_ROW_COLUMNS,
    ROW_ID,
    SCHEMA_DECLARATION,
    SCHEMA_V5,
    SCHEMA_V6,
    SEARCH_UM,
    apply_validity,
    assert_export_radii,
    assert_radius_binding,
    assert_representations_differ,
    atomic_write,
    attach_row_id,
    check_feature_contract,
    classify,
    feat_max_validity,
    gt_arrays,
    guard_out_dir,
    load_features,
    matched_gt_ids,
    max_arrays,
    representation_arrays,
    resolve_schema,
    save_npy,
    sort_rows_with_features,
)

PREDICT = ROOT / "vendor" / "kaggle-cell-tracking" / "scripts" / "predict_unet_transformer.py"
AUDIT = ROOT / "scripts" / "kaggle_edits" / "d1_response_audit.py"


# --------------------------------------------------------------- partition rules
def test_matched_is_M_regardless_of_neighbourhood():
    """M comes from the scorer alone; no neighbourhood statistic may override it."""
    for n7, a7, n15 in ((0, 0, 0), (5, 5, 9), (0, 0, 3)):
        assert classify(True, n7, a7, n15) == "M"


def test_accepted_peak_within_match_radius_is_C():
    assert classify(False, 3, 1, 4) == "C"


def test_unaccepted_local_max_within_match_radius_is_T():
    assert classify(False, 3, 0, 4) == "T"


def test_max_only_in_outer_shell_is_L():
    assert classify(False, 0, 0, 2) == "L"


def test_nothing_within_search_radius_is_D():
    assert classify(False, 0, 0, 0) == "D"


def test_C_takes_precedence_over_T():
    """A GT with both accepted and unaccepted maxima inside MATCH_UM is competition, not
    threshold: an accepted peak was there and still failed to win the assignment."""
    assert classify(False, 5, 2, 7) == "C"


def test_partition_is_total_over_the_reachable_input_space():
    """Every (n_lm_7, n_acc_7, n_lm_15) combination lands in exactly one class."""
    seen = set()
    for n15 in range(4):
        for n7 in range(n15 + 1):           # n_lm_15um is a superset of n_lm_7um
            for a7 in range(n7 + 1):        # accepted are a subset of local maxima
                cls = classify(False, n7, a7, n15)
                assert cls in {"C", "T", "L", "D"}
                seen.add(cls)
    assert seen == {"C", "T", "L", "D"}, seen


def test_required_columns_cover_every_input_the_rules_read():
    for col in ("n_lm_7um", "n_acc_7um", "n_lm_15um"):
        assert col in REQUIRED_ROW_COLUMNS


def test_radii_match_the_scorer_and_the_design():
    assert MATCH_UM == 7.0
    assert SEARCH_UM == 15.0


def test_derived_schema_version_is_pinned():
    """Changing the derivation MUST change this string, or old artifacts silently mix.

    BUMPED FOR THE v6 READER (repair 7). d1-derived-2 named its arrays `feat_gt`/`feat_max`
    whatever they held, so a d1-derived-2 artifact cannot say WHICH representation it is.
    d1-derived-3 names them after the source array and records the export schema and the
    representation in both the parquet and the report.
    """
    assert DERIVED_SCHEMA_VERSION == "d1-derived-3"


# --------------------------------------------------------------- the TTA defect
NOTEBOOKS = [
    ROOT / "notebooks" / "kaggle_p3_d1_smoke_f0" / "biohub-p3-d1-smoke-f0.ipynb",
    ROOT / "notebooks" / "kaggle_p3_d1_smoke_f1" / "biohub-p3-d1-smoke-f1.ipynb",
]


def _nb_source(p: Path) -> str:
    import json
    return "".join("".join(c["source"]) for c in json.loads(p.read_text(encoding="utf-8"))["cells"])


@pytest.mark.skipif(not PREDICT.exists(), reason="vendored predict script not present")
def test_vendored_four_view_block_is_the_patch_target_not_the_deployed_code():
    """The 4-view block in vendor/ is REPLACED at build time. Auditing it audits a file
    that never runs -- that mistake was made once and must not recur."""
    src = PREDICT.read_text(encoding="utf-8")
    assert "tta_flips = [(-1,), (-2,), (-2, -1)]" in src
    assert "det_logits[f] = det_logits[f] / 4" in src, (
        "the vendored 4-view block changed; the notebook's _old anchor will no longer match "
        "and the TTA patch will silently fall back to 4 views"
    )


@pytest.mark.skipif(not all(p.exists() for p in NOTEBOOKS), reason="built notebooks absent")
def test_deployed_tta_makes_eight_encode_calls():
    """The deployed path makes 8 encode calls, not the vendored 4, divided by `_nv`.

    The literal `_nv += 1` appears 4 times because three views are produced inside loops,
    so counting the literal is wrong; this asserts the call SET.
    See `test_deployed_tta_has_only_seven_distinct_views` for what those 8 calls actually
    compute -- it is NOT 8 distinct views.
    """
    for nb in NOTEBOOKS:
        src = _nb_source(nb)
        assert "eight-view planar detection TTA" in src
        blk = src[src.find("_new = "):src.find("if _old in _s:")]
        assert blk, "TTA replacement block not found"
        assert "_nv = 1" in blk, "identity view no longer seeds the counter"
        assert "for dims in [(-1,), (-2,), (-2, -1)]:" in blk      # +3
        assert "for _k in (1, 3):" in blk                          # +2
        assert "imgs_t = imgs.transpose(-1, -2)" in blk            # +1
        assert "imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)" in blk  # +1
        assert blk.count("model.encode") == 4, "encode call count changed"
        assert blk.count("_nv += 1") == 4, "increment sites changed"
        assert "det_logits[f] = det_logits[f] / _nv" in blk, "denominator is not the counter"


def test_deployed_tta_has_only_seven_distinct_views():
    """THE COLLISION, AND IT IS DELIBERATE THAT WE DO NOT FIX IT.

    The view written as the anti-transpose

        torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)

    is EXACTLY `imgs.flip(-1)`, which is already view 1. Every single-order composition of
    rot90 at odd k with transpose collapses to a flip; a true anti-transpose needs k=2.

    So the deployed detector makes 8 encode calls over 7 distinct views: `flip(-1)` carries
    weight 2/8 and the true anti-transpose carries 0/8, while the divisor is still 8. The
    measure is therefore NON-UNIFORM on D4, and a 7-element subset of an order-8 group is
    never a subgroup -- the deployed average is not a group average.

    v6 MUST REPLICATE THIS VERBATIM. v6 exists to describe the DEPLOYED detector. Correcting
    the view set is a detector change: it moves the node population, forces a predict_edges
    rerun, and voids the 0.889 anchor. Legitimate as a standalone experiment; never as a
    silent ride-along.
    """
    import torch

    torch.manual_seed(0)
    x = torch.randn(1, 2, 3, 5, 7, 7)          # (B, W, C, Z, Y, X)

    as_coded = torch.rot90(x, 1, dims=(-2, -1)).transpose(-1, -2)
    assert torch.equal(as_coded, x.flip(-1)), (
        "the anti-transpose no longer collides with flip(-1) -- the deployed view set "
        "changed, which is a DETECTOR change and voids the current anchors"
    )

    views = [
        x,
        x.flip(-1),
        x.flip(-2),
        x.flip((-2, -1)),
        torch.rot90(x, 1, dims=(-2, -1)),
        torch.rot90(x, 3, dims=(-2, -1)),
        x.transpose(-1, -2),
        as_coded,
    ]
    assert len(views) == 8, "encode-call count"
    assert len({v.numpy().tobytes() for v in views}) == 7, "distinct-view count"

    # a true anti-transpose needs k=2 and is absent from the deployed set
    true_anti = torch.rot90(x, 2, dims=(-2, -1)).transpose(-1, -2)
    assert all(not torch.equal(true_anti, v) for v in views)


def test_every_deployed_view_inverse_is_a_true_inverse_including_non_square():
    """The output is mis-WEIGHTED, not corrupted: all 8 inverses round-trip exactly, and
    they do so on non-square input too -- so `Y == X` is NOT the safety property here."""
    import torch

    torch.manual_seed(1)
    for shape in ((1, 2, 3, 5, 7, 7), (1, 2, 3, 5, 6, 8)):   # square and non-square Y/X
        x = torch.randn(*shape)
        pairs = [
            (lambda t: t, lambda t: t),
            (lambda t: t.flip(-1), lambda t: t.flip(-1)),
            (lambda t: t.flip(-2), lambda t: t.flip(-2)),
            (lambda t: t.flip((-2, -1)), lambda t: t.flip((-2, -1))),
            (lambda t: torch.rot90(t, 1, dims=(-2, -1)),
             lambda t: torch.rot90(t, -1, dims=(-2, -1))),
            (lambda t: torch.rot90(t, 3, dims=(-2, -1)),
             lambda t: torch.rot90(t, -3, dims=(-2, -1))),
            (lambda t: t.transpose(-1, -2), lambda t: t.transpose(-1, -2)),
            # composition -- the inverse applies in REVERSE order
            (lambda t: torch.rot90(t, 1, dims=(-2, -1)).transpose(-1, -2),
             lambda t: torch.rot90(t.transpose(-1, -2), -1, dims=(-2, -1))),
        ]
        for i, (fwd, inv) in enumerate(pairs):
            assert torch.equal(inv(fwd(x)), x), f"view {i} inverse failed on {shape}"


@pytest.mark.skipif(not all(p.exists() for p in NOTEBOOKS), reason="built notebooks absent")
def test_a_second_independent_tta_block_exists_for_the_secondary_model():
    """There are TWO TTA blocks -- primary and secondary detection models. Anything
    reasoning about 'the' deployed detection field must account for both.

    FLIPPED FOR v6. The literal `_nv = 1` count is no longer 2: the v6 injector adds anchor
    strings that also contain it, so the count is now 5. Counting a literal was always the
    weak form of this assertion -- it broke on a change that did not touch either block. The
    v6 lock asserts the SECONDARY block is still structurally present and independent, which
    is the property that actually matters. `tests/test_d1_v6_export.py` separately asserts the
    secondary block is byte-untouched.
    """
    src = _nb_source(NOTEBOOKS[0])
    assert src.count("_nv = 1") >= 2, "the secondary-model TTA block moved or merged"
    # both blocks still divide by their own runtime counter, not a literal
    assert src.count("det_logits[f] = det_logits[f] / _nv") >= 2, (
        "a TTA block no longer divides by its own counter"
    )
    assert "BIOHUB_SECONDARY_DETECTION_WEIGHT" in src, (
        "the secondary detection path vanished from the built notebook"
    )


@pytest.mark.skipif(not all(p.exists() for p in NOTEBOOKS), reason="built notebooks absent")
def test_tta_patch_failure_is_caught_downstream_not_silently_tolerated():
    """CORRECTED. I first recorded the TTA patch guard as a silent degradation because it
    ends in a `print`, not a `raise`. That was wrong, and the mechanism matters.

    The guard itself does only print. But the NEXT patch anchors on text containing `_nv`:

        'for f in range(W):\\n    det_logits[f] = det_logits[f] / _nv\\n\\n    del imgs'

    If the TTA patch had failed, the file would still read `/ 4`, that anchor would not
    match, and the patch machinery raises on a match-count mismatch. So a failed TTA patch
    hard-fails one step later rather than shipping a 4-view detector.

    Converting the guard to a `raise` is still worth doing for locality -- the failure
    should name its own cause -- but it is a readability fix, not a correctness hole.

    FLIPPED FOR v6. v6 requirement 4 converted the guard to a hard failure, so the locality
    fix has landed: the notebook now carries `TTA PATCH FAILED` and no `TTA WARNING`. The
    downstream `_nv` anchor still exists as defence in depth.
    """
    src = _nb_source(NOTEBOOKS[0])
    assert 'print("TTA WARNING: block not found - using default 4-way")' not in src, (
        "the silent-degradation guard came back; v6 requires a hard failure"
    )
    assert "TTA PATCH FAILED" in src, "the TTA patch no longer fails loudly"
    assert "det_logits[f] = det_logits[f] / _nv" in src, (
        "the downstream anchor no longer references _nv -- restore the defence in depth"
    )
    # The 8/7 split must be recorded as TWO fields; a single conflated `n_views` is the bug.
    assert "n_views" not in src, (
        "a conflated n_views reappeared -- record n_encode_calls=8 and n_distinct_views=7"
    )


@pytest.mark.skipif(not PREDICT.exists(), reason="vendored predict script not present")
def test_tta_loop_discards_the_flipped_feature_maps():
    """THE DEFECT. `unet_out` is bound once from the identity view and never accumulated,
    so `detect_head(unet_out)` is the identity-view logit, NOT the deployed post-TTA logit.

    Any v6 export claiming post-TTA features must inverse-transform and accumulate feature
    maps over the SAME view list. When that lands, this test must be updated deliberately.
    """
    src = PREDICT.read_text(encoding="utf-8")
    assert "unet_out, det_logits = model.encode(imgs)" in src
    assert "_, det_flip = model.encode(imgs_flip)" in src, (
        "flipped features are no longer discarded -- if v6 landed, update this lock"
    )
    tta = src.split("if cfg.det_tta:", 1)[1].split("del imgs", 1)[0]
    assert "unet_out" not in tta, "unet_out is now touched inside the TTA loop"


# ------------------------------------------- unet_out is the ASSOCIATION representation
@pytest.mark.skipif(not PREDICT.exists(), reason="vendored predict script not present")
def test_unet_out_feeds_predict_edges_so_v6_must_not_reassign_it():
    """THE v6 TRAP. Every research lane proposed accumulating the TTA mean INTO `unet_out`.
    But `unet_out` is read AFTER the TTA block by `_index_features` -> `predict_edges`, so
    reassigning it would silently change every edge feature and alter the 0.915 association
    substrate -- the largest downside risk in the roadmap, introduced by the fix for B3.

    v6 must accumulate into a SEPARATE tensor and leave `unet_out` alone.
    """
    src = PREDICT.read_text(encoding="utf-8")
    assert "unet_out, det_logits = model.encode(imgs)" in src

    # the association reads, and they are downstream of the TTA block
    bind = src.index("unet_out, det_logits = model.encode(imgs)")
    tta = src.index("if cfg.det_tta:")
    assoc = [i for i in range(len(src)) if src.startswith("unet_out[:, f_idx", i)]
    assert len(assoc) == 2, f"expected 2 association reads of unet_out, found {len(assoc)}"
    assert bind < tta < min(assoc), "TTA block is no longer between the bind and the reads"

    # inside the TTA block, unet_out must never appear on a left-hand side
    block = src[tta:src.index("del imgs", tta)]
    assert "unet_out" not in block, (
        "unet_out is touched inside the TTA loop -- if this is the v6 accumulator it MUST "
        "be a separate tensor, or association is silently changed"
    )


# --------------------------------------------------------------- trap 21: schema drop
def test_polars_silently_drops_columns_absent_from_the_inference_window():
    """The defect itself, reproduced. 96 non-GT rows are emitted per frame, so a crop whose
    first GT lands at frame >= 2 puts the first gt row at index >= 192 -- past the default
    100-row inference window -- and EVERY gt-only column vanishes with no error."""
    import polars as pl

    gt_only = ["n_lm_7um", "n_acc_7um", "n_lm_15um", "best7_dist_um", "gt_z", "gt_y", "gt_x"]

    def build(lead_non_gt: int, **kw):
        rows = [{"dataset": "d", "t": 0, "kind": "uniform", "logit": 0.1}
                for _ in range(lead_non_gt)]
        rows.append({"dataset": "d", "t": 0, "kind": "gt_centre", "logit": 0.1,
                     **{c: 1.0 for c in gt_only}})
        return pl.DataFrame(rows, **kw)

    assert all(c in build(99).columns for c in gt_only), "fixture no longer exercises the trap"
    assert not any(c in build(100).columns for c in gt_only), (
        "polars no longer drops silently -- re-check whether the guard is still needed"
    )
    # the fix
    assert all(c in build(100, infer_schema_length=None).columns for c in gt_only)


def test_emission_arithmetic_puts_the_first_gt_row_past_the_window_from_frame_two():
    """96 = 64 uniform + 32 subthr per frame; 96*2 = 192 > 100."""
    from importlib import util
    spec = util.spec_from_file_location("_d1ra", AUDIT)
    src = AUDIT.read_text(encoding="utf-8")
    assert 'BIOHUB_D1_N_UNIFORM", "64"' in src
    assert 'BIOHUB_D1_N_SUBTHR", "32"' in src
    per_frame_non_gt = 64 + 32
    assert per_frame_non_gt * 1 < 100 <= per_frame_non_gt * 2
    assert spec is not None


@pytest.mark.skipif(not AUDIT.exists(), reason="audit edit not present")
def test_exporter_declares_schema_and_asserts_the_column_contract():
    """The fix must stay in place: full-scan inference plus a hard column-contract check."""
    src = AUDIT.read_text(encoding="utf-8")
    assert "_pl.DataFrame(_rows, infer_schema_length=None)" in src, (
        "schema inference is bounded again -- trap 21 is live"
    )
    assert "_D1_REQUIRED_COLS" in src
    assert "D1 column contract violated" in src, "silent drop is no longer fatal"
    for col in ("n_lm_7um", "n_acc_7um", "n_lm_15um", "gt_z"):
        assert f'"{col}"' in src


@pytest.mark.skipif(not AUDIT.exists(), reason="audit edit not present")
def test_audit_passes_identity_view_features_alongside_post_tta_logits():
    """FLIPPED FOR v6. This lock recorded the v5 defect -- post-TTA logits paired with
    IDENTITY-VIEW features -- so that fixing it had to be deliberate. v6 fixed it, so the
    lock now asserts the v6 contract instead: BOTH representations are exported under names
    that cannot be confused, and the identity-view pair is never called post-TTA.

    The identity-view arrays are retained on purpose: `predict_edges` genuinely reads the
    identity view, so they are the ASSOCIATION representation. Conflating them with the
    detector representation is exactly what made v5 D1-F invalid.
    """
    src = AUDIT.read_text(encoding="utf-8")
    for name in ("__feat_tta_mean_gt.npy", "__feat_tta_mean_max.npy",
                 "__feat_idview_gt.npy", "__feat_idview_max.npy"):
        assert name in src, f"v6 must export {name}"
    # the detector feature must come from a SEPARATE accumulator, never from unet_out
    assert "_d1_unet_tta" in src, "the TTA-mean accumulator is gone"
    assert "unet_out" in src, "the identity-view (association) feature is gone"


# =============================================================================================
# THE SIX C6 REPAIRS. One block per defect; every test below fails against the pre-repair
# postprocessor (`d1-derived-1`), which had none of these functions or guarantees.
# =============================================================================================
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402


def _rows(n_frames=3, n_gt=2, n_non_gt=4):
    """Emission order as the kernel actually writes it: each frame emits its gt_centre rows
    FIRST, then the non-GT rows. So GT rows are a NON-CONTIGUOUS subset of the feature
    array -- the mechanism behind defect 1."""
    out = []
    for t in range(n_frames):
        for g in range(n_gt):
            out.append({"t": t, "kind": "gt_centre",
                        # descending in y so a sort genuinely permutes the rows
                        "gt_z": 0.0, "gt_y": float(n_gt - g), "gt_x": 0.0,
                        "n_lm_15um": 1})
        for _ in range(n_non_gt):
            out.append({"t": t, "kind": "uniform",
                        "gt_z": None, "gt_y": None, "gt_x": None, "n_lm_15um": None})
    return pl.DataFrame(out)


def _feats(n, dim=3):
    """Feature row i is the constant vector i, so a misaligned gather is self-evident."""
    return np.repeat(np.arange(n, dtype=np.float32)[:, None], dim, axis=1)


# ------------------------------------------------------------------ defect 1: misalignment
def test_gt_rows_are_a_non_contiguous_subset_so_a_positional_join_misaligns():
    """THE DEFECT, reproduced in miniature. Filtering to gt_centre and then indexing the
    feature array by POSITION reads the wrong feature for all but the first row."""
    df = _rows().with_row_index(ROW_ID).with_columns(pl.col(ROW_ID).cast(pl.Int64))
    feats = _feats(df.height)
    gt = df.filter(pl.col("kind") == "gt_centre")

    # 2 GT rows then 4 non-GT rows per frame => GT feature rows are 0,1, 6,7, 12,13
    assert gt.get_column(ROW_ID).to_list() == [0, 1, 6, 7, 12, 13]

    naive = feats[: gt.height]                          # the bug: positional join
    correct = feats[gt.get_column(ROW_ID).to_numpy()]   # the fix: join by row_id
    assert not np.array_equal(naive, correct)
    # only the FIRST frame's GT block lands by luck; every later frame is offset
    wrong = int((naive[:, 0] != correct[:, 0]).sum())
    assert wrong == 4, wrong


def test_sort_rows_with_features_applies_the_identical_permutation():
    """C6: never sort rows without applying the identical permutation to the features."""
    df, _ = attach_row_id(_rows(), "c", allow_v5_emission_order=True)
    feats = _feats(df.height)
    gt = df.filter(pl.col("kind") == "gt_centre")

    out, gathered = sort_rows_with_features(gt, {"feat_gt": feats})
    # the sort really did reorder the rows
    assert out.get_column(ROW_ID).to_list() != gt.get_column(ROW_ID).to_list()
    # and every feature row still belongs to its own label row
    assert np.array_equal(gathered["feat_gt"][:, 0],
                          out.get_column(ROW_ID).to_numpy().astype(np.float32))
    assert out.height == gt.height == gathered["feat_gt"].shape[0]


def test_sort_is_total_and_reproducible_when_the_spatial_keys_tie():
    """row_id is the final sort key, so tied coordinates cannot reorder run to run."""
    tied = pl.DataFrame(
        [{"t": 0, "kind": "gt_centre", "gt_z": 0.0, "gt_y": 0.0, "gt_x": 0.0}] * 5)
    df, _ = attach_row_id(tied, "c", allow_v5_emission_order=True)
    out, gathered = sort_rows_with_features(df, {"f": _feats(df.height)})
    assert out.get_column(ROW_ID).to_list() == [0, 1, 2, 3, 4]
    assert np.array_equal(gathered["f"][:, 0], np.arange(5, dtype=np.float32))


def test_reordering_without_row_id_is_refused_outright():
    with pytest.raises(SystemExit, match=ROW_ID):
        sort_rows_with_features(_rows(), {"f": _feats(18)})


# --------------------------------------------------------------------- defect 2: row_id
def test_a_v5_export_without_row_id_is_refused_rather_than_guessed():
    """v5 predates the contract. Guessing emission order is exactly the silent-misalignment
    failure this repair exists to prevent, so it must be declared, not assumed."""
    with pytest.raises(SystemExit, match="row_id"):
        attach_row_id(_rows(), "c", allow_v5_emission_order=False)


def test_the_v5_assumption_can_be_declared_explicitly_and_is_recorded():
    df, source = attach_row_id(_rows(), "c", allow_v5_emission_order=True)
    assert source == "v5-emission-order-declared"
    assert df.get_column(ROW_ID).to_list() == list(range(df.height))


def test_an_exported_row_id_is_consumed_not_regenerated():
    raw = _rows().with_columns(pl.Series(ROW_ID, list(range(18)), dtype=pl.Int64))
    df, source = attach_row_id(raw, "c", allow_v5_emission_order=False)
    assert source == "export"
    assert df.get_column(ROW_ID).to_list() == list(range(18))


def test_a_reordered_export_is_still_exact_because_features_are_gathered_by_row_id():
    """row_id -- not parquet position -- is the join key. A shuffled parquet is recoverable
    and must NOT be rejected, but the fact is recorded rather than assumed."""
    raw = _rows().with_columns(pl.Series(ROW_ID, list(range(18)), dtype=pl.Int64))
    shuffled = raw.sample(fraction=1.0, shuffle=True, seed=7)
    df, source = attach_row_id(shuffled, "c", allow_v5_emission_order=False)
    assert source == "export-reordered"
    out, gathered = sort_rows_with_features(
        df.filter(pl.col("kind") == "gt_centre"), {"f": _feats(18)})
    assert np.array_equal(gathered["f"][:, 0],
                          out.get_column(ROW_ID).to_numpy().astype(np.float32))


@pytest.mark.parametrize("bad", [
    list(range(17)) + [16],          # duplicate
    list(range(1, 19)),              # 1-based
    list(range(17)) + [99],          # out of range
])
def test_a_row_id_that_is_not_a_bijection_is_fatal(bad):
    raw = _rows().with_columns(pl.Series(ROW_ID, bad, dtype=pl.Int64))
    with pytest.raises(SystemExit, match="bijection"):
        attach_row_id(raw, "c", allow_v5_emission_order=False)


def test_a_null_row_id_is_fatal():
    raw = _rows().with_columns(pl.Series(ROW_ID, [None] + list(range(1, 18)), dtype=pl.Int64))
    with pytest.raises(SystemExit, match="null"):
        attach_row_id(raw, "c", allow_v5_emission_order=False)


# ------------------------------------------------------- defect 3: MATCH_UM is not a literal
def test_match_um_is_bound_to_the_scorer_rather_than_written_down():
    """MATCH_UM must BE the scorer's radius, not a literal that happens to agree today."""
    import inspect

    from biotrack.metric import MAX_DISTANCE
    from tracking_cellmot.metrics import evaluate as _ev

    assert MATCH_UM == float(MAX_DISTANCE)
    assert MATCH_UM == float(inspect.signature(_ev).parameters["max_distance"].default)
    src = (ROOT / "scripts" / "d1_postprocess.py").read_text(encoding="utf-8")
    assert "MATCH_UM: float = float(MAX_DISTANCE)" in src, (
        "MATCH_UM is a literal again; a scorer change would silently desynchronise the "
        "partition from the matching that defines it"
    )


def test_a_drifting_component_makes_the_postprocessor_refuse_to_run():
    import biotrack.d1_partition as _d1p

    assert assert_radius_binding()["scorer_max_distance"] == MATCH_UM
    old = _d1p.MATCH_UM
    try:
        _d1p.MATCH_UM = 8.0
        with pytest.raises(SystemExit, match="desynchronised"):
            assert_radius_binding()
    finally:
        _d1p.MATCH_UM = old
    assert assert_radius_binding()["d1_partition_match_um"] == MATCH_UM


def test_the_outer_search_radius_must_exceed_the_match_radius():
    assert SEARCH_UM > MATCH_UM


# ------------------------------------------------------------------ defect 4: zero-edge match
def _graph(coords, edges=()):
    import tracksdata as td

    from biotrack.submission import submission_to_graphs
    if not coords:
        g = td.graph.InMemoryGraph()
        for key in ("z", "y", "x"):
            g.add_node_attr_key(key, pl.Float64, -999999.0)
        return g
    rows = [{"id": i, "dataset": "X", "row_type": "node", "node_id": i + 1,
             "t": t, "z": float(z), "y": float(y), "x": float(x),
             "source_id": -1, "target_id": -1}
            for i, (t, z, y, x) in enumerate(coords)]
    for j, (s, d) in enumerate(edges):
        rows.append({"id": len(rows) + j, "dataset": "X", "row_type": "edge", "node_id": -1,
                     "t": -1, "z": -1.0, "y": -1.0, "x": -1.0,
                     "source_id": s + 1, "target_id": d + 1})
    return submission_to_graphs(pl.DataFrame(rows))["X"]


COORDS = [(0, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 0.0), (2, 0.0, 0.0, 0.0)]
GT_EDGES = [(0, 1), (1, 2)]


def test_the_scorer_never_writes_a_matching_for_a_zero_edge_prediction():
    """THE CAUSE. `_evaluate` warns and returns 0.0 BEFORE `graph.match(...)`, so
    MATCHED_NODE_ID is never written. Reading that as 'nothing matched' is the defect."""
    import tracksdata as td

    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE
    from tracking_cellmot.metrics import evaluate as _ev

    pred, gt = _graph(COORDS), _graph(COORDS, GT_EDGES)
    assert pred.num_nodes() == 3 and pred.num_edges() == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ev(pred, gt, scale=DEFAULT_SCALE, max_distance=MAX_DISTANCE)
    assert td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID not in pred.node_attr_keys(), (
        "the vendored scorer now matches zero-edge graphs; the special case can be removed"
    )


def test_a_zero_edge_prediction_still_matches_every_node_it_should():
    """A prediction can have nodes and no edges, and those nodes still match. The zero SCORE
    is about edges; node matching is a different question."""
    pred, gt = _graph(COORDS), _graph(COORDS, GT_EDGES)
    ids, path = matched_gt_ids(pred, gt)
    assert path == "distance-matching-zero-edge"
    assert len(ids) == gt.num_nodes() == 3, (
        "zero-edge nodes were dropped from the matching; every GT of such a crop would be "
        "misclassified as unmatched"
    )


def test_a_prediction_with_edges_still_uses_the_scorer_itself():
    ids, path = matched_gt_ids(_graph(COORDS, GT_EDGES), _graph(COORDS, GT_EDGES))
    assert path == "evaluate"
    assert len(ids) == 3


def test_a_prediction_with_no_nodes_at_all_matches_nothing():
    ids, path = matched_gt_ids(_graph([]), _graph(COORDS, GT_EDGES))
    assert path == "no-nodes" and ids == set()


def test_far_away_zero_edge_nodes_do_not_match():
    """The zero-edge path must respect the same radius, not match everything blindly."""
    far = [(t, 500.0, 500.0, 500.0) for t, *_ in COORDS]
    ids, path = matched_gt_ids(_graph(far), _graph(COORDS, GT_EDGES))
    assert path == "distance-matching-zero-edge" and ids == set()


# ------------------------------------------------------- defect 5: feat_max_valid semantics
def test_infinities_are_neither_valid_nor_invalid_and_are_rejected():
    arr = np.zeros((3, 4), dtype=np.float32)
    arr[1, 2] = np.inf
    with pytest.raises(SystemExit, match="inf"):
        check_feature_contract("c", "feat_max", arr)


def test_a_row_may_not_mix_nan_with_finite_values():
    arr = np.zeros((3, 4), dtype=np.float32)
    arr[1, 2] = np.nan
    with pytest.raises(SystemExit, match="mix"):
        check_feature_contract("c", "feat_max", arr)


def test_all_nan_iff_invalid_is_enforced_in_both_directions():
    arr = np.zeros((3, 4), dtype=np.float32)
    arr[1] = np.nan
    assert np.array_equal(check_feature_contract("c", "f", arr, [True, False, True]),
                          [True, False, True])
    for wrong in ([True, True, True], [False, False, True]):
        with pytest.raises(SystemExit, match=FEAT_MAX_VALID):
            check_feature_contract("c", "f", arr, wrong)


def test_apply_validity_makes_invalid_rows_all_nan_without_touching_valid_ones():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = apply_validity(arr, [True, False, True])
    assert np.isnan(out[1]).all()
    assert np.array_equal(out[[0, 2]], arr[[0, 2]])
    assert np.array_equal(arr, np.arange(12, dtype=np.float32).reshape(3, 4)), "input mutated"
    assert apply_validity(np.zeros((2, 2), dtype=np.float64), [True, True]).dtype == np.float64


def test_v5_feat_max_validity_is_reconstructed_from_the_export_and_verified():
    """v5 wrote a COPY OF feat_gt where no maximum existed instead of NaN, and shipped no
    validity column. The reconstruction must be checked against the data, not trusted."""
    df = pl.DataFrame({"kind": ["gt_centre", "gt_centre", "uniform"],
                       "n_lm_15um": [1, 0, None]})
    fg = np.arange(9, dtype=np.float32).reshape(3, 3)
    fm = fg.copy()
    fm[0] = 99.0                                  # row 0 has a real maximum
    valid, source = feat_max_validity("c", df, {"feat_gt": fg, "feat_max": fm},
                                      schema=SCHEMA_V5)
    assert source == "v5-reconstructed-and-verified"
    assert valid.tolist() == [True, False, False]


def test_the_v5_reconstruction_refuses_when_the_fallback_is_not_feat_gt():
    """If an invalid row does not literally carry feat_gt, v5 semantics are not what the
    repair assumes -- refuse rather than fabricate a validity mask."""
    df = pl.DataFrame({"kind": ["gt_centre", "uniform"], "n_lm_15um": [0, None]})
    fg = np.zeros((2, 3), dtype=np.float32)
    fm = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(SystemExit, match="fallback"):
        feat_max_validity("c", df, {"feat_gt": fg, "feat_max": fm}, schema=SCHEMA_V5)


def test_an_exported_validity_column_is_used_verbatim():
    df = pl.DataFrame({"kind": ["gt_centre"] * 2, "n_lm_15um": [0, 0],
                       FEAT_MAX_VALID: [True, False]})
    valid, source = feat_max_validity("c", df, {}, schema=SCHEMA_V5)
    assert source == "export" and valid.tolist() == [True, False]


# ---------------------------------------------------------------------- defect 6: atomicity
def test_a_failed_write_leaves_neither_a_partial_nor_a_truncated_output(tmp_path):
    target = tmp_path / "out.parquet"

    def boom(p):
        Path(p).write_bytes(b"half")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        atomic_write(target, boom)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_catches_a_writer_that_renames_the_temp_file(tmp_path):
    """REGRESSION LOCK. `np.save(path, arr)` APPENDS '.npy', so writing the array through a
    path -- not a handle -- produced `out.npy.partial.npy`, os.replace raised
    FileNotFoundError and a stray file was left behind. Atomicity was defeated by the very
    call that was supposed to provide it."""
    target = tmp_path / "out.npy"
    with pytest.raises(SystemExit, match="did not produce"):
        atomic_write(target, lambda p: np.save(p, np.zeros(3), allow_pickle=False))


def test_save_npy_writes_through_a_handle_so_the_partial_name_survives(tmp_path):
    target = tmp_path / "out.npy"
    arr = np.arange(6, dtype=np.float32).reshape(3, 2)
    atomic_write(target, save_npy(arr))
    assert target.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["out.npy"], "a stray file was left"
    assert np.array_equal(np.load(target), arr)


def _report(tmp_path, fold="0", **over):
    import json
    r = {"derived_schema_version": DERIVED_SCHEMA_VERSION, "fold": fold,
         "match_authority": "pregraph", "representation": ASSOCIATION,
         "export_schema_version": SCHEMA_V5,
         "scorer": {"tracking_cellmot_files": {"a.py": "h"}}}
    r.update(over)
    (tmp_path / f"d1_derived_split{fold}.json").write_text(json.dumps(r), encoding="utf-8")
    (tmp_path / f"d1_derived_split{fold}.parquet").write_bytes(b"x")
    return {"tracking_cellmot_files": {"a.py": "h"}}


def test_a_second_fold_may_share_a_derived_directory(tmp_path):
    scorer = _report(tmp_path, "0")
    assert guard_out_dir(tmp_path, "1", "pregraph", scorer, representation=ASSOCIATION,
                         export_schema=SCHEMA_V5) == ["0"]


def test_rerunning_a_fold_needs_an_explicit_overwrite(tmp_path):
    scorer = _report(tmp_path, "0")
    with pytest.raises(SystemExit, match="overwrite"):
        guard_out_dir(tmp_path, "0", "pregraph", scorer, representation=ASSOCIATION,
                      export_schema=SCHEMA_V5)
    assert guard_out_dir(tmp_path, "0", "pregraph", scorer, representation=ASSOCIATION,
                         export_schema=SCHEMA_V5, overwrite=True) == ["0"]


@pytest.mark.parametrize("over,authority,match", [
    ({"derived_schema_version": "d1-derived-1"}, "pregraph", "schema"),
    ({}, "submission", "authorit"),
])
def test_mixed_derived_directories_are_refused(tmp_path, over, authority, match):
    scorer = _report(tmp_path, "0", **over)
    with pytest.raises(SystemExit, match=match):
        guard_out_dir(tmp_path, "1", authority, scorer, representation=ASSOCIATION,
                      export_schema=SCHEMA_V5)


def test_a_different_scorer_build_is_refused(tmp_path):
    _report(tmp_path, "0")
    with pytest.raises(SystemExit, match="scorer"):
        guard_out_dir(tmp_path, "1", "pregraph",
                      {"tracking_cellmot_files": {"a.py": "DIFFERENT"}},
                      representation=ASSOCIATION, export_schema=SCHEMA_V5)


def test_an_aborted_run_is_refused_rather_than_merged(tmp_path):
    scorer = _report(tmp_path, "0")
    (tmp_path / "d1_derived_split1.parquet.partial").write_bytes(b"x")
    with pytest.raises(SystemExit, match="aborted"):
        guard_out_dir(tmp_path, "1", "pregraph", scorer, representation=ASSOCIATION,
                      export_schema=SCHEMA_V5)


def test_an_artifact_with_no_report_is_refused(tmp_path):
    scorer = _report(tmp_path, "0")
    (tmp_path / "d1_derived_split3__feat_gt.npy").write_bytes(b"x")
    with pytest.raises(SystemExit, match="no matching report"):
        guard_out_dir(tmp_path, "1", "pregraph", scorer, representation=ASSOCIATION,
                      export_schema=SCHEMA_V5)


def test_an_empty_or_absent_directory_is_fine(tmp_path):
    assert guard_out_dir(tmp_path / "nope", "0", "pregraph", {},
                         representation=ASSOCIATION, export_schema=SCHEMA_V5) == []
    assert guard_out_dir(tmp_path, "0", "pregraph", {},
                         representation=ASSOCIATION, export_schema=SCHEMA_V5) == []


# =============================================================================================
# REPAIR 7: THE v6 READER INTEGRATION.
#
# The v6 export writes FOUR differently named arrays and declares `schema_version` in every
# terminal record. Until this landed the postprocessor declared
# FEATURE_ARRAYS = ("feat_gt", "feat_max") and `load_features` SystemExited with
# `missing feature array <crop>__feat_gt.npy` on the first v6 artifact it was pointed at.
# Every test in this section fails against that reader.
# =============================================================================================
AGGREGATE = ROOT / "scripts" / "kaggle_edits" / "d1_aggregate.py"
V6_ARRAYS = ("feat_tta_mean_gt", "feat_tta_mean_max", "feat_idview_gt", "feat_idview_max")


# The deployed block: 8 encode calls over 7 distinct views, because
# rot90(imgs,1,dims=(-2,-1)).transpose(-1,-2) IS imgs.flip(-1). The two counts are
# declared separately and must never be merged back into one field.
_V6_VIEW_DECLARATION = {
    "tta_view_set": ["identity", "rot90_k1", "rot90_k2", "rot90_k3", "flipY",
                     "transpose", "flipX", "flipX"],
    "n_encode_calls": 8, "n_distinct_views": 7}


def _v6_terminal(dataset="c", **over):
    """A v6 terminal record: the fields `_d1_flush` writes that dispatch actually reads."""
    rec = {"dataset": dataset, "status": "complete", "fold": "0",
           "checkpoint_sha256": "deadbeef", "n_rows": 18, "gt_rows": 6,
           "feat_rows": 18, "feat_dim": 32, "feat_finite": True,
           "gt_load_error": None, "exception": None,
           SCHEMA_DECLARATION: SCHEMA_V6, "feat_arrays": list(V6_ARRAYS),
           "match_um": MATCH_UM, "search_um": SEARCH_UM,
           **_V6_VIEW_DECLARATION}
    rec.update(over)
    return rec


def _v5_terminal(dataset="c", **over):
    """v5 declared nothing about itself: no schema_version, no feat_arrays, no radii."""
    rec = {"dataset": dataset, "status": "complete", "fold": "0",
           "checkpoint_sha256": "deadbeef", "n_rows": 18, "gt_rows": 6,
           "feat_rows": 18, "feat_dim": 32, "feat_finite": True,
           "gt_load_error": None, "exception": None}
    rec.update(over)
    return rec


def _terms(*recs):
    return {r["dataset"]: r for r in recs}


def _agg(**over):
    a = {"fold": "0", "COMPLETE": True, "match_um": MATCH_UM, "search_um": SEARCH_UM}
    a.update(over)
    return a


# ------------------------------------------------------------------ the name tables
def test_the_reader_knows_the_four_names_the_exporter_actually_writes():
    """The gap itself. These four names are read out of the exporter's source, so the two
    sides cannot drift apart without this failing."""
    assert FEATURE_ARRAYS_BY_SCHEMA[SCHEMA_V6] == V6_ARRAYS
    assert FEATURE_ARRAYS_BY_SCHEMA[SCHEMA_V5] == ("feat_gt", "feat_max")
    src = AUDIT.read_text(encoding="utf-8")
    for name in V6_ARRAYS:
        assert f"__{name}.npy" in src, f"the exporter no longer writes {name}"


def test_gt_and_max_arrays_partition_every_declared_array():
    """Each schema's arrays split cleanly into always-defined and validity-gated halves; a
    name in neither half would silently escape the NaN-sentinel contract."""
    for schema, declared in FEATURE_ARRAYS_BY_SCHEMA.items():
        halves = gt_arrays(schema) + max_arrays(schema)
        assert sorted(halves) == sorted(declared), schema
        assert not set(gt_arrays(schema)) & set(max_arrays(schema)), schema


# ------------------------------------------------------------------ the representation API
def test_the_two_v6_representations_are_distinct_and_named_for_what_they_are():
    """The TTA-mean pair is what `detect_head` scores; the identity-view pair is what
    `predict_edges` reads. Different tensors, different meanings, disjoint names."""
    assert representation_arrays(SCHEMA_V6, DETECTOR) == ("feat_tta_mean_gt",
                                                          "feat_tta_mean_max")
    assert representation_arrays(SCHEMA_V6, ASSOCIATION) == ("feat_idview_gt",
                                                             "feat_idview_max")
    assert set(representation_arrays(SCHEMA_V6, DETECTOR)).isdisjoint(
        representation_arrays(SCHEMA_V6, ASSOCIATION))
    assert set(REPRESENTATIONS[SCHEMA_V6]) == {DETECTOR, ASSOCIATION}


def test_a_v5_export_has_no_detector_representation_at_all():
    """v5 sampled the IDENTITY VIEW beside POST-TTA logits. Asking it for the detector
    representation must name B3, not quietly hand back the association arrays under the
    detector's name -- which is the whole of what made the v5 D1-F probe invalid."""
    assert representation_arrays(SCHEMA_V5, ASSOCIATION) == ("feat_gt", "feat_max")
    assert set(REPRESENTATIONS[SCHEMA_V5]) == {ASSOCIATION}
    with pytest.raises(SystemExit, match="B3"):
        representation_arrays(SCHEMA_V5, DETECTOR)


def test_no_caller_can_obtain_a_representation_without_naming_one():
    """The API-level guard: no default at the function, and `required=True` at the CLI."""
    import inspect
    sig = inspect.signature(representation_arrays)
    assert sig.parameters["representation"].default is inspect.Parameter.empty
    with pytest.raises(SystemExit) as exc:
        d1pp.main(["--fold-dir", "nowhere", "--out-dir", "nowhere"])
    assert exc.value.code == 2, "argparse must refuse a run with no --representation"


def test_an_unknown_schema_or_representation_is_named_rather_than_defaulted():
    with pytest.raises(SystemExit, match="unknown export schema"):
        representation_arrays("d1_v9", ASSOCIATION)
    with pytest.raises(SystemExit, match="has no"):
        representation_arrays(SCHEMA_V6, "whatever")


# ------------------------------------------------------------------ schema resolution
def test_a_declared_v6_export_needs_no_operator_flag():
    assert resolve_schema(_terms(_v6_terminal()), {}, v5_declared=False) == SCHEMA_V6


def test_an_undeclared_export_is_refused_until_v5_is_declared():
    """v5 predates the declaration. Silently assuming v5 is how a half-copied v6 directory
    would get read with the wrong array names."""
    terms = _terms(_v5_terminal())
    with pytest.raises(SystemExit, match=SCHEMA_DECLARATION):
        resolve_schema(terms, {}, v5_declared=False)
    assert resolve_schema(terms, {}, v5_declared=True) == SCHEMA_V5


def test_the_v5_flag_is_rejected_against_a_v6_export_not_silently_honoured():
    """Requirement 4. v6 ships a real row_id, so the flag declares an assumption that is
    both unnecessary and unverifiable here; honouring it would hide a real mistake."""
    with pytest.raises(SystemExit, match="v5-emission-order-row-id"):
        resolve_schema(_terms(_v6_terminal()), {}, v5_declared=True)


def test_an_unrecognised_schema_declaration_is_fatal():
    with pytest.raises(SystemExit, match="unrecognised"):
        resolve_schema(_terms(_v6_terminal(schema_version="d1_v7")), {}, v5_declared=False)


def test_crops_declaring_different_schemas_are_fatal():
    with pytest.raises(SystemExit, match="two schemas"):
        resolve_schema(_terms(_v6_terminal("a"), _v5_terminal("b")), {}, v5_declared=False)


def test_the_aggregate_declaration_must_agree_with_the_terminal_records():
    with pytest.raises(SystemExit, match="aggregate manifest declares"):
        resolve_schema(_terms(_v6_terminal()), {SCHEMA_DECLARATION: SCHEMA_V5},
                       v5_declared=False)


@pytest.mark.parametrize("bad", [None, [], ["feat_gt", "feat_max"],
                                 ["feat_tta_mean_gt", "feat_tta_mean_max"]])
def test_a_v6_record_must_declare_the_four_arrays_it_ships(bad):
    with pytest.raises(SystemExit, match="feat_arrays"):
        resolve_schema(_terms(_v6_terminal(feat_arrays=bad)), {}, v5_declared=False)


def test_the_schema_is_never_inferred_from_the_files_on_disk(tmp_path):
    """THE DEFECT THIS DISPATCH PREVENTS. A directory can hold the four v6 arrays and still
    not be a v6 export -- a half-finished copy, a hand-assembled fixture, a future schema.
    Guessing from filenames pairs one representation's NAME with the other's DATA."""
    import inspect
    assert not {"audit_dir", "path", "dir", "fold_dir"} & set(
        inspect.signature(resolve_schema).parameters), (
        "resolve_schema gained a directory argument; the schema must come from the "
        "declaration, never from a glob over the filenames")
    for name in V6_ARRAYS:
        np.save(tmp_path / f"c__{name}.npy", np.zeros((4, 32), dtype=np.float32))
    with pytest.raises(SystemExit, match="Refusing to guess"):
        resolve_schema(_terms(_v5_terminal()), {}, v5_declared=False)


# ------------------------------------------------------------------ load_features
def _write_arrays(tmp_path, crop, names, n=6, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for i, name in enumerate(names):
        arr = (rng.standard_normal((n, dim)) + i).astype(np.float32)
        np.save(tmp_path / f"{crop}__{name}.npy", arr)
        out[name] = arr
    return out


def test_load_features_reads_the_v6_names_the_old_reader_could_not(tmp_path):
    """THE GAP, closed. The pre-repair reader hardcoded ("feat_gt", "feat_max") and died
    with `missing feature array <crop>__feat_gt.npy` on the first v6 artifact."""
    want = _write_arrays(tmp_path, "c", V6_ARRAYS)
    got = load_features(tmp_path, "c", 6, FEATURE_ARRAYS_BY_SCHEMA[SCHEMA_V6])
    assert set(got) == set(V6_ARRAYS)
    for name in V6_ARRAYS:
        assert np.array_equal(got[name], want[name])
    with pytest.raises(SystemExit, match="missing feature array"):
        load_features(tmp_path, "c", 6, FEATURE_ARRAYS_BY_SCHEMA[SCHEMA_V5])


# ------------------------------------------------- the v5 reconstruction must not touch v6
def test_the_v5_reconstruction_is_refused_by_schema_before_it_can_touch_v6_data():
    """Repair 7, item 2 -- gated on the DECLARED SCHEMA, not on the data.

    Reaching the v5 rule with v6 rows is not harmless. The reconstruction infers validity
    from `is_gt & n_lm_15um > 0` and then verifies it against the v5 silent fallback
    (`feat_max == feat_gt`). v6 writes NaN there instead, so the verification fails and
    blames the DATA -- "v5 feat_max semantics are not what this repair assumes" -- when the
    actual fault is the reader being pointed at the wrong schema. An operator who "fixed"
    that message by relaxing the verification would then get a mask that is not the one the
    export declares. Refusing on the declaration keeps the diagnosis correct.
    """
    df = pl.DataFrame({"kind": ["gt_centre", "gt_centre", "uniform"],
                       "n_lm_15um": [1, 0, None]})
    fg = np.arange(9, dtype=np.float32).reshape(3, 3)
    fm = fg.copy()
    fm[1:] = np.nan                       # the v6 NaN sentinel, not a copy of feat_gt
    feats = {"feat_gt": fg, "feat_max": fm}
    with pytest.raises(SystemExit, match="Refusing"):
        feat_max_validity("c", df, feats, schema=SCHEMA_V6)
    # DEFECT REINJECTION: the identical call under the v5 schema reaches the v5 rule and
    # misattributes the failure to the export.
    with pytest.raises(SystemExit, match="v5 feat_max semantics"):
        feat_max_validity("c", df, feats, schema=SCHEMA_V5)


def test_a_v6_export_always_supplies_its_own_validity_column():
    """v6 sets feat_max_valid on EVERY emitted row, so the reader never needs to guess.
    Asserted against the exporter's source so the two cannot drift apart."""
    src = AUDIT.read_text(encoding="utf-8")
    assert 'row["feat_max_valid"] = bool(best15 is not None)' in src
    df = pl.DataFrame({"kind": ["gt_centre", "uniform"], "n_lm_15um": [1, None],
                       FEAT_MAX_VALID: [True, False]})
    valid, source = feat_max_validity("c", df, {}, schema=SCHEMA_V6)
    assert source == "export" and valid.tolist() == [True, False]


# ------------------------------------------------------------ the accumulator-collapse check
def test_a_collapsed_tta_accumulator_is_refused_on_read():
    """If the TTA-mean array is byte-for-byte the identity view, the export contains no
    detector representation and handing it to the probe is B3 wearing the v6 name."""
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    feats = {"feat_tta_mean_gt": a, "feat_idview_gt": a.copy()}
    with pytest.raises(SystemExit, match="B3"):
        assert_representations_differ("c", SCHEMA_V6, feats)


def test_distinct_representations_pass_and_record_their_separation():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = assert_representations_differ(
        "c", SCHEMA_V6, {"feat_tta_mean_gt": a + 0.5, "feat_idview_gt": a})
    assert out["arrays"] == ["feat_tta_mean_gt", "feat_idview_gt"]
    assert out["max_abs_diff"] == pytest.approx(0.5)


def test_the_collapse_check_does_not_apply_to_v5():
    """v5 has one pair, so there is nothing to compare; the check must be inert, not wrong."""
    assert assert_representations_differ("c", SCHEMA_V5, {"feat_gt": np.zeros((2, 2))}) is None


# ------------------------------------------------------------------ the radius binding (3)
def test_the_promoted_radii_arm_a_check_that_was_inert_against_v5():
    """Repair 7, item 3. v5 recorded no radius anywhere, so `assert_export_radii` had
    nothing to compare and the binding to the scorer's MAX_DISTANCE was decorative."""
    v5 = assert_export_radii({"fold": "0"}, _terms(_v5_terminal()), SCHEMA_V5)
    assert v5 == {"match_um": None, "search_um": None, "recorded": False}
    v6 = assert_export_radii(_agg(), _terms(_v6_terminal()), SCHEMA_V6)
    assert v6 == {"match_um": MATCH_UM, "search_um": SEARCH_UM, "recorded": True}


def test_a_v6_export_whose_radii_were_not_promoted_is_refused():
    """The inert state is now fatal for v6: the per-crop records carry the radii, so a
    top-level absence means the aggregator did not promote them."""
    with pytest.raises(SystemExit, match="promote"):
        assert_export_radii({"fold": "0"}, _terms(_v6_terminal()), SCHEMA_V6)


def test_an_export_radius_that_disagrees_with_the_scorer_is_fatal():
    with pytest.raises(SystemExit, match="different radius"):
        assert_export_radii(_agg(match_um=8.0), _terms(_v6_terminal(match_um=8.0)),
                            SCHEMA_V6)


def test_a_promoted_radius_with_no_per_crop_source_is_fatal():
    with pytest.raises(SystemExit, match="no source"):
        assert_export_radii(_agg(), _terms(_v6_terminal(match_um=None)), SCHEMA_V6)


def test_per_crop_radii_that_disagree_with_the_promotion_are_fatal():
    """Two radii inside one export means different crops' neighbourhood statistics are not
    comparable. A promotion that quietly picked one would be silent corruption."""
    terms = _terms(_v6_terminal("a"), _v6_terminal("b", search_um=20.0))
    with pytest.raises(SystemExit, match="two different radii"):
        assert_export_radii(_agg(), terms, SCHEMA_V6)


def test_the_export_radius_gate_follows_the_scorer_and_is_not_a_literal(monkeypatch):
    """MATCH_UM IS the scorer's MAX_DISTANCE. Move the binding and the identical export that
    passed a moment ago is refused -- which is exactly what "not a literal" buys."""
    assert assert_export_radii(_agg(), _terms(_v6_terminal()), SCHEMA_V6)["match_um"] == 7.0
    monkeypatch.setattr(d1pp, "MATCH_UM", 9.0)
    with pytest.raises(SystemExit, match="different radius"):
        assert_export_radii(_agg(), _terms(_v6_terminal()), SCHEMA_V6)


# ------------------------------------------------------------------ the parent aggregator
class _FakeOs:
    """The aggregator cell reads `os.environ` from the notebook's globals. A shim keeps the
    real process environment untouched."""

    def __init__(self, env):
        self.environ = dict(env)


def _run_aggregator(root, records, *, fold="0", ckpt="deadbeef"):
    """Execute the SHIPPED aggregator cell with only its /kaggle/working literal redirected.

    Same harness shape tests/test_d1_v6_export.py uses for the audit block: one literal is
    rewritten so the tested object stays byte-identical to the shipped one everywhere that
    matters. Returns (manifest, RuntimeError message or None).
    """
    import json as _json
    src = AGGREGATE.read_text(encoding="utf-8")
    assert src.count('_AggPath("/kaggle/working/d1_audit")') == 1, (
        "the aggregator's Kaggle root literal changed; this harness cannot redirect it")
    src = src.replace('_AggPath("/kaggle/working/d1_audit")', f'_AggPath(r"{root}")', 1)
    man_dir = Path(root) / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    for stem, rec in records.items():
        (man_dir / f"{stem}.start.json").write_text(_json.dumps({"dataset": stem}),
                                                    encoding="utf-8")
        (man_dir / f"{stem}.complete.json").write_text(_json.dumps(rec), encoding="utf-8")
    g = {"os": _FakeOs({"BIOHUB_LOEO_STEMS": _json.dumps(sorted(records)),
                        "BIOHUB_D1_FOLD": fold, "BIOHUB_D1_CKPT_SHA": ckpt})}
    failure = None
    try:
        exec(compile(src, "d1_aggregate", "exec"), g)
    except RuntimeError as exc:
        failure = str(exc)
    out = Path(root) / "d1_manifest.json"
    return (_json.loads(out.read_text(encoding="utf-8")) if out.exists() else None), failure


def test_the_aggregator_promotes_the_per_crop_declarations_to_the_top_level(tmp_path):
    """Repair 7, item 3. v6 records the radii PER CROP; the postprocessor reads the TOP
    level. Without this promotion the scorer binding never sees the export at all."""
    man, failure = _run_aggregator(tmp_path, {s: _v6_terminal(s) for s in ("a", "b")})
    assert failure is None, failure
    assert man["COMPLETE"] is True
    assert man["match_um"] == MATCH_UM and man["search_um"] == SEARCH_UM
    assert man[SCHEMA_DECLARATION] == SCHEMA_V6
    # d1f_probe.validate_manifest blocks without these, and it blocks on the ARTIFACT --
    # i.e. after the GPU has been paid for. They are promoted separately, never merged.
    assert man["n_encode_calls"] == 8 and man["n_distinct_views"] == 7
    assert len(man["tta_view_set"]) == 8 and len(set(man["tta_view_set"])) == 7
    assert "n_views" not in man


def test_crops_that_disagree_on_a_radius_fail_the_aggregation(tmp_path):
    """A per-crop disagreement means one export used two radii. Fatal, never a vote."""
    man, failure = _run_aggregator(
        tmp_path, {"a": _v6_terminal("a"), "b": _v6_terminal("b", match_um=8.0)})
    assert failure is not None
    assert any("disagree on match_um" in p for p in man["problems"]), man["problems"]
    assert man["COMPLETE"] is False
    assert man["match_um"] is None, "a disagreed field must not be promoted anyway"


def test_an_export_that_records_no_radius_fails_the_aggregation(tmp_path):
    man, failure = _run_aggregator(tmp_path, {"a": _v6_terminal("a", search_um=None)})
    assert failure is not None
    assert any("no crop records search_um" in p for p in man["problems"]), man["problems"]


def test_the_promoted_manifest_is_exactly_what_the_postprocessor_checks(tmp_path):
    """THE HANDSHAKE, end to end: the exporter declares per crop, the aggregator promotes,
    the postprocessor's scorer-bound gate accepts. Any one of the three drifting breaks it."""
    records = {s: _v6_terminal(s) for s in ("a", "b")}
    man, failure = _run_aggregator(tmp_path, records)
    assert failure is None, failure
    assert assert_export_radii(man, records, SCHEMA_V6) == {
        "match_um": MATCH_UM, "search_um": SEARCH_UM, "recorded": True}
    assert resolve_schema(records, man, v5_declared=False) == SCHEMA_V6


# ------------------------------------------------------- derived directories stay unmixed
def test_the_two_representations_may_not_share_a_derived_directory(tmp_path):
    """They are different tensors. Anything concatenating folds from one directory would
    otherwise blend the detector features of one fold with the association features of
    another and never notice."""
    scorer = _report(tmp_path, "0")             # written as the association representation
    with pytest.raises(SystemExit, match="representation"):
        guard_out_dir(tmp_path, "1", "pregraph", scorer, representation=DETECTOR,
                      export_schema=SCHEMA_V5)


def test_two_export_schemas_may_not_share_a_derived_directory(tmp_path):
    scorer = _report(tmp_path, "0")
    with pytest.raises(SystemExit, match="export schema"):
        guard_out_dir(tmp_path, "1", "pregraph", scorer, representation=ASSOCIATION,
                      export_schema=SCHEMA_V6)


# =============================================================================================
# END TO END against the REAL v5 pull, rewritten into the v6 SHAPE.
#
# No v6 export exists yet, so the fixture is synthetic -- but it is synthesised FROM the real
# artifacts and its aggregate manifest is built by the SHIPPED aggregator, so the only invented
# things are the four array names, the declarations and a deterministic offset between the two
# representations. The partition must not move: 44b6_0113de3b is 52/52 under pregraph authority
# and 49/52 under submission. That fold is IN-FAMILY and carries NO 6bba representation.
# =============================================================================================
def _research_root():
    for base in (ROOT, *ROOT.parents):
        cand = base.parent / "Biohub-CellTracking-2026_RESEARCH"
        if cand.is_dir():
            return cand
    return None


_RESEARCH = _research_root()
V5_F0 = (_RESEARCH / "agent_runs" / "v6_build_20260806" / "agent4" / "v5_pull" / "f0"
         if _RESEARCH else None)
GT_DIR = ROOT / "data" / "train"
CROP = "44b6_0113de3b"
HAVE_V5 = bool(V5_F0 and V5_F0.is_dir() and (GT_DIR / f"{CROP}.geff").exists())
needs_v5 = pytest.mark.skipif(not HAVE_V5, reason="v5 pull or data/train not present")


def _v6_fixture_from_v5(dst: Path) -> Path:
    """Rewrite the REAL v5 fold-0 export into the v6 SHAPE without changing a measurement.

    Rows, neighbourhood statistics and both graphs are copied verbatim. Only what v6
    DECLARES is added: row_id, feat_max_valid with honest NaN sentinels, schema_version,
    feat_arrays, the radii, and the four array names.

    The identity-view pair IS the v5 pair, so a --representation association read of this
    fixture must reproduce the v5 read byte for byte. The TTA-mean pair is a deterministic
    offset of it, so the two representations are genuinely distinct.
    """
    import json as _json
    import shutil as _shutil
    src_audit = V5_F0 / "d1_audit"
    dst_audit = dst / "d1_audit"
    dst_audit.mkdir(parents=True)
    for name in ("loeo_split0_strict.csv.gz", "pregraphs_split0.parquet"):
        _shutil.copy2(V5_F0 / name, dst / name)

    df = pl.read_parquet(src_audit / f"{CROP}__rows.parquet")
    fg = np.load(src_audit / f"{CROP}__feat_gt.npy")
    fm = np.load(src_audit / f"{CROP}__feat_max.npy")
    is_gt = (df.get_column("kind") == "gt_centre").to_numpy()
    n15 = df.get_column("n_lm_15um").fill_null(0).cast(pl.Int64).to_numpy()
    valid = is_gt & (n15 > 0)
    assert valid.any() and not valid.all(), "the fixture must exercise both sides"

    delta = (np.arange(1, fg.shape[1] + 1, dtype=np.float32) * np.float32(0.001))
    arrays = {
        "feat_tta_mean_gt": (fg + delta).astype(np.float32),
        "feat_tta_mean_max": apply_validity((fm + delta).astype(np.float32), valid),
        "feat_idview_gt": fg,
        "feat_idview_max": apply_validity(fm, valid),
    }
    for name, arr in arrays.items():
        np.save(dst_audit / f"{CROP}__{name}.npy", arr)
    df.with_columns([
        pl.arange(0, df.height, dtype=pl.Int64).alias(ROW_ID),
        pl.Series(FEAT_MAX_VALID, valid, dtype=pl.Boolean),
    ]).write_parquet(dst_audit / f"{CROP}__rows.parquet")

    v5rec = _json.loads(
        (src_audit / "manifests" / f"{CROP}.complete.json").read_text(encoding="utf-8"))
    rec = dict(v5rec)
    rec.update({SCHEMA_DECLARATION: SCHEMA_V6, "feat_arrays": list(V6_ARRAYS),
                "match_um": MATCH_UM, "search_um": SEARCH_UM,
                **_V6_VIEW_DECLARATION})
    man, failure = _run_aggregator(dst_audit, {CROP: rec}, fold="0",
                                   ckpt=v5rec["checkpoint_sha256"])
    assert failure is None, failure
    assert man["COMPLETE"] and man["match_um"] == MATCH_UM
    return dst


@pytest.fixture(scope="module")
def v6_roundtrip(tmp_path_factory):
    if not HAVE_V5:
        pytest.skip("v5 pull or data/train not present")
    tmp = tmp_path_factory.mktemp("v6rt")
    fold = _v6_fixture_from_v5(tmp / "v6_fold")
    outs = {k: tmp / f"derived_{k}" for k in ("v5", "assoc", "det")}
    assert d1pp.main(["--fold-dir", str(V5_F0), "--out-dir", str(outs["v5"]),
                      "--gt-dir", str(GT_DIR), "--representation", ASSOCIATION,
                      "--v5-emission-order-row-id"]) == 0
    for key, rep in (("assoc", ASSOCIATION), ("det", DETECTOR)):
        assert d1pp.main(["--fold-dir", str(fold), "--out-dir", str(outs[key]),
                          "--gt-dir", str(GT_DIR), "--representation", rep]) == 0
    return {**outs, "fold": fold}


@needs_v5
def test_a_v6_export_reproduces_the_v5_partition_exactly(v6_roundtrip):
    """THE REGRESSION. Reading the v6 shape must not move a single number."""
    import json as _json
    for key in ("v5", "assoc", "det"):
        r = _json.loads(
            (v6_roundtrip[key] / "d1_derived_split0.json").read_text(encoding="utf-8"))
        assert r["totals"] == {"n_gt": 52, "M": 52, "C": 0, "T": 0, "L": 0, "D": 0}, key
        c = r["census"][0]
        assert c["matched_pregraph"] == 52 and c["matched_submission"] == 49, key


@needs_v5
def test_the_v6_association_read_reproduces_the_v5_arrays_byte_for_byte(v6_roundtrip):
    """v5's arrays ARE the identity view. Reading them under the name that says so must
    change nothing but the name."""
    for v5_name, v6_name in (("feat_gt", "feat_idview_gt"),
                             ("feat_max", "feat_idview_max")):
        a = (v6_roundtrip["v5"] / f"d1_derived_split0__{v5_name}.npy").read_bytes()
        b = (v6_roundtrip["assoc"] / f"d1_derived_split0__{v6_name}.npy").read_bytes()
        assert a == b, f"{v6_name} is not byte-identical to the v5 {v5_name}"


@needs_v5
def test_the_detector_and_association_reads_are_different_numbers(v6_roundtrip):
    a = np.load(v6_roundtrip["det"] / "d1_derived_split0__feat_tta_mean_gt.npy")
    b = np.load(v6_roundtrip["assoc"] / "d1_derived_split0__feat_idview_gt.npy")
    assert a.shape == b.shape
    assert not np.array_equal(a, b), "the two representations collapsed into one"


@needs_v5
def test_the_v6_read_gathers_features_through_the_sort_not_positionally(v6_roundtrip):
    """The non-negotiable, checked on the v6 path: a positional join was wrong for 51/52,
    1646/1659 and 1354/1368 GT rows. The derived feature row must be the RAW row whose
    row_id the derived parquet records."""
    d = pl.read_parquet(v6_roundtrip["det"] / "d1_derived_split0.parquet")
    raw = np.load(v6_roundtrip["fold"] / "d1_audit" / f"{CROP}__feat_tta_mean_gt.npy")
    got = np.load(v6_roundtrip["det"] / "d1_derived_split0__feat_tta_mean_gt.npy")
    rid = d.get_column(ROW_ID).to_numpy()
    assert np.array_equal(got, raw[rid])
    assert not np.array_equal(raw[: d.height], got), (
        "the fixture no longer exercises the misalignment defect")


@needs_v5
def test_the_derived_artifact_says_which_representation_it_holds(v6_roundtrip):
    import json as _json
    d = pl.read_parquet(v6_roundtrip["det"] / "d1_derived_split0.parquet")
    assert d.get_column("representation").unique().to_list() == [DETECTOR]
    assert d.get_column("export_schema_version").unique().to_list() == [SCHEMA_V6]
    r = _json.loads(
        (v6_roundtrip["det"] / "d1_derived_split0.json").read_text(encoding="utf-8"))
    assert r["feature_arrays"] == {"gt": "feat_tta_mean_gt", "max": "feat_tta_mean_max"}
    assert r["declared_feat_arrays"] == list(V6_ARRAYS)
    assert r["export_radii"] == {"match_um": MATCH_UM, "search_um": SEARCH_UM,
                                 "recorded": True}
    prov = r["provenance"][CROP]
    assert prov["row_id_source"] == "export", "v6 must not need the v5 emission-order flag"
    assert prov["feat_max_valid_source"] == "export", "the v5 reconstruction ran against v6"
    assert prov["representations_differ"]["max_abs_diff"] > 0


@needs_v5
def test_the_v5_flag_is_refused_against_the_v6_fixture(v6_roundtrip, tmp_path):
    with pytest.raises(SystemExit, match="v5-emission-order-row-id"):
        d1pp.main(["--fold-dir", str(v6_roundtrip["fold"]),
                   "--out-dir", str(tmp_path / "x"), "--gt-dir", str(GT_DIR),
                   "--representation", ASSOCIATION, "--v5-emission-order-row-id"])


@needs_v5
def test_the_real_v5_export_cannot_be_asked_for_the_detector_representation(tmp_path):
    with pytest.raises(SystemExit, match="B3"):
        d1pp.main(["--fold-dir", str(V5_F0), "--out-dir", str(tmp_path / "y"),
                   "--gt-dir", str(GT_DIR), "--representation", DETECTOR,
                   "--v5-emission-order-row-id"])
