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

from d1_postprocess import (  # noqa: E402
    DERIVED_SCHEMA_VERSION,
    FEAT_MAX_VALID,
    MATCH_UM,
    REQUIRED_ROW_COLUMNS,
    ROW_ID,
    SEARCH_UM,
    apply_validity,
    assert_radius_binding,
    atomic_write,
    attach_row_id,
    check_feature_contract,
    classify,
    feat_max_validity,
    guard_out_dir,
    matched_gt_ids,
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
    """Changing the derivation MUST change this string, or old artifacts silently mix."""
    assert DERIVED_SCHEMA_VERSION == "d1-derived-2"


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
    """There are TWO 8-view TTA blocks -- primary and secondary detection models. Anything
    reasoning about 'the' deployed detection field must account for both."""
    src = _nb_source(NOTEBOOKS[0])
    assert src.count("_nv = 1") == 2, "the secondary-model TTA block moved or merged"
    assert src.count("_nv += 1") == 8, "4 literal increments per block x 2 blocks"


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
    """
    src = _nb_source(NOTEBOOKS[0])
    assert 'print("TTA WARNING: block not found - using default 4-way")' in src, (
        "the guard changed; re-check whether the downstream anchor still protects it"
    )
    after = src[src.find("TTA WARNING"):]
    assert "det_logits[f] = det_logits[f] / _nv" in after[:1200], (
        "the downstream anchor no longer references _nv, so a failed TTA patch would "
        "become genuinely silent -- restore the anchor or make the guard raise"
    )
    assert "n_views" not in src, "n_views now recorded; update the v6 locks"


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
    """Records the pairing that makes v5 D1-F invalid, so a v6 fix is a conscious change."""
    src = AUDIT.read_text(encoding="utf-8")
    assert "def _d1_audit_frame(dataset, gt_dir, t, logits_1zyx, feats_czyx" in src
    assert "feat_max" in src and "feat_gt" in src
    assert "__feat_tta_mean_gt.npy" not in src, (
        "TTA-mean features appear present -- this is the v6 contract, update the locks"
    )


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
    valid, source = feat_max_validity("c", df, {"feat_gt": fg, "feat_max": fm})
    assert source == "v5-reconstructed-and-verified"
    assert valid.tolist() == [True, False, False]


def test_the_v5_reconstruction_refuses_when_the_fallback_is_not_feat_gt():
    """If an invalid row does not literally carry feat_gt, v5 semantics are not what the
    repair assumes -- refuse rather than fabricate a validity mask."""
    df = pl.DataFrame({"kind": ["gt_centre", "uniform"], "n_lm_15um": [0, None]})
    fg = np.zeros((2, 3), dtype=np.float32)
    fm = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(SystemExit, match="fallback"):
        feat_max_validity("c", df, {"feat_gt": fg, "feat_max": fm})


def test_an_exported_validity_column_is_used_verbatim():
    df = pl.DataFrame({"kind": ["gt_centre"] * 2, "n_lm_15um": [0, 0],
                       FEAT_MAX_VALID: [True, False]})
    valid, source = feat_max_validity("c", df, {})
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
         "match_authority": "pregraph", "scorer": {"tracking_cellmot_files": {"a.py": "h"}}}
    r.update(over)
    (tmp_path / f"d1_derived_split{fold}.json").write_text(json.dumps(r), encoding="utf-8")
    (tmp_path / f"d1_derived_split{fold}.parquet").write_bytes(b"x")
    return {"tracking_cellmot_files": {"a.py": "h"}}


def test_a_second_fold_may_share_a_derived_directory(tmp_path):
    scorer = _report(tmp_path, "0")
    assert guard_out_dir(tmp_path, "1", "pregraph", scorer) == ["0"]


def test_rerunning_a_fold_needs_an_explicit_overwrite(tmp_path):
    scorer = _report(tmp_path, "0")
    with pytest.raises(SystemExit, match="overwrite"):
        guard_out_dir(tmp_path, "0", "pregraph", scorer)
    assert guard_out_dir(tmp_path, "0", "pregraph", scorer, overwrite=True) == ["0"]


@pytest.mark.parametrize("over,authority,match", [
    ({"derived_schema_version": "d1-derived-1"}, "pregraph", "schema"),
    ({}, "submission", "authorit"),
])
def test_mixed_derived_directories_are_refused(tmp_path, over, authority, match):
    scorer = _report(tmp_path, "0", **over)
    with pytest.raises(SystemExit, match=match):
        guard_out_dir(tmp_path, "1", authority, scorer)


def test_a_different_scorer_build_is_refused(tmp_path):
    _report(tmp_path, "0")
    with pytest.raises(SystemExit, match="scorer"):
        guard_out_dir(tmp_path, "1", "pregraph",
                      {"tracking_cellmot_files": {"a.py": "DIFFERENT"}})


def test_an_aborted_run_is_refused_rather_than_merged(tmp_path):
    scorer = _report(tmp_path, "0")
    (tmp_path / "d1_derived_split1.parquet.partial").write_bytes(b"x")
    with pytest.raises(SystemExit, match="aborted"):
        guard_out_dir(tmp_path, "1", "pregraph", scorer)


def test_an_artifact_with_no_report_is_refused(tmp_path):
    scorer = _report(tmp_path, "0")
    (tmp_path / "d1_derived_split3__feat_gt.npy").write_bytes(b"x")
    with pytest.raises(SystemExit, match="no matching report"):
        guard_out_dir(tmp_path, "1", "pregraph", scorer)


def test_an_empty_or_absent_directory_is_fine(tmp_path):
    assert guard_out_dir(tmp_path / "nope", "0", "pregraph", {}) == []
    assert guard_out_dir(tmp_path, "0", "pregraph", {}) == []
