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
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from d1_postprocess import (  # noqa: E402
    DERIVED_SCHEMA_VERSION,
    MATCH_UM,
    REQUIRED_ROW_COLUMNS,
    SEARCH_UM,
    classify,
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
    assert DERIVED_SCHEMA_VERSION == "d1-derived-1"


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
def test_deployed_tta_is_eight_views_derived_from_the_generated_notebook():
    """THE DEPLOYED VIEW COUNT IS 8, not the vendored 4.

    Runtime accumulation is identity(1) + 3 flips + rot90(k=1,3) + transpose +
    anti-transpose = 8, divided by the RUNTIME COUNTER `_nv`. The literal `_nv += 1`
    appears 4 times because three of the views are produced inside loops -- counting the
    literal is wrong, so this asserts the view SET.
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
        # 1 identity + 3 flips + 2 rots + 1 transpose + 1 anti-transpose
        assert 1 + 3 + 2 + 1 + 1 == 8


@pytest.mark.skipif(not all(p.exists() for p in NOTEBOOKS), reason="built notebooks absent")
def test_a_second_independent_tta_block_exists_for_the_secondary_model():
    """There are TWO 8-view TTA blocks -- primary and secondary detection models. Anything
    reasoning about 'the' deployed detection field must account for both."""
    src = _nb_source(NOTEBOOKS[0])
    assert src.count("_nv = 1") == 2, "the secondary-model TTA block moved or merged"
    assert src.count("_nv += 1") == 8, "4 literal increments per block x 2 blocks"


@pytest.mark.skipif(not all(p.exists() for p in NOTEBOOKS), reason="built notebooks absent")
def test_tta_patch_failure_is_silent_and_must_become_fatal_in_v6():
    """The patch guard falls back to 4 views with a PRINT, not a raise, and the view count
    is recorded in no manifest field. v6 must raise and record `n_views`."""
    src = _nb_source(NOTEBOOKS[0])
    assert 'print("TTA WARNING: block not found - using default 4-way")' in src, (
        "if this became a raise, update this lock -- the silent fallback is the defect"
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
