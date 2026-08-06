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
@pytest.mark.skipif(not PREDICT.exists(), reason="vendored predict script not present")
def test_tta_averages_logits_over_exactly_four_views():
    """The deployed view list is identity + 3 planar flips, and logits are divided by 4."""
    src = PREDICT.read_text(encoding="utf-8")
    assert "tta_flips = [(-1,), (-2,), (-2, -1)]" in src, "TTA view list changed"
    assert "det_logits[f] = det_logits[f] + det_flip[f].flip(dims)" in src
    assert "det_logits[f] = det_logits[f] / 4" in src, "TTA denominator changed"


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


@pytest.mark.skipif(not AUDIT.exists(), reason="audit edit not present")
def test_audit_passes_identity_view_features_alongside_post_tta_logits():
    """Records the pairing that makes v5 D1-F invalid, so a v6 fix is a conscious change."""
    src = AUDIT.read_text(encoding="utf-8")
    assert "def _d1_audit_frame(dataset, gt_dir, t, logits_1zyx, feats_czyx" in src
    assert "feat_max" in src and "feat_gt" in src
    assert "__feat_tta_mean_gt.npy" not in src, (
        "TTA-mean features appear present -- this is the v6 contract, update the locks"
    )
