"""The cache auditor must be shown to REJECT, not asserted to work (PKT-0037).

A checker that has never been demonstrated rejecting anything is indistinguishable from a
checker that returns True. This project has already paid for that lesson twice: a test that
grepped source instead of running it, and a Gate-1 harness that compared zero crops and would
have licensed two GPU sessions had it defaulted to a pass (`FACT-0387`).

So these tests do not read `audit_feature_cache.py`. They manufacture caches on disk with a
specific defect planted in each, run the real auditor over them, and require a rejection whose
REASON names the planted defect - because "it rejected" and "it rejected for the right reason"
are different claims, and only the second one helps whoever has to fix the cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import audit_feature_cache as A  # noqa: E402


@pytest.fixture()
def bench(tmp_path):
    """A clean fold-0 cache, its trunk, and a manifest that binds them."""
    nb = tmp_path / "notebook.ipynb"
    nb.write_text('{"cells": []}', encoding="utf-8")
    trunk = tmp_path / "trunk_official.pth"
    trunk.write_bytes(b"OFFICIAL-TRUNK-BYTES" * 64)
    cache = tmp_path / "cache"
    cache.mkdir()
    A._synth_cache(cache / "44b6_aaaaaaaa.npz", crop="44b6_aaaaaaaa", trunk_seed=1)
    args = A._bind_args(cache, trunk, fold="0", role="official",
                        weights_glob="/kaggle/input/*/split_0/edge_predictor_best.pth",
                        notebook=nb)
    manifest = cache / "cache_manifest.json"
    manifest.write_text(json.dumps(A.build_manifest(args), indent=2), encoding="utf-8")
    return {"root": tmp_path, "cache": cache, "manifest": manifest, "trunk": trunk,
            "notebook": nb}


def test_the_clean_cache_is_accepted(bench):
    """The accept case is half the proof: an auditor that rejects everything checks nothing."""
    report = A.audit(bench["cache"], bench["manifest"])
    assert report["passed"] is True
    assert report["crops"] == 1
    assert any(c["check"] == "features_still_bound_to_the_recorded_trunk" for c in report["checks"])


def test_the_manifest_binds_every_field_the_packet_requires(bench):
    """PKT-0037 task 4 enumerated the binding. Assert the fields exist rather than trusting a
    docstring - a manifest missing one of these cannot disambiguate a null."""
    m = json.loads(bench["manifest"].read_text(encoding="utf-8"))
    assert m["trunk"]["sha256"] and m["trunk"]["provenance"] and m["trunk"]["role"]
    assert m["fold"]["fold"] and m["fold"]["held_out_embryo"] and m["fold"]["crops"]
    assert m["feature_normalisation"]["transform"]
    crop = m["crops"][0]
    assert crop["node_order"]["convention"] and crop["node_order"]["coords_digest"]
    assert crop["node_order"]["features_digest"]
    assert m["candidate_graph"]["rule"] and m["candidate_graph"]["deployed_floor"] is not None
    assert m["provenance"]["notebook"] and m["provenance"]["spec"]
    assert m["provenance"]["source_commit"]


def test_a_swapped_trunk_is_rejected_by_the_feature_binding(bench):
    """FACT-0392's third risk in its realistic accidental form: two trunk caches produced in one
    session and the wrong directory consumed. Node set, shapes and counts are identical; only
    the 32-dim feature values differ."""
    A._synth_cache(bench["cache"] / "44b6_aaaaaaaa.npz", crop="44b6_aaaaaaaa", trunk_seed=2)
    with pytest.raises(A.Reject, match="features_unchanged"):
        A.audit(bench["cache"], bench["manifest"])


def test_a_swapped_trunk_checkpoint_is_rejected_by_its_hash(bench):
    other = bench["root"] / "trunk_stabledet.pth"
    other.write_bytes(b"STABLEDET-TRUNK-BYTE" * 64)
    with pytest.raises(A.Reject, match="trunk_checkpoint_sha256"):
        A.audit(bench["cache"], bench["manifest"], trunk=other)


def test_a_consumer_can_refuse_a_cache_from_the_wrong_trunk(bench):
    """The other half of the dual-trunk protection: a head trained on one trunk asserts which
    cache it is allowed to read, instead of discovering the mismatch as a null."""
    with pytest.raises(A.Reject, match="trunk_role_matches_expectation"):
        A.audit(bench["cache"], bench["manifest"], expect_role="stabledet")


def test_reordered_nodes_are_rejected(bench):
    """Positional node ids are only meaningful against a fixed order. This permutes coordinates
    inside one frame, changing nothing about shape, count or feature values."""
    A._synth_cache(bench["cache"] / "44b6_aaaaaaaa.npz", crop="44b6_aaaaaaaa", trunk_seed=1,
                   reorder=True)
    with pytest.raises(A.Reject, match="node_order_unchanged"):
        A.audit(bench["cache"], bench["manifest"])


def test_an_empty_learnable_band_is_rejected(bench):
    """FACT-0382 puts ALL 691 fold-0 contested errors below the deployed 0.5 floor, so a cache
    whose sub-threshold band is empty holds none of the learnable population - and a null on it
    would say nothing about any head."""
    A._synth_cache(bench["cache"] / "44b6_aaaaaaaa.npz", crop="44b6_aaaaaaaa", trunk_seed=1,
                   empty_sub_band=True)
    args = A._bind_args(bench["cache"], bench["trunk"], fold="0", role="official",
                        weights_glob="/kaggle/input/*/split_0/edge_predictor_best.pth",
                        notebook=bench["notebook"])
    bench["manifest"].write_text(json.dumps(A.build_manifest(args)), encoding="utf-8")
    with pytest.raises(A.Reject, match="learnable_band_is_not_empty"):
        A.audit(bench["cache"], bench["manifest"])


def test_wrong_fold_weights_are_rejected(tmp_path):
    """The EXP-0019 defect, reached through the cache rather than through the spec: a fold-1
    cache carrying the pack's split_0 weights, which were trained on the embryo fold 1 holds out."""
    nb = tmp_path / "nb.ipynb"; nb.write_text('{"cells": []}', encoding="utf-8")
    trunk = tmp_path / "t.pth"; trunk.write_bytes(b"T" * 512)
    cache = tmp_path / "f1"; cache.mkdir()
    A._synth_cache(cache / "6bba_bbbbbbbb.npz", crop="6bba_bbbbbbbb", trunk_seed=1)
    args = A._bind_args(cache, trunk, fold="1", role="oof_split1",
                        weights_glob="/kaggle/input/*/split_0/edge_predictor_best.pth",
                        notebook=nb)
    manifest = cache / "cache_manifest.json"
    manifest.write_text(json.dumps(A.build_manifest(args)), encoding="utf-8")
    with pytest.raises(A.Reject, match="fold1_does_not_use_the_leaky_pack_weights"):
        A.audit(cache, manifest)


def test_a_fold_declared_over_the_other_embryo_is_rejected(tmp_path):
    """Report both embryo directions separately (AGENTS.md). A cache whose crops belong to the
    other embryo than its declared fold would silently pool them."""
    nb = tmp_path / "nb.ipynb"; nb.write_text('{"cells": []}', encoding="utf-8")
    trunk = tmp_path / "t.pth"; trunk.write_bytes(b"T" * 512)
    cache = tmp_path / "f1"; cache.mkdir()
    A._synth_cache(cache / "6bba_bbbbbbbb.npz", crop="6bba_bbbbbbbb", trunk_seed=1)
    args = A._bind_args(cache, trunk, fold="1", role="oof_split1",
                        weights_glob="/kaggle/input/*/edge_predictor_best_split_1.pth",
                        notebook=nb)
    m = A.build_manifest(args)
    m["fold"]["fold"] = "0"
    m["fold"]["held_out_embryo"] = "44b6"
    m["fold"]["weights_glob"] = "/kaggle/input/*/split_0/edge_predictor_best.pth"
    manifest = cache / "bad.json"
    manifest.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(A.Reject, match="embryo_matches_fold"):
        A.audit(cache, manifest)


def test_an_unbound_cache_is_rejected(bench):
    """No manifest is not 'no opinion'. An unbound cache cannot distinguish 'the head does not
    work' from 'we fed it the wrong features', which is the whole failure this packet removes."""
    with pytest.raises(A.Reject, match="no manifest"):
        A.audit(bench["cache"], bench["root"] / "absent.json")


def test_a_cache_without_its_frame_partition_is_rejected(tmp_path):
    """Fail closed on an unauditable shape rather than auditing it loosely. Note that
    scripts/win_bet/assoc_feature_cache.py --cache-out currently writes exactly this shape."""
    import numpy as np
    cache = tmp_path / "c"; cache.mkdir()
    np.savez_compressed(cache / "44b6_x.npz", coords=np.zeros((4, 4)),
                        frames=np.arange(2), feat_0=np.zeros((2, 32)))
    with pytest.raises(A.Reject, match="missing"):
        A.load_cache(cache / "44b6_x.npz")


def test_a_cache_without_its_candidate_surface_is_rejected(tmp_path):
    import numpy as np
    cache = tmp_path / "c"; cache.mkdir()
    np.savez_compressed(cache / "44b6_x.npz", coords=np.zeros((4, 4)),
                        frames=np.arange(2), starts=np.array([0, 2]), ends=np.array([2, 4]),
                        feat_frames=np.arange(2), feat_0=np.zeros((2, 32)),
                        feat_1=np.zeros((2, 32)))
    with pytest.raises(A.Reject, match="no candidate surface"):
        A.load_cache(cache / "44b6_x.npz")


def test_the_self_test_command_passes(tmp_path):
    """The CLI's own mutation battery, run as part of the suite so it cannot rot."""
    out = tmp_path / "selftest.json"
    assert A.self_test(out) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["all_passed"] is True
    planted = {r["mutation"] for r in payload["results"] if r["expected"] == "reject"}
    assert {"swapped_trunk_features", "reordered_nodes", "empty_probability_band",
            "wrong_fold_weights_split0_on_fold1"} <= planted


# --- THE DUAL-TRUNK PAIR (FACT-0392 risk three) ----------------------------------------------
# Trunk identity is a --weights CLI argument recorded nowhere in the artifact, so the only way to
# make an HOCT null interpretable is to cache both trunks and compare them IN ONE SESSION. That
# only helps if the pair is well formed, which these pin.


def _sibling(bench, *, seed, role="stabledet", n_per_frame=4, fold="0"):
    other = bench["root"] / f"sib_{seed}_{n_per_frame}_{fold}"
    other.mkdir()
    trunk = bench["root"] / f"trunk_{seed}_{fold}.pth"
    trunk.write_bytes(b"OTHER-TRUNK" * (64 + seed))
    crop = "44b6_aaaaaaaa" if fold == "0" else "6bba_bbbbbbbb"
    A._synth_cache(other / f"{crop}.npz", crop=crop, trunk_seed=seed, n_per_frame=n_per_frame)
    glob = ("/kaggle/input/*/split_0/edge_predictor_best.pth" if fold == "0"
            else "/kaggle/input/*/edge_predictor_best_split_1.pth")
    args = A._bind_args(other, trunk, fold=fold, role=role, weights_glob=glob,
                        notebook=bench["notebook"])
    (other / "cache_manifest.json").write_text(json.dumps(A.build_manifest(args)),
                                               encoding="utf-8")
    return other


def test_a_well_formed_dual_trunk_pair_is_accepted(bench):
    other = _sibling(bench, seed=7)
    report = A.audit_dual_trunk(bench["cache"], other)
    assert report["passed"] is True
    assert set(report["roles"]) == {"official", "stabledet"}


def test_one_trunk_written_twice_is_not_a_pair(bench):
    """Identical features under two checkpoint names: the second trunk was never loaded."""
    other = _sibling(bench, seed=1)
    with pytest.raises(A.Reject, match="dual_trunk_features_differ"):
        A.audit_dual_trunk(bench["cache"], other)


def test_a_pair_whose_trunks_saw_different_node_sets_is_rejected(bench):
    """If the two trunks do not index ONE detector pass, trunk identity is confounded with node
    identity and the comparison the pair exists to enable is unavailable."""
    other = _sibling(bench, seed=7, n_per_frame=5)
    with pytest.raises(A.Reject, match="dual_trunk_shares_the_node_set"):
        A.audit_dual_trunk(bench["cache"], other)


def test_a_pair_across_folds_is_rejected(bench):
    """Comparing trunks across folds reintroduces the embryo confound the contract forbids."""
    other = _sibling(bench, seed=7, fold="1")
    with pytest.raises(A.Reject, match="dual_trunk_same_fold"):
        A.audit_dual_trunk(bench["cache"], other)
