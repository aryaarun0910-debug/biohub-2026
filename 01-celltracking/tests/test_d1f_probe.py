"""Acceptance locks for the D1-F frozen-feature linear probe.

`scripts/d1f_probe.py` was rewritten wholesale and merged to master WITHOUT TESTS. This
module is the missing gate. It is organised as one section per acceptance criterion, and
each section states what would have to be true of the instrument for the corresponding
number to mean anything.

WHY THESE EIGHT. Every one of them names a way the v5 script produced a number that read
as evidence and was not:

  1. H0 must READ the checkpoint head. The v5 script FITTED it, so the parity reference
     was a seventh fitted arm wearing the deployed head's name.
  2. H1-H4 must differ in their OBJECTIVE. The v5 script hardcoded `pi_crop=None` and
     never wrote a temporal branch, so all four were byte-identical and four identical
     rows read as convergent evidence.
  3. Source and target must be encoded by the SAME checkpoint (C1). The two 32-D bases
     are independent (rel-L2 1.41542 ~ sqrt(2), cos(w0,w1) = -0.155); a head fitted in
     one has no meaning in the other.
  4. Family and checkpoint must not be confounded, structurally rather than by
     convention.
  5. `unet_out` is read at L442/L445 by `predict_edges` AFTER the TTA block. It is the
     association representation and must never be mutated.
  6. A manifest is authoritative; a glob is not. A missing crop must raise and be named.
  7. The M/C/T/L/D partition is the SCORER's. Bind to `biotrack.metric.MAX_DISTANCE`,
     never to a literal 7.0, and reuse `scripts/d1_postprocess.py` rather than
     reimplementing it.
  8. Joins must be deterministic and the 32-D features finite: two runs, identical bytes.

Plus the constraints that decide what the instrument may SAY: exactly three permitted
verdicts with `REPRESENTATION DEFICIT` impossible by construction (C2), a label-shuffled
negative control that must return a null, the 1/4096 Horvitz-Thompson intercept
correction of 8.3178 logits, and ranking / calibration / both transfer directions
reported separately and never pooled (C7).

BASIS TAG: FIXTURE. Every corpus here is synthetic. The v6 export does not exist yet and
the v5 3-crop smoke is forbidden as training data. Nothing in this module is evidence
about the real corpus; it is evidence about the instrument.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import d1f_probe as D  # noqa: E402

FACT_DIR = ROOT / "data" / "d1_factorial"

# Small enough to run the whole instrument repeatedly, large enough that the grouped
# 4-fold split and the nested selection are real. The grid and uniform counts keep the
# uniform sampling rate at EXACTLY 1/4096, which is the rate the 8.3178-logit intercept
# correction is quoted at.
SPEC_KW = dict(n_crops_per_family=4, n_frames=25, n_uniform_per_frame=16,
               n_subthr_per_crop=150, gt_min=40, gt_max=120)


# ============================================================ shared fixtures
@pytest.fixture(scope="module")
def heads():
    return D.fixture_heads()


@pytest.fixture(scope="module")
def corpus_latent(heads):
    return D.make_factorial_fixture(heads, D.FixtureSpec(**SPEC_KW))


@pytest.fixture(scope="module")
def corpus(corpus_latent):
    return corpus_latent[0]


@pytest.fixture(scope="module")
def latent(corpus_latent):
    return corpus_latent[1]


@pytest.fixture(scope="module")
def pools(corpus):
    return D.source_fit_pool(corpus, basis_split=1)


@pytest.fixture(scope="module")
def caps(corpus, heads, pools, latent):
    """Full capabilities, so H1-H4 are all REACHABLE. The temporal tables are built
    against the FIT POOL, not the source family: `TemporalTerm` rejects any index outside
    `allowed_rows`, so a table built against `source` silently blocks H3/H4."""
    t = D.make_temporal_tables(corpus, latent, source_mask=pools["fit_pool"],
                              n_pseudo=120, n_pairs=200)
    return D.Capabilities(
        checkpoint_head=heads[1], dist_to_nearest_gt_um=corpus.dist_to_nearest_gt_um,
        mask_radius_um=5.0, pi_crop=corpus.pi_crop(),
        temporal_pseudo_pos_idx=t["pseudo_pos_idx"], temporal_pseudo_pos_w=t["pseudo_pos_w"],
        temporal_pair_idx=t["pair_idx"], temporal_pair_w=t["pair_w"],
        sampling_rate=corpus.sampling_rate())


@pytest.fixture(scope="module")
def built(corpus, caps, pools):
    b, blocked, fps = D.build_arms(corpus, caps, source_mask=pools["fit_pool"],
                                   n_shuffles=3)
    return b, blocked, fps


@pytest.fixture(scope="module")
def payload(corpus, heads):
    """One full 2x2 run, reused by every reporting-shape assertion."""
    return D.run_factorial(corpus, heads, mask_radius_um=5.0)


def _write_checkpoint(path, w, b):
    """A minimal state dict shaped like the real `edge_predictor_best_split_k.pth`."""
    import torch

    torch.save({"detect_head.weight": torch.tensor(
        np.asarray(w, dtype=np.float32).reshape(1, D.FEAT_DIM, 1, 1, 1)),
        "detect_head.bias": torch.tensor(np.asarray([b], dtype=np.float32))}, path)
    return path


# ==================================================================================
# 1. H0 READS THE CHECKPOINT HEAD AND DOES NOT FIT A REPLACEMENT
# ==================================================================================
def test_deployed_threshold_is_0_96875_and_its_logit_is_3_4340():
    assert D.DET_THRESHOLD == 0.96875
    assert D.DET_THRESHOLD_LOGIT == pytest.approx(3.4339872, abs=1e-6)
    assert D.DET_THRESHOLD_LOGIT == pytest.approx(
        math.log(D.DET_THRESHOLD / (1 - D.DET_THRESHOLD)), rel=1e-12)


def test_h0_head_is_read_from_the_checkpoint_state_dict(tmp_path):
    w = np.arange(D.FEAT_DIM, dtype=np.float64) * 0.01 - 0.15
    p = _write_checkpoint(tmp_path / "ckpt.pth", w, -2.5)
    h = D.LinearHead.from_checkpoint(p, split=None)
    assert h.w == pytest.approx(w.astype(np.float32).astype(np.float64), abs=1e-7)
    assert h.b == pytest.approx(-2.5, abs=1e-7)
    assert h.provenance["fitted"] is False
    assert h.provenance["kind"] == "checkpoint_detect_head"


def test_h0_arm_is_never_fitted_and_selects_no_threshold(built):
    b, _blocked, fps = built
    assert b["H0"].fitted is False
    assert b["H0"].notes["threshold_selected"] is False
    # H0 has no objective, so it is exempt from the arm-distinctness fingerprint.
    assert "H0" not in fps


def test_fit_arm_returns_h0_unchanged_and_produces_no_fit_result(built, heads):
    b, _blocked, _fps = built
    head, res, _score = D.fit_arm(b["H0"], basis_split=1)
    assert res is None, "H0 must not have been fitted"
    assert head.param_sha256() == heads[1].param_sha256()
    assert head.w == pytest.approx(heads[1].w)
    assert head.b == pytest.approx(heads[1].b)


def test_h0_in_a_full_run_keeps_the_deployed_threshold(payload):
    for direction in payload["directions"]:
        h0 = direction["arms"]["H0"]
        assert h0["fitted"] is False
        assert h0["threshold_logit"] == pytest.approx(D.DET_THRESHOLD_LOGIT)
        assert h0["selection"]["quantile"] is None
        assert "NOT selected" in h0["selection"]["BASIS"]
        assert h0["fit"] is None


def test_every_other_arm_does_select_a_threshold(payload):
    """The point of criterion 1 is a CONTRAST: if nothing selected a threshold, H0's
    not selecting one would carry no information."""
    for direction in payload["directions"]:
        selected = [r for n, r in direction["arms"].items()
                    if n != "H0" and D.arm_base(n) not in D.SHUFFLED_ARMS]
        assert selected, "no fitted arm survived to contrast against H0"
        for row in selected:
            assert row["selection"]["quantile"] is not None
            assert row["selection"]["refit_per_fold"] is True


def test_checkpoint_sha256_mismatch_is_refused(tmp_path):
    p = _write_checkpoint(tmp_path / "c.pth", np.zeros(D.FEAT_DIM), 0.0)
    with pytest.raises(D.ParityError, match="sha256"):
        D.LinearHead.from_checkpoint(p, verify_sha256="0" * 64)


def test_for_basis_pins_the_split_checkpoint_hash(tmp_path):
    """`for_basis` must not accept just any file named like the split weight."""
    _write_checkpoint(tmp_path / "edge_predictor_best_split_1.pth",
                      np.zeros(D.FEAT_DIM), 0.0)
    with pytest.raises(D.ParityError):
        D.LinearHead.for_basis(1, tmp_path)


def test_checkpoint_without_a_detect_head_is_refused(tmp_path):
    import torch

    p = tmp_path / "no_head.pth"
    torch.save({"encoder.weight": torch.zeros(3)}, p)
    with pytest.raises(D.ParityError, match="detect_head"):
        D.LinearHead.from_checkpoint(p)


def test_detect_head_of_the_wrong_shape_is_refused(tmp_path):
    import torch

    p = tmp_path / "bad_shape.pth"
    torch.save({"detect_head.weight": torch.zeros(1, 16, 1, 1, 1),
                "detect_head.bias": torch.zeros(1)}, p)
    with pytest.raises(D.ParityError, match=r"\(1,32,1,1,1\)"):
        D.LinearHead.from_checkpoint(p)


def test_h0_parity_gate_passes_for_the_true_head(corpus, heads):
    m = corpus.basis_mask(1)
    rep = D.assert_logit_parity(heads[1], corpus.X[m], corpus.logit[m], label="true")
    assert rep["passed"] is True
    assert rep["n_over_tol"] == 0


def test_h0_parity_gate_rejects_a_stale_head(corpus, heads):
    """A head that is not the one that produced the logits must abort the run. There is
    no proceed-with-caveat branch."""
    m = corpus.basis_mask(1)
    wrong = D.LinearHead(w=heads[1].w + 0.05, b=heads[1].b, basis_split=1)
    with pytest.raises(D.ParityError, match="rows exceed atol"):
        D.assert_logit_parity(wrong, corpus.X[m], corpus.logit[m], label="stale")


def test_parity_gate_refuses_an_empty_row_set(heads):
    with pytest.raises(D.ParityError, match="no rows"):
        D.assert_logit_parity(heads[1], np.zeros((0, D.FEAT_DIM)), np.zeros(0))


def test_a_run_reports_h0_parity_for_every_direction(payload):
    for direction in payload["directions"]:
        assert direction["h0_parity"]["passed"] is True
        assert direction["h0_parity"]["n_rows"] > 0


# ==================================================================================
# 2. H1-H4 HAVE DISTINCT CONFIGURATION *AND* OUTPUT HASHES
# ==================================================================================
def test_h1_h2_h3_h4_are_all_reachable(built):
    """If they are blocked, their distinctness is untested rather than proven."""
    b, blocked, _fps = built
    for arm in ("H1", "H2", "H3", "H4"):
        assert arm in b, f"{arm} BLOCKED: {blocked.get(arm)}"


def test_hypothesis_arms_have_pairwise_distinct_objective_fingerprints(built):
    """The acceptance criterion: `(objective, grad_sha256)` pairwise distinct at a FIXED
    probe beta. A distinct label or config hash is not evidence."""
    _b, _blocked, fps = built
    keys = [(round(fps[a]["objective"], 9), fps[a]["grad_sha256"])
            for a in ("H1", "H2", "H3", "H4")]
    assert len(set(keys)) == 4, f"H1-H4 are not pairwise distinct: {keys}"


def test_all_built_arms_are_pairwise_distinct_within_their_parameter_space(built):
    _b, _blocked, fps = built
    keys = [(fps[a]["param_space"], round(fps[a]["objective"], 9), fps[a]["grad_sha256"])
            for a in fps]
    assert len(set(keys)) == len(keys)


def test_each_hypothesis_arm_declares_the_terms_that_make_it_different(built):
    b, _blocked, _fps = built
    assert [t.name for t in b["H1"].terms] == []
    assert [t.name for t in b["H2"].terms] == ["count_ge"]
    assert [t.name for t in b["H3"].terms] == ["temporal"]
    assert [t.name for t in b["H4"].terms] == ["count_ge", "temporal"]


def test_two_arms_with_an_identical_objective_raise(corpus, pools):
    """This is the v5 defect, reproduced deliberately: same objective, two labels."""
    s = np.asarray(pools["fit_pool"], dtype=np.float64)
    mk = lambda nm: D.Arm(name=nm, design=corpus.X, y=corpus.y, s=s, offset=None,  # noqa: E731
                          param_space="full33")
    with pytest.raises(D.MissingCapability, match="IDENTICAL objective"):
        D.assert_arms_distinct({"H1": mk("H1"), "H2": mk("H2")})


def test_a_distinct_name_alone_does_not_make_two_arms_distinct(corpus, pools):
    s = np.asarray(pools["fit_pool"], dtype=np.float64)
    a = D.Arm(name="H3", design=corpus.X, y=corpus.y, s=s, offset=None,
              param_space="full33")
    b = D.Arm(name="H4_TOTALLY_DIFFERENT_LABEL", design=corpus.X, y=corpus.y, s=s,
              offset=None, param_space="full33")
    with pytest.raises(D.MissingCapability):
        D.assert_arms_distinct({"a": a, "b": b})


def test_fitted_heads_that_converge_to_the_same_parameters_raise():
    """Distinct objectives are not enough: two arms may still land on one point, and the
    results table would then carry the same head twice."""
    h = D.LinearHead(w=np.linspace(-1, 1, D.FEAT_DIM), b=0.3, basis_split=1)
    same = D.LinearHead(w=np.linspace(-1, 1, D.FEAT_DIM), b=0.3, basis_split=1)
    with pytest.raises(D.MissingCapability, match="IDENTICAL deployed"):
        D.assert_fitted_heads_distinct({"H1": h, "H2": same})


def test_a_full_run_emits_no_two_arms_with_the_same_deployed_parameters(payload):
    for direction in payload["directions"]:
        prints = [r["fingerprint"]["grad_sha256"] for r in direction["arms"].values()
                  if r["fingerprint"]]
        assert len(set(prints)) == len(prints)


def test_h2_refuses_a_null_count_prior(corpus, pools):
    """`pi_crop=None` is the exact line that made H2 identical to H1."""
    with pytest.raises(D.MissingCapability, match="pi_crop"):
        D.CountGETerm(X=corpus.X, crop_of_row=corpus.crop,
                      uniform_mask=(corpus.kind == "uniform") & pools["fit_pool"],
                      pi_crop=None)


def test_h2_refuses_a_count_prior_outside_zero_one(corpus, pools):
    with pytest.raises(D.MissingCapability, match=r"outside \(0, 1\)"):
        D.CountGETerm(X=corpus.X, crop_of_row=corpus.crop,
                      uniform_mask=(corpus.kind == "uniform") & pools["fit_pool"],
                      pi_crop={c: 12.0 for c in corpus.crops()})


def test_h3_and_h4_are_blocked_rather_than_degraded_without_temporal_tables(corpus, heads,
                                                                            pools):
    caps = D.Capabilities(checkpoint_head=heads[1],
                          dist_to_nearest_gt_um=corpus.dist_to_nearest_gt_um,
                          mask_radius_um=5.0, pi_crop=corpus.pi_crop(),
                          sampling_rate=corpus.sampling_rate())
    b, blocked, _fps = D.build_arms(corpus, caps, source_mask=pools["fit_pool"],
                                    n_shuffles=3)
    assert "H3" not in b and "H4" not in b
    for arm in ("H3", "H4"):
        assert "temporal" in blocked[arm]
    # and the arms that DO NOT need temporal data are unaffected
    assert "H1" in b and "H2" in b


def test_temporal_term_requires_both_tables(corpus):
    with pytest.raises(D.MissingCapability, match="BOTH"):
        D.TemporalTerm(X=corpus.X, pseudo_pos_idx=np.array([0, 1]),
                       pseudo_pos_w=np.array([1.0, 1.0]), pair_idx=None, pair_w=None)


def test_h1_refuses_a_mask_that_zeroes_nothing(corpus, heads, pools):
    """A radius above every distance leaves H1 byte-identical to the base exposure
    profile — a relabelled copy, which is the defect in a new costume."""
    caps = D.Capabilities(checkpoint_head=heads[1],
                          dist_to_nearest_gt_um=corpus.dist_to_nearest_gt_um,
                          mask_radius_um=1e9, pi_crop=corpus.pi_crop(),
                          sampling_rate=corpus.sampling_rate())
    _b, blocked, _fps = D.build_arms(corpus, caps, source_mask=pools["fit_pool"],
                                     arms=("H1",), n_shuffles=3)
    assert "byte-identical" in blocked["H1"]


def test_a_mask_derived_from_kind_alone_is_not_a_row_mask(corpus, heads, pools):
    """`row_mask` requires a real per-row distance. Without one the arm is BLOCKED, not
    silently approximated from the sampler's own strata."""
    caps = D.Capabilities(checkpoint_head=heads[1], dist_to_nearest_gt_um=None,
                          mask_radius_um=5.0, pi_crop=corpus.pi_crop(),
                          sampling_rate=corpus.sampling_rate())
    assert caps.have("row_mask") is False
    _b, blocked, _fps = D.build_arms(corpus, caps, source_mask=pools["fit_pool"],
                                     arms=("H1",), n_shuffles=3)
    assert "row_mask" in blocked["H1"]


def test_count_and_temporal_terms_are_active_and_psd_at_the_probe_point(built):
    b, _blocked, _fps = built
    probe = D.probe_beta(D.N_HEAD_PARAMS)
    for arm, term in (("H2", "count_ge"), ("H3", "temporal")):
        st = [t for t in b[arm].terms if t.name == term][0].self_test(probe)
        assert st["active"] is True
        assert st["hess_min_eig"] >= -1e-8


# ==================================================================================
# 3. SOURCE AND TARGET USE THE SAME CHECKPOINT WHEN TESTING REPRESENTATION (C1)
# ==================================================================================
def test_source_and_target_rows_share_one_checkpoint_basis(corpus, pools):
    both = pools["source"] | pools["target"]
    assert set(np.unique(corpus.encoder_split[both]).tolist()) == {1}


def test_source_is_the_family_the_checkpoint_trained_on(pools):
    """Verified role semantics: SOURCE = the checkpoint's own training family, TARGET =
    the family it never saw. split_k HELD OUT family k."""
    assert D.SPLIT_HELDOUT_FAMILY == {0: "44b6", 1: "6bba"}
    assert D.SPLIT_SOURCE_FAMILY == {0: "6bba", 1: "44b6"}
    assert pools["source_family"] == "44b6"
    assert pools["target_family"] == "6bba"


def test_a_head_from_one_basis_may_not_touch_the_other_basis(corpus, heads):
    with pytest.raises(D.BasisError, match="independent"):
        D.assert_same_basis(heads[0], corpus, corpus.basis_mask(1), label="cross")


def test_a_head_with_no_basis_tag_is_refused(corpus):
    untagged = D.LinearHead(w=np.ones(D.FEAT_DIM), b=0.0, basis_split=None)
    with pytest.raises(D.BasisError, match="no basis tag"):
        D.assert_same_basis(untagged, corpus, corpus.basis_mask(1))


def test_run_direction_refuses_a_head_from_the_wrong_split(corpus, heads):
    with pytest.raises(D.BasisError, match="split"):
        D.run_direction(corpus, basis_split=1, h0=heads[0])


def test_a_routed_only_export_is_refused_and_says_why(corpus, heads):
    """In a routed-only export each family sits in its own basis, so no basis carries
    both. That cannot support the 2x2 and must raise rather than return a number."""
    routed = corpus.subset(
        ((corpus.encoder_split == 0) & (corpus.family == "44b6"))
        | ((corpus.encoder_split == 1) & (corpus.family == "6bba")))
    with pytest.raises(D.BasisError, match="CROSS-ENCODED"):
        D.run_direction(routed, basis_split=1, h0=heads[1])


def test_probe_routing_constants_match_the_merged_factorial_manifest():
    """C1: consume the manifests at `data/d1_factorial/`; do not invent crop routing."""
    idx = FACT_DIR / "manifest_index.json"
    if not idx.exists():                       # pragma: no cover - census not built
        pytest.skip("data/d1_factorial manifests absent")
    seen = 0
    for tier in ("smoke", "pilot", "full"):
        p = FACT_DIR / f"manifest_{tier}.json"
        if not p.exists():
            continue
        for cell in json.loads(p.read_text(encoding="utf-8"))["cells"].values():
            fold = int(cell["fold"])
            assert cell["checkpoint_heldout_family"] == D.SPLIT_HELDOUT_FAMILY[fold]
            assert cell["checkpoint_trained_on_family"] == D.SPLIT_SOURCE_FAMILY[fold]
            assert cell["checkpoint_sha256"] == D.CHECKPOINT_SHA256[fold]
            expect = "source" if cell["family"] == D.SPLIT_SOURCE_FAMILY[fold] else "target"
            assert cell["role"] == expect
            seen += 1
    assert seen >= 4, "no 2x2 cells were checked"


def test_the_manifest_and_the_probe_agree_that_a_target_never_saw_its_checkpoint():
    p = FACT_DIR / "manifest_smoke.json"
    if not p.exists():                         # pragma: no cover
        pytest.skip("data/d1_factorial manifests absent")
    for cell in json.loads(p.read_text(encoding="utf-8"))["cells"].values():
        if cell["role"] == "target":
            assert cell["encoder_saw_this_family_in_training"] is False
            assert cell["fit_and_select_here"] is False


# ==================================================================================
# 4. NO FAMILY / CHECKPOINT CONFOUNDING — ASSERTED STRUCTURALLY
# ==================================================================================
def test_every_checkpoint_basis_carries_both_families(corpus):
    for split in corpus.bases():
        fams = set(np.unique(corpus.family[corpus.basis_mask(split)]).tolist())
        assert fams == {"44b6", "6bba"}, (
            f"basis {split} carries {fams}; family and checkpoint are confounded")


def test_every_family_is_carried_by_both_checkpoints(corpus):
    for fam in ("44b6", "6bba"):
        splits = set(np.unique(corpus.encoder_split[corpus.family_mask(fam)]).tolist())
        assert splits == {0, 1}


def test_target_family_rows_carry_exactly_zero_fitting_weight(built, pools):
    b, _blocked, _fps = built
    tgt = pools["target"]
    for name, arm in b.items():
        assert float(np.abs(arm.s[tgt]).max()) == 0.0, f"{name} fits on target rows"


def test_same_family_heldout_crops_are_excluded_from_the_fit_pool(corpus, pools):
    held = pools["same_family_held_crops"]
    assert held, "no same-family hold-out was formed"
    assert not (pools["fit_pool"] & np.isin(corpus.crop, held)).any()
    assert (pools["same_family_held"]).any()


def test_temporal_pseudo_labels_outside_the_fit_pool_are_refused(corpus, pools, latent):
    """A fold-dishonest path must be impossible, not merely unused."""
    bad = D.make_temporal_tables(corpus, latent, source_mask=pools["target"],
                                 n_pseudo=40, n_pairs=60)
    with pytest.raises(D.MissingCapability, match="outside the source-family rows"):
        D.TemporalTerm(X=corpus.X, pseudo_pos_idx=bad["pseudo_pos_idx"],
                       pseudo_pos_w=bad["pseudo_pos_w"], pair_idx=bad["pair_idx"],
                       pair_w=bad["pair_w"], allowed_rows=pools["fit_pool"])


def test_the_count_term_constrains_only_source_uniform_rows(built, pools):
    b, _blocked, _fps = built
    term = [t for t in b["H2"].terms if t.name == "count_ge"][0]
    idx = np.concatenate([i for i, _pi in term._groups.values()])
    assert pools["fit_pool"][idx].all()


def test_both_transfer_directions_use_different_checkpoints_and_families(payload):
    dirs = payload["directions"]
    assert len(dirs) == 2
    assert {d["checkpoint_basis_split"] for d in dirs} == {0, 1}
    assert {d["source_family"] for d in dirs} == {"44b6", "6bba"}
    for d in dirs:
        assert d["source_family"] != d["target_family"]


# ==================================================================================
# 5. NO MUTATION OF `unet_out` AS USED BY `predict_edges`
# ==================================================================================
def test_the_probe_never_names_unet_out_at_all():
    """`unet_out` is read at L442/L445 by `predict_edges`, AFTER the TTA block. This
    module is a numpy probe over an exported row table and has no business touching it;
    the safest guarantee is that the identifier does not appear."""
    src = (ROOT / "scripts" / "d1f_probe.py").read_text(encoding="utf-8")
    assert "unet_out" not in src


def test_running_the_whole_instrument_does_not_alter_the_feature_bytes(corpus, heads):
    """The exported representation is an input. If any stage wrote through a view, a
    later stage would be reading something the detector never produced."""
    before = (hashlib.sha256(corpus.X.tobytes()).hexdigest(),
              hashlib.sha256(corpus.X_max.tobytes()).hexdigest(),
              hashlib.sha256(corpus.logit.tobytes()).hexdigest())
    D.run_direction(corpus, basis_split=1, h0=heads[1], mask_radius_um=5.0,
                    arms=("H0", "CAL_ONLY", "LIN_HEAD", "H1"))
    after = (hashlib.sha256(corpus.X.tobytes()).hexdigest(),
             hashlib.sha256(corpus.X_max.tobytes()).hexdigest(),
             hashlib.sha256(corpus.logit.tobytes()).hexdigest())
    assert before == after


def test_scoring_does_not_mutate_the_feature_block(corpus, heads):
    X = corpus.X[:500].copy()
    sig = hashlib.sha256(X.tobytes()).hexdigest()
    heads[1].logit(X, dtype="float32")
    heads[1].logit(X, dtype="float64")
    assert hashlib.sha256(X.tobytes()).hexdigest() == sig


@pytest.mark.parametrize("fn", ["local_mask_weights", "exposure_balance",
                                "design_weights", "ht_weights"])
def test_weight_builders_do_not_mutate_their_inputs(corpus, fn):
    base = D.kind_weights(corpus.kind, uniform_weight=0.01)
    sig = hashlib.sha256(base.tobytes()).hexdigest()
    rate = corpus.sampling_rate()
    if fn == "local_mask_weights":
        D.local_mask_weights(base, corpus.dist_to_nearest_gt_um, radius_um=5.0,
                             is_positive=corpus.is_positive)
    elif fn == "exposure_balance":
        D.exposure_balance(base, corpus.y, groups=corpus.crop)
    elif fn == "design_weights":
        D.design_weights(base, corpus.kind, corpus.crop, rate)
    else:
        D.ht_weights(corpus.kind, corpus.crop, rate)
    assert hashlib.sha256(base.tobytes()).hexdigest() == sig


def test_subset_returns_a_new_corpus_and_leaves_the_original_intact(corpus):
    sig = hashlib.sha256(corpus.X.tobytes()).hexdigest()
    sub = corpus.subset(corpus.basis_mask(1))
    assert sub.n < corpus.n
    assert hashlib.sha256(corpus.X.tobytes()).hexdigest() == sig


# ==================================================================================
# 6. COMPLETE CROP AGGREGATION — A MANIFEST IS AUTHORITATIVE, A GLOB IS NOT
# ==================================================================================
def _write_audit_dir(tmp_path, crops=("44b6_aa", "44b6_bb"), *, split=1, n=40,
                     drop_features=(), extra_columns=None, statuses=None,
                     stray_crop=None, n_views=8):
    """A minimal but real v6 audit directory: parquet rows + two feature blocks + a
    manifest. Everything the loader is supposed to check is expressible here."""
    import polars as pl

    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    manifest = {"tta_view_set": "planar8", "n_views": n_views, "crops": {}}
    for c in crops:
        cols = {"dataset": [c] * n, "kind": ["gt_centre"] * n,
                "t": list(range(n)), "logit": [0.0] * n}
        if extra_columns and c == crops[0]:
            cols.update({k: [0.0] * n for k in extra_columns})
        pl.DataFrame(cols).write_parquet(tmp_path / f"{c}__rows.parquet")
        for suffix in D.V6_REQUIRED_FEATURE_FILES:
            if (c, suffix) in drop_features:
                continue
            np.save(tmp_path / f"{c}{suffix}",
                    rng.normal(size=(n, D.FEAT_DIM)).astype(np.float32))
        manifest["crops"][c] = {
            "status": (statuses or {}).get(c, "complete"), "grid_zyx": [16, 64, 64],
            "n_frames": n, "n_uniform_per_frame": 16, "estimated_number_of_nodes": 100,
            "checkpoint_sha256": "x" * 64, "split": split, "encoder_split": split}
    if stray_crop:                            # on disk but NOT in the manifest
        np.save(tmp_path / f"{stray_crop}{D.V6_REQUIRED_FEATURE_FILES[0]}",
                rng.normal(size=(n, D.FEAT_DIM)).astype(np.float32))
    (tmp_path / "d1_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_a_crop_named_in_the_manifest_with_no_features_raises_and_is_named(tmp_path):
    d = _write_audit_dir(tmp_path / "a",
                         drop_features=(("44b6_bb", "__feat_tta_mean_max.npy"),))
    with pytest.raises(D.ContractError) as exc:
        D.load_basis(d)
    assert "44b6_bb" in str(exc.value)


def test_a_stray_feature_file_not_in_the_manifest_is_ignored(tmp_path):
    """A glob would pick it up and silently widen the corpus. The manifest decides."""
    d = _write_audit_dir(tmp_path / "b", stray_crop="6bba_ghost")
    c = D.load_basis(d)
    assert "6bba_ghost" not in c.crops()
    assert c.crops() == ["44b6_aa", "44b6_bb"]


def test_incomplete_crops_are_dropped_and_an_all_incomplete_export_raises(tmp_path):
    d = _write_audit_dir(tmp_path / "c", statuses={"44b6_bb": "partial"})
    assert D.load_basis(d).crops() == ["44b6_aa"]
    d2 = _write_audit_dir(tmp_path / "d",
                          statuses={"44b6_aa": "partial", "44b6_bb": "partial"})
    with pytest.raises(D.ContractError, match="partial export"):
        D.load_basis(d2)


def test_a_schema_difference_between_crops_raises(tmp_path):
    """Trap 21: a column present in only some crops is silently dropped by a vertical
    concat, so it must be caught per crop, before the concat."""
    d = _write_audit_dir(tmp_path / "e", extra_columns=["only_here"])
    with pytest.raises(D.ContractError, match="Trap 21"):
        D.load_basis(d)


@pytest.mark.parametrize("col", D.V5_ABSENT_COLUMNS)
def test_a_v5_column_that_the_export_does_not_emit_raises_if_present(tmp_path, col):
    """Their PRESENCE is as much a contract violation as their absence: it means
    something other than the v6 export is being read."""
    d = _write_audit_dir(tmp_path / f"f{col}", crops=("44b6_aa",), extra_columns=[col])
    with pytest.raises(D.ContractError, match=col):
        D.load_basis(d)


def test_a_manifest_missing_top_level_keys_raises(tmp_path):
    d = _write_audit_dir(tmp_path / "g")
    m = json.loads((d / "d1_manifest.json").read_text(encoding="utf-8"))
    del m["n_views"]
    (d / "d1_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(D.ContractError, match="n_views"):
        D.load_basis(d)


def test_a_manifest_missing_per_crop_keys_raises_and_names_the_crop(tmp_path):
    d = _write_audit_dir(tmp_path / "h")
    m = json.loads((d / "d1_manifest.json").read_text(encoding="utf-8"))
    del m["crops"]["44b6_bb"]["grid_zyx"]
    (d / "d1_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(D.ContractError) as exc:
        D.load_basis(d)
    assert "44b6_bb" in str(exc.value) and "grid_zyx" in str(exc.value)


def test_a_view_set_mismatch_raises(tmp_path):
    """A TTA-mean feature averaged over a different view set than the logits is not the
    detector's representation."""
    d = _write_audit_dir(tmp_path / "i", n_views=4)
    with pytest.raises(D.ContractError, match="n_views"):
        D.load_basis(d, expect_views=8)


def test_a_missing_manifest_raises(tmp_path):
    d = _write_audit_dir(tmp_path / "j")
    (d / "d1_manifest.json").unlink()
    with pytest.raises(D.ContractError, match="no d1_manifest.json"):
        D.load_basis(d)


def test_the_factorial_loader_merges_every_basis_directory(tmp_path):
    root = tmp_path / "fact"
    _write_audit_dir(root / "basis_0", crops=("44b6_aa", "6bba_cc"), split=0)
    _write_audit_dir(root / "basis_1", crops=("44b6_aa", "6bba_cc"), split=1)
    c = D.load_factorial(root)
    assert c.bases() == [0, 1]
    assert set(c.crops()) == {"44b6_aa", "6bba_cc"}
    assert c.n == 4 * 40


def test_a_manifest_crop_count_is_reported_not_inferred(tmp_path):
    d = _write_audit_dir(tmp_path / "k", crops=("44b6_aa", "44b6_bb", "6bba_cc"))
    m = json.loads((d / "d1_manifest.json").read_text(encoding="utf-8"))
    assert D.validate_manifest(m)["n_crops"] == 3


# ==================================================================================
# 7. SCORER-EXACT M/C/T/L/D PARTITION — BOUND, NEVER A LITERAL 7.0
# ==================================================================================
def test_the_partition_radius_is_bound_to_the_scorers_max_distance():
    from biotrack.metric import MAX_DISTANCE

    import d1_postprocess as P

    assert P.MATCH_UM == float(MAX_DISTANCE)
    assert P.SEARCH_UM > P.MATCH_UM


def test_every_component_agrees_on_the_match_radius():
    """`assert_radius_binding` is the reference check and it must actually pass, not just
    exist."""
    import d1_postprocess as P

    P.assert_radius_binding()


def test_the_probe_does_not_reimplement_the_partition():
    """C6 / criterion 7: `scripts/d1_postprocess.py` is the merged reference
    implementation with 58 tests. A second copy inside the probe would drift."""
    src = (ROOT / "scripts" / "d1f_probe.py").read_text(encoding="utf-8")
    for token in ("MATCH_UM", "SEARCH_UM", "def classify"):
        assert token not in src, f"the probe appears to reimplement the partition: {token}"
    # `d_stratum` may appear ONLY in the list of columns the export must not emit.
    for line in src.splitlines():
        if "d_stratum" in line:
            assert "V5_ABSENT_COLUMNS" in line or "d1_class" in line, line


def test_the_probe_contains_no_bare_match_radius_literal():
    """A literal 7.0 anywhere in the probe would be a second, unbound copy of the
    scorer's match radius."""
    src = (ROOT / "scripts" / "d1f_probe.py").read_text(encoding="utf-8")
    assert "7.0" not in src


def test_the_probes_mask_radius_is_a_supplied_parameter_with_no_default(corpus, heads,
                                                                        pools):
    """The Linajea local-mask radius is NOT the scorer's match radius. It has no default,
    so the probe can never quietly adopt 7.0 as if it were bound to anything."""
    caps = D.Capabilities(checkpoint_head=heads[1],
                          dist_to_nearest_gt_um=corpus.dist_to_nearest_gt_um,
                          mask_radius_um=None, pi_crop=corpus.pi_crop(),
                          sampling_rate=corpus.sampling_rate())
    assert caps.have("row_mask") is False
    _b, blocked, _fps = D.build_arms(corpus, caps, source_mask=pools["fit_pool"],
                                     arms=("H1",), n_shuffles=3)
    assert "row_mask" in blocked["H1"]


# ==================================================================================
# 8. DETERMINISTIC JOINS AND FINITE 32-D FEATURES
# ==================================================================================
def test_the_feature_dimension_is_32_and_the_head_has_33_parameters():
    assert D.FEAT_DIM == 32
    assert D.N_HEAD_PARAMS == 33
    h = D.LinearHead(w=np.zeros(32), b=0.0, basis_split=1)
    assert h.beta().shape == (33,)


def test_the_fixture_corpus_is_finite_and_32_wide(corpus):
    assert corpus.X.shape[1] == D.FEAT_DIM
    assert corpus.X_max.shape[1] == D.FEAT_DIM
    assert np.isfinite(corpus.X).all()
    assert np.isfinite(corpus.X_max).all()
    assert np.isfinite(corpus.logit).all()


def test_a_non_finite_feature_block_is_refused(corpus):
    X = corpus.X.copy()
    X[3, 5] = np.nan
    with pytest.raises(D.ContractError, match="non-finite"):
        corpus.subset(np.ones(corpus.n, dtype=bool)).__class__(
            X=X, X_max=corpus.X_max, kind=corpus.kind, crop=corpus.crop,
            family=corpus.family, t=corpus.t, logit=corpus.logit,
            encoder_split=corpus.encoder_split)


def test_a_head_of_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match=r"\(32,\)"):
        D.LinearHead(w=np.zeros(16), b=0.0)
    with pytest.raises(ValueError, match=r"\(33,\)"):
        D.LinearHead.from_beta(np.zeros(20))


def test_a_non_finite_head_is_refused():
    with pytest.raises(ValueError, match="non-finite"):
        D.LinearHead(w=np.full(32, np.nan), b=0.0)


def test_scoring_a_feature_block_of_the_wrong_width_is_refused(heads):
    with pytest.raises(ValueError, match="expected"):
        heads[1].logit(np.zeros((10, 8)))


def test_an_unknown_row_kind_is_refused(corpus):
    kind = corpus.kind.copy()
    kind[0] = "mystery"
    with pytest.raises(D.ContractError, match="unknown row kinds"):
        D.Corpus(X=corpus.X, X_max=corpus.X_max, kind=kind, crop=corpus.crop,
                 family=corpus.family, t=corpus.t, logit=corpus.logit,
                 encoder_split=corpus.encoder_split)


def test_ragged_columns_are_refused(corpus):
    with pytest.raises(D.ContractError, match="rows"):
        D.Corpus(X=corpus.X, X_max=corpus.X_max, kind=corpus.kind[:-1],
                 crop=corpus.crop, family=corpus.family, t=corpus.t,
                 logit=corpus.logit, encoder_split=corpus.encoder_split)


def test_two_runs_of_a_direction_are_bit_identical(corpus, heads):
    kw = dict(basis_split=1, h0=heads[1], mask_radius_um=5.0,
              arms=("H0", "CAL_ONLY", "LIN_HEAD", "H1"))
    a = D.run_direction(corpus, **kw)
    b = D.run_direction(corpus, **kw)
    sa = json.dumps(a, sort_keys=True, default=str)
    sb = json.dumps(b, sort_keys=True, default=str)
    assert hashlib.sha256(sa.encode()).hexdigest() == hashlib.sha256(sb.encode()).hexdigest()


def test_the_fitter_is_deterministic(corpus, pools):
    s = D.kind_weights(corpus.kind, uniform_weight=0.01) * pools["fit_pool"]
    r1 = D.fit_head(corpus.X, corpus.y, s, spec=D.FitSpec(l2=1.0))
    r2 = D.fit_head(corpus.X, corpus.y, s, spec=D.FitSpec(l2=1.0))
    assert r1.beta.tobytes() == r2.beta.tobytes()


def test_grouped_folds_are_deterministic_and_partition_the_crops():
    crops = [f"44b6_{i:03d}" for i in range(11)]
    a = D.grouped_folds(crops, n_folds=4, seed=7)
    b = D.grouped_folds(list(reversed(crops)), n_folds=4, seed=7)
    assert a == b, "fold assignment depends on input order"
    assert set().union(*a) == set(crops)
    assert sum(len(f) for f in a) == len(crops)


def test_the_fixture_corpus_is_reproducible(heads):
    c1, _ = D.make_factorial_fixture(heads, D.FixtureSpec(**SPEC_KW))
    c2, _ = D.make_factorial_fixture(heads, D.FixtureSpec(**SPEC_KW))
    assert c1.X.tobytes() == c2.X.tobytes()
    assert c1.logit.tobytes() == c2.logit.tobytes()


# ==================================================================================
# PERMITTED VERDICTS — `REPRESENTATION DEFICIT` IS IMPOSSIBLE (C2)
# ==================================================================================
def test_permitted_verdicts_are_exactly_four():
    assert D.PERMITTED_VERDICTS == ("CALIBRATION_GLOBAL", "CALIBRATION_PER_CROP",
                                    "LINEAR_HEAD", "LINEAR_PROBE_NULL")


@pytest.mark.parametrize("verdict", [
    "REPRESENTATION DEFICIT", "REPRESENTATION_DEFICIT", "representation deficit",
    "DEFICIT", "ENCODER RETRAIN", "RETRAIN THE BACKBONE", "SUBSTRATE FAILURE",
])
def test_a_representation_or_encoder_verdict_is_refused(verdict):
    with pytest.raises(D.VerdictError, match="forbidden token"):
        D._assert_permitted_verdict(verdict)


@pytest.mark.parametrize("verdict", ["CALIBRATION_GLOBAL", "CALIBRATION_PER_CROP",
                                     "LINEAR_HEAD", "LINEAR_PROBE_NULL"])
def test_the_permitted_verdicts_pass(verdict):
    assert D._assert_permitted_verdict(verdict) == verdict


def test_an_unlisted_verdict_is_refused_even_without_a_forbidden_token():
    with pytest.raises(D.VerdictError, match="not one of"):
        D._assert_permitted_verdict("PROMOTE")


def test_the_module_mentions_representation_deficit_only_as_a_prohibition():
    """Audit item: the phrase must appear only where it is being FORBIDDEN."""
    src = (ROOT / "scripts" / "d1f_probe.py").read_text(encoding="utf-8")
    lines = [ln for ln in src.splitlines() if "REPRESENTATION DEFICIT" in ln.upper()]
    assert lines, "the prohibition itself has gone missing"
    for ln in lines:
        assert ("not" in ln.lower() or "cannot" in ln.lower()
                or "NOT_PERMITTED" in ln), ln


def test_a_planted_forbidden_verdict_is_caught_anywhere_in_the_payload():
    with pytest.raises(D.VerdictError):
        D.assert_payload_clean({"a": {"b": [{"verdict": "REPRESENTATION DEFICIT"}]}})


def test_a_real_payload_passes_the_final_guard(payload):
    D.assert_payload_clean(payload)
    for v in payload["verdict_by_direction"].values():
        assert v in D.PERMITTED_VERDICTS


@pytest.mark.parametrize("regime,expected", [
    ("calibration_global", "CALIBRATION_GLOBAL"),
    ("calibration_per_crop", "CALIBRATION_PER_CROP"),
    ("linear_head", "LINEAR_HEAD"),
    ("null", "LINEAR_PROBE_NULL"),
])
def test_the_instrument_returns_the_verdict_each_regime_was_built_to_produce(regime,
                                                                             expected):
    """The instrument must DISCRIMINATE. Three arms all returning one verdict would be
    an instrument that says the same thing whatever it is shown."""
    hd = D.fixture_heads()
    spec = D.FixtureSpec(**{**SPEC_KW, **D.FIXTURE_REGIMES[regime]})
    c, _ = D.make_factorial_fixture(hd, spec)
    got = D.run_factorial(c, hd)["verdict_by_direction"]
    assert set(got.values()) == {expected}, got


# ==================================================================================
# ACCEPTANCE IS A LEVEL SET — WHY THE CALIBRATION TOKEN SPLITS IN TWO
# ==================================================================================
def _accept(score, tau, neighbourhood=3):
    """The deployed acceptance rule, in miniature: `logit == maxpool(logit)` on the RAW
    logits, AND `sigmoid(logit) > tau`."""
    score = np.asarray(score, dtype=np.float64)
    k = neighbourhood // 2
    ismax = np.array([score[i] == score[max(0, i - k):i + k + 1].max()
                      for i in range(score.size)])
    return ismax & (_sig(score) > tau)


@pytest.mark.parametrize("phi,phi_tau", [
    (lambda s: 2.5 * s, lambda t: 2.5 * t),                       # temperature
    (lambda s: 0.7 * s - 1.3, lambda t: 0.7 * t - 1.3),           # Platt
    (lambda s: np.cbrt(s), lambda t: np.cbrt(t)),                 # any monotone map
])
def test_any_strictly_increasing_calibrator_leaves_the_accepted_set_identical(phi, phi_tau):
    """`A(phi(s), phi(tau)) == A(s, tau)` EXACTLY. This is why temperature, Platt, beta,
    histogram binning and isotonic are all worth exactly ONE scalar on this detector, and
    why a bare `CALIBRATION` verdict buys only a threshold sweep."""
    rng = np.random.default_rng(0)
    s = rng.normal(0.0, 3.0, size=400)
    tau = 1.1
    base = _accept(s, _sig(tau))
    moved = _accept(phi(s), _sig(phi_tau(tau)))
    assert np.array_equal(base, moved)


def test_the_bare_calibration_token_is_retired_and_says_why():
    assert "CALIBRATION" not in D.PERMITTED_VERDICTS
    with pytest.raises(D.VerdictError, match="LEVEL SET"):
        D._assert_permitted_verdict("CALIBRATION")


def test_the_two_calibration_tokens_are_separately_reachable():
    """If only one were reachable the split would be decoration."""
    ranks = [{"auc": 0.5 + 0.001 * i} for i in range(-9, 10)]
    cals = [{"logloss_nats": 0.30}] * 19
    kw = dict(h0_rank={"auc": 0.9}, lin_rank={"auc": 0.90}, shuf_lin_ranks=ranks,
              h0_cal={"logloss_nats": 0.50}, cal_only_cal={"logloss_nats": 0.10},
              shuf_cal_cals=cals)
    assert D.decide_verdict(**kw)["verdict"] == "CALIBRATION_GLOBAL"
    assert D.decide_verdict(**kw, cal_heterogeneity={"heterogeneous": True},
                            )["verdict"] == "CALIBRATION_PER_CROP"


def test_per_crop_calibration_is_reachable_without_any_pooled_log_loss_gain():
    """One constant can be no better than the deployed one corpus-wide and STILL be
    unable to serve every crop. A pooled log-loss gain cannot express that."""
    ranks = [{"auc": 0.5 + 0.001 * i} for i in range(-9, 10)]
    cals = [{"logloss_nats": 0.30}] * 19
    out = D.decide_verdict(
        h0_rank={"auc": 0.9}, lin_rank={"auc": 0.90}, shuf_lin_ranks=ranks,
        h0_cal={"logloss_nats": 0.50}, cal_only_cal={"logloss_nats": 0.50},
        shuf_cal_cals=cals, cal_heterogeneity={"heterogeneous": True})
    assert out["calibration_gain_nats"] == pytest.approx(0.0)
    assert out["verdict"] == "CALIBRATION_PER_CROP"


def test_the_heterogeneity_test_is_null_when_one_constant_serves_every_crop():
    rng = np.random.default_rng(1)
    n = 300
    crops = np.array([f"c{i % 5}" for i in range(n)])
    y = (np.arange(n) % 3 == 0).astype(float)
    score = rng.normal(0.0, 1.0, size=n) + 2.0 * y
    out = D.calibration_heterogeneity(score, y, np.ones(n), crops)
    assert out["heterogeneous"] is False
    assert out["p_value"] > 0.05


def test_the_heterogeneity_test_fires_when_each_crop_needs_its_own_constant():
    rng = np.random.default_rng(1)
    n = 900
    crops = np.array([f"c{i % 5}" for i in range(n)])
    y = (np.arange(n) % 3 == 0).astype(float)
    offset = np.array([{"c0": -3.0, "c1": -1.5, "c2": 0.0, "c3": 1.5,
                        "c4": 3.0}[c] for c in crops])
    score = rng.normal(0.0, 1.0, size=n) + 2.0 * y + offset
    out = D.calibration_heterogeneity(score, y, np.ones(n), crops)
    assert out["heterogeneous"] is True
    assert out["p_value"] < 1e-6
    assert out["per_crop_spread_logits"] > D.MIN_PER_CROP_SPREAD_LOGITS


def test_the_heterogeneity_test_needs_enough_crops_to_have_a_denominator():
    rng = np.random.default_rng(2)
    n = 120
    crops = np.array([f"c{i % 2}" for i in range(n)])
    y = (np.arange(n) % 3 == 0).astype(float)
    score = rng.normal(size=n) + 2 * y + np.where(crops == "c0", -4.0, 4.0)
    out = D.calibration_heterogeneity(score, y, np.ones(n), crops, min_crops=3)
    assert out["enough_crops"] is False
    assert out["heterogeneous"] is False


def test_the_heterogeneity_result_is_explicitly_not_a_router(payload):
    for direction in payload["directions"]:
        het = direction["calibration_degrees_of_freedom"]
        assert het is not None
        assert "does NOT license crop or family identity as a deployment router" in \
            het["NOT_A_ROUTER"]
        assert het["promotes"] is False


def test_the_heterogeneity_statistic_rests_on_the_exhaustive_positive_census(payload):
    """Not on the HT-weighted intercept: at weight 4096 per uniform row the sandwich se is
    ~15 logits per crop and the verdict would be unreachable by construction."""
    for direction in payload["directions"]:
        het = direction["calibration_degrees_of_freedom"]
        assert "ANNOTATED CENTRES" in het["statistic"]
        assert "no HT weight" in het["statistic"]
        for entry in het["per_crop"].values():
            if "mean_positive_score" in entry:
                assert entry["n_positive_rows"] >= 2


def test_a_prior_shift_arm_is_refused_with_its_reason(corpus, caps, pools):
    """SLD/BBSE assume LABEL shift; the gap here is CONDITIONAL shift, and the correction
    comes out wrong-signed on the family that matters most."""
    for name in ("PRIOR_SHIFT", "SLD", "BBSE"):
        _b, blocked, _fps = D.build_arms(corpus, caps, source_mask=pools["fit_pool"],
                                         arms=(name,), n_shuffles=3)
        assert name in blocked
        assert blocked[name].startswith("REFUSED")
    assert "0.808 logits" in D.REFUSED_ARMS["PRIOR_SHIFT"]
    assert "9.7x the miss rate" in D.REFUSED_ARMS["PRIOR_SHIFT"]
    assert "CONDITIONAL shift" in D.REFUSED_ARMS["PRIOR_SHIFT"]


def test_a_refused_arm_never_reaches_the_results_table(corpus, heads):
    d = D.run_direction(corpus, basis_split=1, h0=heads[1],
                        arms=("H0", "CAL_ONLY", "LIN_HEAD", "PRIOR_SHIFT"))
    assert "PRIOR_SHIFT" not in d["arms"]
    assert "PRIOR_SHIFT" in d["blocked"]


def test_the_re_acceptance_head_is_never_called_calibration():
    """M1 is a re-DIRECTION with 32 degrees of freedom: it escapes the level-set theorem
    entirely. Conflating a 32-DOF rotation with a 1-DOF monotone rescale is exactly the
    error the verdict split exists to prevent."""
    src = (ROOT / "scripts" / "d1f_probe.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        low = line.lower()
        if "m1" in low.split() or "re-acceptance" in low:
            assert "not calibration" in low or "32 degrees of freedom" in low, line
    assert "M1 IS NOT CALIBRATION" in src


# ==================================================================================
# CONTROLS THAT MAKE THE INSTRUMENT TRUSTWORTHY
# ==================================================================================
def test_calibration_only_freezes_w_at_the_checkpoint_weight(built, heads):
    """`a*eta0 + c` is a 2-parameter space. The deployed head is `a*w_ckpt`, so its
    weight must stay exactly colinear with the checkpoint's."""
    b, _blocked, _fps = built
    assert b["CAL_ONLY"].param_space == "cal2"
    assert b["CAL_ONLY"].design.shape[1] == 1
    assert b["CAL_ONLY"].notes["n_free_params"] == 2
    b["CAL_ONLY"].notes["_h0"] = heads[1]
    head, res, _score = D.fit_arm(b["CAL_ONLY"], basis_split=1)
    assert res.beta.shape == (2,)
    cos = float(head.w @ heads[1].w / (np.linalg.norm(head.w) * np.linalg.norm(heads[1].w)))
    assert abs(abs(cos) - 1.0) < 1e-9, "calibration changed the weight DIRECTION"


def test_linear_head_refits_all_thirty_three_parameters(built):
    b, _blocked, _fps = built
    assert b["LIN_HEAD"].param_space == "full33"
    assert b["LIN_HEAD"].design.shape[1] == D.FEAT_DIM
    head, res, _score = D.fit_arm(b["LIN_HEAD"], basis_split=1)
    assert res.beta.shape == (D.N_HEAD_PARAMS,)
    assert head.provenance["kind"] == "fitted"


def test_calibration_cannot_change_the_ranking(payload):
    for direction in payload["directions"]:
        chk = direction["calibration_monotonicity_check"]
        assert chk is not None
        assert chk["max_abs_diff"] < 1e-9
        h0 = direction["arms"]["H0"]["target_family"]["ranking"]["auc"]
        cal = direction["arms"]["CAL_ONLY"]["target_family"]["ranking"]["auc"]
        assert cal == pytest.approx(h0, abs=1e-9)


def test_a_calibration_that_reranks_is_a_hard_failure():
    """A genuinely DIFFERENT ordering — not a reversal, which is still monotone."""
    y = np.array([0.0, 0, 1, 1, 0, 1])
    w = np.ones(6)
    h0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    scrambled = np.array([0.5, 0.1, 0.6, 0.2, 0.3, 0.4])
    with pytest.raises(D.ControlFailure, match="changed the ranking"):
        D.assert_calibration_preserves_ranking(h0, scrambled, y, w)


def test_an_inverted_calibration_is_reported_rather_than_raised():
    """Under a true null the slope is a coin flip. Raising on `a < 0` made
    LINEAR_PROBE_NULL — the verdict that exists for that case — unreachable."""
    y = np.array([0.0, 0, 1, 1, 0, 1])
    w = np.ones(6)
    h0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    chk = D.assert_calibration_preserves_ranking(h0, -2.0 * h0 + 1.0, y, w)
    assert chk["inverted"] is True


def test_an_inverted_calibration_cannot_be_sold_as_a_calibration_verdict():
    rank = {"auc": 0.9}
    null_ranks = [{"auc": 0.5 + 0.01 * i} for i in range(-9, 10)]
    null_cals = [{"logloss_nats": 0.30} for _ in range(19)]
    kw = dict(h0_rank=rank, lin_rank={"auc": 0.90}, shuf_lin_ranks=null_ranks,
              h0_cal={"logloss_nats": 0.50}, cal_only_cal={"logloss_nats": 0.10},
              shuf_cal_cals=null_cals)
    assert D.decide_verdict(**kw)["verdict"] == "CALIBRATION_GLOBAL"
    assert D.decide_verdict(**kw, cal_slope_positive=False)["verdict"] == "LINEAR_PROBE_NULL"


def test_the_label_shuffled_control_returns_a_null(payload):
    """The acceptance criterion: this MUST produce a null, or the instrument is measuring
    leakage."""
    for direction in payload["directions"]:
        null = direction["verdict_detail"]["shuffled_null"]
        assert null["n_replicates"] >= 19
        assert null["auc_mean"] - 0.5 <= direction["verdict_detail"][
            "shuffle_null_tolerance_applied"]


def test_a_single_shuffled_refit_is_not_a_null(corpus, pools):
    """MEASURED, and the reason the replicated control exists: one draw has a spread far
    wider than the +/- 0.05 band that was once applied to it."""
    ht = D.ht_weights(corpus.kind, corpus.crop, corpus.sampling_rate())
    base = D.design_weights(
        D.kind_weights(corpus.kind, uniform_weight=0.01) * pools["fit_pool"],
        corpus.kind, corpus.crop, corpus.sampling_rate())
    tgt = pools["target"]
    aucs = []
    for seed in range(8):
        ysh = D.shuffle_labels(corpus.y, corpus.crop, base > 0, seed=seed)
        res = D.fit_head(corpus.X, ysh, base, spec=D.FitSpec(l2=1.0))
        h = D.LinearHead.from_beta(res.beta, basis_split=1)
        aucs.append(D._weighted_auc(h.logit(corpus.X)[tgt], corpus.y[tgt], ht[tgt]))
    assert float(np.std(aucs, ddof=1)) > 0.05, (
        "if one draw really were this tight, the replicated control would be "
        "unnecessary — but the band it was gated against was a standard ERROR of a mean")


def test_fewer_than_two_shuffles_is_refused(corpus, caps, pools):
    with pytest.raises(ValueError, match="not a null"):
        D.build_arms(corpus, caps, source_mask=pools["fit_pool"], n_shuffles=1)


def test_the_shuffled_arms_carry_distinct_seeds_and_distinct_labels(built):
    b, _blocked, _fps = built
    seeds = {b[n].notes["shuffle_seed"] for n in b if D.arm_base(n) == "SHUF_LIN"}
    assert len(seeds) == 3
    ys = {b[n].y.tobytes() for n in b if D.arm_base(n) == "SHUF_LIN"}
    assert len(ys) == 3, "replicated shuffles produced identical label vectors"


def test_too_few_replicates_for_the_stated_alpha_raises_rather_than_returning_null():
    """A verdict of LINEAR_PROBE_NULL that is really 'not enough shuffles' would be a
    negative result with no denominator."""
    ranks = [{"auc": 0.5}] * 5
    cals = [{"logloss_nats": 0.3}] * 5
    with pytest.raises(D.ControlFailure, match="never reaches alpha"):
        D.decide_verdict(h0_rank={"auc": 0.9}, lin_rank={"auc": 0.99},
                         shuf_lin_ranks=ranks, h0_cal={"logloss_nats": 0.5},
                         cal_only_cal={"logloss_nats": 0.1}, shuf_cal_cals=cals)


def test_an_above_chance_shuffled_null_is_leakage_and_blocks_every_verdict():
    ranks = [{"auc": 0.72 + 0.001 * i} for i in range(19)]
    cals = [{"logloss_nats": 0.3}] * 19
    with pytest.raises(D.ControlFailure, match="measuring leakage"):
        D.decide_verdict(h0_rank={"auc": 0.9}, lin_rank={"auc": 0.99},
                         shuf_lin_ranks=ranks, h0_cal={"logloss_nats": 0.5},
                         cal_only_cal={"logloss_nats": 0.1}, shuf_cal_cals=cals)


def test_a_below_chance_shuffled_null_is_not_leakage():
    """Leakage is scoring ABOVE chance on permuted labels. Permuting labels on a FIXED
    corpus leaves a corpus-level offset that no number of permutations averages away, and
    a two-sided gate reads that offset as leakage and emits nothing at all."""
    ranks = [{"auc": 0.30 + 0.001 * i} for i in range(19)]
    cals = [{"logloss_nats": 0.30} for _ in range(19)]
    out = D.decide_verdict(h0_rank={"auc": 0.9}, lin_rank={"auc": 0.99},
                           shuf_lin_ranks=ranks, h0_cal={"logloss_nats": 0.5},
                           cal_only_cal={"logloss_nats": 0.1}, shuf_cal_cals=cals)
    assert out["shuffle_null_below_chance"] is True
    assert out["verdict"] in D.PERMITTED_VERDICTS


def test_the_same_family_control_is_reported_and_is_not_a_transfer_result(payload):
    for direction in payload["directions"]:
        for row in direction["arms"].values():
            sf = row["same_family_heldout_crops"]
            assert sf["family"] == direction["source_family"]
            assert sf["crops"]
            assert "NOT a transfer result" in sf["BASIS"]


def test_the_verdict_gate_is_the_gain_over_h0_not_the_gain_over_the_null():
    """Regression lock. `net = (lin - h0) - (null - h0)` collapses to `lin - null`, which
    clears +0.01 whenever the features carry any signal at all — INCLUDING when the refit
    ranks strictly worse than the deployed head."""
    ranks = [{"auc": 0.5 + 0.001 * i} for i in range(-9, 10)]
    cals = [{"logloss_nats": 0.50}] * 19
    out = D.decide_verdict(
        h0_rank={"auc": 0.990}, lin_rank={"auc": 0.985},      # refit is WORSE than H0
        shuf_lin_ranks=ranks, h0_cal={"logloss_nats": 0.50},
        cal_only_cal={"logloss_nats": 0.50}, shuf_cal_cals=cals)
    assert out["ranking_gain_auc"] < 0
    assert out["verdict"] != "LINEAR_HEAD"


def test_a_calibration_gain_on_a_chance_ranking_is_not_a_calibration_verdict():
    """Log-loss falls whenever the intercept is refitted, because the sampled prior is
    nowhere near the population base rate. Perfectly calibrating a coin flip recovers no
    nuclei."""
    ranks = [{"auc": 0.5 + 0.001 * i} for i in range(-9, 10)]
    cals = [{"logloss_nats": 0.50}] * 19
    out = D.decide_verdict(
        h0_rank={"auc": 0.501}, lin_rank={"auc": 0.502}, shuf_lin_ranks=ranks,
        h0_cal={"logloss_nats": 0.50}, cal_only_cal={"logloss_nats": 0.02},
        shuf_cal_cals=cals)
    assert out["calibration_gain_nats"] > D.MIN_CALIBRATION_DELTA_NATS
    assert out["h0_ranks_above_null"] is False
    assert out["verdict"] == "LINEAR_PROBE_NULL"


def test_h5_reweights_positives_only_and_leaves_the_loss_functional_alone(built, pools):
    b, _blocked, _fps = built
    base = b["LIN_HEAD"].s
    h5 = b["H5"].s
    neg = pools["fit_pool"] & (b["H5"].y == 0)
    assert h5[neg] == pytest.approx(base[neg]), "H5 moved NEGATIVE weights"
    pos = pools["fit_pool"] & (b["H5"].y == 1)
    assert not np.allclose(h5[pos], base[pos]), "H5 did not reweight positives"


def test_subthreshold_local_maxima_are_never_labelled_negative():
    """A sub-threshold local maximum is an UNLABELLED voxel, very plausibly a real
    unannotated nucleus."""
    kind = np.array(["gt_centre", "uniform", "subthr_localmax"])
    w = D.kind_weights(kind, uniform_weight=0.01)
    assert w[2] == 0.0
    assert D.ht_weights(kind, np.array(["c", "c", "c"]), {"c": 1.0})[2] == 0.0


# ==================================================================================
# 1/4096 HORVITZ-THOMPSON CORRECTION
# ==================================================================================
def test_the_1_over_4096_intercept_correction_is_8_3178_logits():
    assert D.UNIFORM_SUBSAMPLE_DENOM == 4096
    assert D.HT_LOGIT_CORRECTION_1_4096 == pytest.approx(8.3177662, abs=1e-6)
    assert D.intercept_correction(1 / 4096) == pytest.approx(D.HT_LOGIT_CORRECTION_1_4096)


def test_the_correction_is_wider_than_the_base_rate_to_threshold_span():
    """Uncorrected, the intercept is wrong by more than the whole span from the
    population base rate (-7.29) to the deployed threshold (+3.4340)... no: it is wider
    than the span the threshold itself sits in, which is why no threshold derived from an
    uncorrected fit can be trusted."""
    assert D.HT_LOGIT_CORRECTION_1_4096 > D.DET_THRESHOLD_LOGIT
    assert D.HT_LOGIT_CORRECTION_1_4096 > abs(-7.29) + D.DET_THRESHOLD_LOGIT - 3.0


def test_an_uncorrected_intercept_is_overstated_by_exactly_log_4096(corpus, pools):
    """MEASURED on the intercept-only model, where the claim is exact: the only
    difference between the two fits is the design weight on `uniform` rows."""
    rate = corpus.sampling_rate()
    raw = D.kind_weights(corpus.kind, uniform_weight=1.0) * pools["fit_pool"]
    ht = D.design_weights(raw, corpus.kind, corpus.crop, rate)
    zero_design = np.zeros((corpus.n, 1))
    spec = D.FitSpec(l2=0.0, firth="never")
    b_raw = D.fit_head(zero_design, corpus.y, raw, spec=spec).beta[1]
    b_ht = D.fit_head(zero_design, corpus.y, ht, spec=spec).beta[1]
    assert (b_raw - b_ht) == pytest.approx(D.HT_LOGIT_CORRECTION_1_4096, abs=1e-4)


def test_absent_grid_dimensions_raise_rather_than_letting_the_rate_be_guessed(corpus):
    broken = D.Corpus(
        X=corpus.X, X_max=corpus.X_max, kind=corpus.kind, crop=corpus.crop,
        family=corpus.family, t=corpus.t, logit=corpus.logit,
        encoder_split=corpus.encoder_split,
        manifest={"crops": {c: {"n_frames": 10} for c in corpus.crops()}})
    with pytest.raises(D.ContractError, match="guessing it is not an option"):
        broken.sampling_rate()


def test_design_weights_refuse_a_crop_with_no_declared_rate(corpus):
    base = D.kind_weights(corpus.kind, uniform_weight=1.0)
    with pytest.raises(D.MissingCapability, match="sampling_rate"):
        D.design_weights(base, corpus.kind, corpus.crop, {})


def test_design_weights_scale_uniform_rows_by_the_inverse_rate(corpus):
    rate = corpus.sampling_rate()
    base = D.kind_weights(corpus.kind, uniform_weight=1.0)
    w = D.design_weights(base, corpus.kind, corpus.crop, rate)
    uni = corpus.kind == "uniform"
    assert w[uni] == pytest.approx(4096.0)
    assert w[corpus.kind == "gt_centre"] == pytest.approx(1.0)


def test_the_fixture_sampling_rate_is_exactly_one_over_4096(corpus):
    assert set(corpus.sampling_rate().values()) == {1 / 4096}


def test_a_run_reports_the_correction_it_applied(payload):
    for direction in payload["directions"]:
        ht = direction["horvitz_thompson"]
        assert ht["reference_1_over_4096_logits"] == pytest.approx(
            D.HT_LOGIT_CORRECTION_1_4096)
        assert ht["intercept_correction_logits"] == pytest.approx(
            [D.HT_LOGIT_CORRECTION_1_4096], abs=1e-6)


# ==================================================================================
# RANKING AND CALIBRATION SEPARATELY; BOTH DIRECTIONS SEPARATELY (C7)
# ==================================================================================
def test_ranking_and_calibration_are_reported_as_separate_basis_tagged_blocks(payload):
    for direction in payload["directions"]:
        for row in direction["arms"].values():
            tf = row["target_family"]
            assert "RANKING" in tf["ranking"]["BASIS"]
            assert "CALIBRATION" in tf["calibration"]["BASIS"]
            assert set(tf["ranking"]) & set(tf["calibration"]) <= {"BASIS", "promotes"}


def test_no_sampled_row_diagnostic_claims_to_promote_anything(payload):
    seen = 0

    def walk(o):
        nonlocal seen
        if isinstance(o, dict):
            if "promotes" in o:
                assert o["promotes"] is False
                seen += 1
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(payload)
    assert seen > 0
    assert payload["PROMOTION"]["promotable_from_this_instrument"] is False


def test_ranking_is_invariant_to_a_monotone_rescaling_of_the_score():
    score = np.linspace(-4, 4, 200)
    y = (np.arange(200) % 3 == 0).astype(float)
    w = np.ones(200)
    a = D.ranking_diagnostics(score, y, w)["auc"]
    b = D.ranking_diagnostics(3.0 * score + 11.0, y, w)["auc"]
    assert a == pytest.approx(b, abs=1e-12)


def test_calibration_is_not_invariant_to_a_rescaling_of_the_score():
    """If it were, ranking and calibration would not be measuring different things."""
    score = np.linspace(-4, 4, 200)
    y = (np.arange(200) % 3 == 0).astype(float)
    w = np.ones(200)
    a = D.calibration_diagnostics(score, y, w)["logloss_nats"]
    b = D.calibration_diagnostics(3.0 * score + 11.0, y, w)["logloss_nats"]
    assert abs(a - b) > 1e-6


def test_a_constant_score_gets_an_auc_of_exactly_one_half():
    """Ties are not a corner case: a shuffled refit under ridge collapses toward a
    constant, and a tie-blind cumulative sum would return whatever the row order implied
    instead of the 0.5 a null must produce."""
    assert D._weighted_auc(np.zeros(100), (np.arange(100) % 2).astype(float),
                           np.ones(100)) == pytest.approx(0.5, abs=1e-12)


def test_the_payload_carries_no_pooled_headline(payload):
    assert payload["pooled_verdict"] is None
    assert payload["pooled_auc"] is None
    assert "C7_NOTE" in payload
    assert len(payload["verdict_by_direction"]) == 2


def test_a_planted_pooled_headline_is_refused(payload):
    bad = dict(payload)
    bad["pooled_auc"] = 0.87
    with pytest.raises(D.VerdictError, match="reported separately"):
        D.assert_payload_clean(bad)


def test_the_two_directions_are_never_merged_into_one_row(payload):
    names = [d["direction"] for d in payload["directions"]]
    assert len(set(names)) == 2
    assert all("_to_" in n for n in names)


def test_the_crop_block_bootstrap_carries_the_two_embryo_caveat(corpus, pools):
    ht = D.ht_weights(corpus.kind, corpus.crop, corpus.sampling_rate())
    tgt = pools["target"]
    out = D.crop_block_bootstrap(corpus.logit[tgt], corpus.y[tgt], ht[tgt],
                                 corpus.crop[tgt], n_boot=25)
    assert "WITHIN-EMBRYO" in out["CAVEAT"]
    assert "not an estimate of private-embryo" in out["CAVEAT"]


def test_f1_on_this_row_pool_is_stamped_and_cannot_promote(payload):
    for direction in payload["directions"]:
        op = direction["arms"]["H0"]["target_family"]["operating_point"]
        assert "f1_SAMPLED_BASIS_ONLY" in op
        assert op["promotes"] is False


def test_f1_selection_is_refused_unless_explicitly_unlocked():
    tp = np.array([1.0]); fp = np.array([1.0]); fn = np.array([1.0])
    with pytest.raises(ValueError, match="degenerate"):
        D._objective_values("f1", tp, fp, fn, total_neg=1.0, budget=1.0, allow_f1=False)


# ==================================================================================
# NESTED SELECTION — THE THRESHOLD IS NEVER CHOSEN ON THE FIT ROWS
# ==================================================================================
def test_threshold_selection_refits_per_inner_fold(payload):
    for direction in payload["directions"]:
        for name, row in direction["arms"].items():
            if name == "H0" or D.arm_base(name) in D.SHUFFLED_ARMS:
                continue
            sel = row["selection"]
            assert sel["refit_per_fold"] is True
            assert sel["grouped_by"] == "crop"
            assert sel["n_folds"] >= 2
            assert "out-of-fold" in sel["BASIS"]


def test_selection_on_a_single_crop_is_refused(corpus, pools):
    one = corpus.crops(pools["fit_pool"])[:1]
    tiny = corpus.subset(pools["fit_pool"] & np.isin(corpus.crop, one))
    with pytest.raises(ValueError, match="selection on the fit rows"):
        D.nested_select_quantile(lambda _c: None, lambda _h: None, corpus=tiny,
                                 source_mask=np.ones(tiny.n, dtype=bool),
                                 ht_w=np.ones(tiny.n), n_est=tiny.n_est())


def test_grouped_folds_need_at_least_two():
    with pytest.raises(ValueError, match="n_folds must be"):
        D.grouped_folds(["a", "b", "c"], n_folds=1)


def test_more_folds_than_crops_is_refused():
    with pytest.raises(ValueError, match="cannot make"):
        D.grouped_folds(["a", "b"], n_folds=4)


# ==================================================================================
# FITTER AUDIT — the numpy-only fitter inherited from the v5 lane-2 swarm
# ==================================================================================
@pytest.fixture(scope="module")
def small_problem():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, D.FEAT_DIM))
    y = (rng.random(300) < _sig(X[:, 0] * 1.2 - 0.4)).astype(np.float64)
    s = rng.uniform(0.3, 3.0, size=300)
    return X, y, s


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_the_gradient_matches_finite_differences(small_problem):
    X, y, s = small_problem
    beta = D.probe_beta(D.N_HEAD_PARAMS)
    _o, g, _h = D._accumulate(X, y, s, None, beta, 1 << 18)
    num = np.empty_like(g)
    eps = 1e-6
    for k in range(g.size):
        d = np.zeros_like(beta)
        d[k] = eps
        num[k] = (D._accumulate(X, y, s, None, beta + d, 1 << 18, want_hess=False)[0]
                  - D._accumulate(X, y, s, None, beta - d, 1 << 18, want_hess=False)[0]
                  ) / (2 * eps)
    assert g == pytest.approx(num, rel=1e-5, abs=1e-6)


def test_the_hessian_matches_finite_differences_of_the_gradient(small_problem):
    X, y, s = small_problem
    beta = D.probe_beta(D.N_HEAD_PARAMS)
    _o, _g, H = D._accumulate(X, y, s, None, beta, 1 << 18)
    eps = 1e-6
    for k in (0, 5, D.FEAT_DIM):
        d = np.zeros_like(beta)
        d[k] = eps
        num = (D._accumulate(X, y, s, None, beta + d, 1 << 18)[1]
               - D._accumulate(X, y, s, None, beta - d, 1 << 18)[1]) / (2 * eps)
        assert H[k] == pytest.approx(num, rel=1e-4, abs=1e-6)


def test_the_count_term_gradient_matches_finite_differences(built):
    b, _blocked, _fps = built
    term = [t for t in b["H2"].terms if t.name == "count_ge"][0]
    beta = D.probe_beta(D.N_HEAD_PARAMS)
    _v, g, _h = term.value_grad_hess(beta)
    eps = 1e-6
    for k in (0, 7, D.FEAT_DIM):
        d = np.zeros_like(beta)
        d[k] = eps
        num = (term.value_grad_hess(beta + d)[0] - term.value_grad_hess(beta - d)[0]) / (2 * eps)
        assert g[k] == pytest.approx(num, rel=1e-4, abs=1e-6)


def test_the_temporal_term_gradient_matches_finite_differences(built):
    b, _blocked, _fps = built
    term = [t for t in b["H3"].terms if t.name == "temporal"][0]
    beta = D.probe_beta(D.N_HEAD_PARAMS)
    _v, g, _h = term.value_grad_hess(beta)
    eps = 1e-6
    for k in (0, 11, D.FEAT_DIM):
        d = np.zeros_like(beta)
        d[k] = eps
        num = (term.value_grad_hess(beta + d)[0] - term.value_grad_hess(beta - d)[0]) / (2 * eps)
        assert g[k] == pytest.approx(num, rel=1e-4, abs=1e-6)


def test_the_fit_lands_on_a_stationary_point(small_problem):
    X, y, s = small_problem
    res = D.fit_head(X, y, s, spec=D.FitSpec(l2=1.0))
    assert res.converged
    _o, g, _h = D._accumulate(X, y, s, None, res.beta, 1 << 18)
    g = g + 1.0 * np.concatenate([res.beta[:D.FEAT_DIM], [0.0]])
    assert float(np.abs(g).max()) < 1e-6


def test_the_separation_certificate_detects_complete_separation():
    X, y, s = D.make_separation_fixture()
    diag = D.diagnose_separation(X, y, s)
    assert diag.separated
    assert diag.kind in ("complete", "quasi")
    assert diag.min_margin > 0 or diag.kind == "quasi"


def test_an_overlapping_problem_is_not_reported_as_separated(small_problem):
    X, y, s = small_problem
    assert D.diagnose_separation(X, y, s).kind == "none"


def test_firth_engages_only_when_there_is_no_ridge_to_make_the_mle_exist():
    X, y, s = D.make_separation_fixture()
    assert D.fit_head(X, y, s, spec=D.FitSpec(l2=0.0)).firth_engaged is True
    assert D.fit_head(X, y, s, spec=D.FitSpec(l2=1.0)).firth_engaged is False
    assert D.fit_head(X, y, s, spec=D.FitSpec(l2=0.0, firth="never")).firth_engaged is False


def test_firth_keeps_the_coefficients_finite_under_separation():
    X, y, s = D.make_separation_fixture()
    firth = D.fit_head(X, y, s, spec=D.FitSpec(l2=0.0, firth="always"))
    plain = D.fit_head(X, y, s, spec=D.FitSpec(l2=0.0, firth="never"))
    assert np.isfinite(firth.beta).all()
    assert np.linalg.norm(firth.beta) < np.linalg.norm(plain.beta)


def test_negative_sample_weights_are_refused(small_problem):
    X, y, s = small_problem
    bad = s.copy()
    bad[0] = -1.0
    with pytest.raises(ValueError, match="negative sample weights"):
        D.fit_head(X, y, bad)


def test_an_arm_with_no_data_is_refused(small_problem):
    X, y, s = small_problem
    with pytest.raises(ValueError, match="no data"):
        D.fit_head(X, y, np.zeros_like(s))


# ==================================================================================
# C4 — |T u L| IS NOT A CEILING. THE RECOVERY ARITHMETIC, LOCKED.
# ==================================================================================
def test_the_value_per_recovered_node_and_the_counts_it_implies():
    """C4. Any larger threshold embeds a precision assumption and must state it."""
    value = 0.10332 / 15296
    assert value == pytest.approx(6.754707e-06, rel=1e-6)
    assert math.ceil(0.015 / value) == 2221
    assert math.ceil(0.020 / value) == 2961
