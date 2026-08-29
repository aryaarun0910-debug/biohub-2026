"""Contract tests for the association training harness (PKT-0034) that RUN it.

WHY THIS SHAPE. This tree has already been burned by a test that grepped source instead of running
it, and by a Gate-1 attempt that compared zero crops while looking well formed (FACT-0387). So
every test here builds a real synthetic cache on disk - a real pre-ILP parquet, a real ECB sidecar,
a real Gate-1 receipt, a real npz whose coordinates and features are what a faithful cache would
carry - and then executes the harness against it. Nothing is stubbed except the trained model,
which is the one thing a CPU test cannot have.

The synthetic world is deliberately built so the RECORDED probabilities are the ones a declared
reproducer re-derives from the cache alone. That is the only way a cache-gate test can fail for the
right reason: perturb the cache and the reproduction must break, not merely disagree with a
hard-coded expectation.

These are SOFTWARE contract tests. They decide nothing scientific.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import assoc_train_harness as H  # noqa: E402
from assoc_parent_dataset import evaluate  # noqa: E402

CROPS = [f"44b6_{i:02d}" for i in range(8)]
N_SRC, N_TGT, DIM, SCALE = 24, 24, 24, 2.0
GROUPS_PER_CROP = 24

REPRODUCER_SRC = '''
"""A declared reproducer: rebuilds the full source-by-target probability matrix from the cache."""
import numpy as np


def reproduce(crop, cache):
    frames = cache["frames"].tolist()
    starts = cache["starts"].tolist()
    ends = cache["ends"].tolist()
    feats = cache["features"]
    spans = list(zip(frames, starts, ends))
    out = {}
    for (ts, ss, se), (tt, tsq, te) in zip(spans, spans[1:]):
        if tt != ts + 1 or ts not in feats or tt not in feats:
            continue
        logits = feats[ts].astype(np.float64) @ feats[tt].astype(np.float64).T
        e = np.exp(logits - logits.max(axis=0, keepdims=True))
        p = e / e.sum(axis=0, keepdims=True)      # softmax over the SOURCE axis, as deployed
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                out[(ss + i, tsq + j)] = float(p[i, j])
    return out
'''


# --------------------------------------------------------------------------------------
# the synthetic world
# --------------------------------------------------------------------------------------

def _features(n_rows: int, offset: int) -> np.ndarray:
    f = np.zeros((n_rows, DIM), dtype=np.float32)
    for i in range(n_rows):
        f[i, (i + offset) % DIM] = SCALE
    return f


def _coords(crop_seed: int) -> np.ndarray:
    rows = []
    for t, n in ((0, N_SRC), (1, N_TGT)):
        for i in range(n):
            rows.append([t, 1 + (i % 7), 10 + i, 20 + ((i + crop_seed) % 5)])
    return np.asarray(rows, dtype=np.int16)


@pytest.fixture(scope="module")
def repro_module(tmp_path_factory):
    d = tmp_path_factory.mktemp("repro")
    (d / "synth_repro.py").write_text(REPRODUCER_SRC, encoding="utf-8")
    sys.path.insert(0, str(d))
    return "callable:synth_repro:reproduce"


def build_world(root: Path, repro_spec: str) -> dict:
    """Write cache npz, pre-ILP parquet, ECB sidecars and a Gate-1 receipt that agree."""
    import importlib

    reproduce = getattr(importlib.import_module(repro_spec.split(":")[1]), repro_spec.split(":")[2])
    cache_dir = root / "cache"
    ecb_dir = root / "ecb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ecb_dir.mkdir(parents=True, exist_ok=True)

    node_rows, edge_rows, receipt_crops = [], [], []
    for ci, crop in enumerate(CROPS):
        coords = _coords(ci)
        fs, ft = _features(N_SRC, 0), _features(N_TGT, 0)
        np.savez_compressed(
            cache_dir / f"{crop}.npz",
            coords=coords,
            frames=np.asarray([0, 1], dtype=np.int64),
            starts=np.asarray([0, N_SRC], dtype=np.int64),
            ends=np.asarray([N_SRC, N_SRC + N_TGT], dtype=np.int64),
            feat_frames=np.asarray([0, 1], dtype=np.int64),
            image_shape=np.asarray([8, 64, 64], dtype=np.int64),
            window=np.int64(2),
            downsample=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            feat_0=fs, feat_1=ft,
        )
        for nid, (t, z, y, x) in enumerate(coords.tolist()):
            node_rows.append({"dataset": crop, "row_type": "node", "node_id": nid, "t": t,
                              "z": float(z), "y": float(y), "x": float(x),
                              "source_id": None, "target_id": None, "edge_prob": None})
        probs = reproduce(crop, H.load_cache(cache_dir / f"{crop}.npz"))
        for (a, b), p in sorted(probs.items()):
            if p > H.DEPLOYED_FLOOR:
                edge_rows.append({"dataset": crop, "row_type": "edge", "node_id": None,
                                  "t": None, "z": None, "y": None, "x": None,
                                  "source_id": a, "target_id": b, "edge_prob": p})
        # The ECB sidecar keeps the top few candidates per target, deployed band included.
        by_target: dict[int, list] = {}
        for (a, b), p in probs.items():
            by_target.setdefault(b, []).append((p, a))
        s_id, t_id, e_p = [], [], []
        for b, lst in by_target.items():
            for p, a in sorted(lst, reverse=True)[:4]:
                s_id.append(a)
                t_id.append(b)
                e_p.append(p)
        np.savez_compressed(ecb_dir / f"{crop}.npz",
                            source_id=np.asarray(s_id, dtype=np.int64),
                            target_id=np.asarray(t_id, dtype=np.int64),
                            edge_prob=np.asarray(e_p, dtype=np.float32))
        receipt_crops.append({"crop": crop, "passed": True, "node_count_mismatches": [],
                              "band_a": {"missing": 0, "extra": 0, "max_abs_prob_delta": 0.0},
                              "band_b": {"checked": len(e_p), "missing": 0,
                                         "max_abs_prob_delta": 0.0}})

    preilp = root / "preilp.parquet"
    pl.DataFrame(node_rows + edge_rows,
                 schema={"dataset": pl.String, "row_type": pl.String, "node_id": pl.Int64,
                         "t": pl.Int64, "z": pl.Float64, "y": pl.Float64, "x": pl.Float64,
                         "source_id": pl.Int64, "target_id": pl.Int64,
                         "edge_prob": pl.Float64}).write_parquet(preilp)
    receipt = root / "assoc_feature_parity.json"
    receipt.write_text(json.dumps({"gate": "feature_parity", "attempt": 2,
                                   "all_passed": True, "crops": receipt_crops}), encoding="utf-8")
    return {"cache_dir": cache_dir, "ecb_dir": ecb_dir, "preilp": preilp, "receipt": receipt}


# group patterns: (n_candidates, baseline_is_correct). The true parent always has the smallest
# dist_um, so a model that reads geometry can convert the wrong ones; the deployed probability
# alone cannot.
PATTERN = [(3, True), (2, False), (1, True), (3, False), (1, True), (2, True)]


def build_surface(crops=CROPS, groups=GROUPS_PER_CROP) -> pl.DataFrame:
    rows = []
    for ci, crop in enumerate(crops):
        out_deg: dict[int, int] = {}
        for g in range(groups):
            n_cand, ok = PATTERN[g % len(PATTERN)]
            target = N_SRC + (g % N_TGT)
            sources = [(g * 3 + k + ci) % N_SRC for k in range(n_cand)]
            if len(set(sources)) != n_cand:
                sources = [(s + i) % N_SRC for i, s in enumerate(sources)]
            true_idx = 0 if ok else 1 % n_cand
            probs = [0.90 - 0.25 * k for k in range(n_cand)]
            best = max(probs)
            unreachable = (g % 12 == 11)
            for k, s in enumerate(sources):
                out_deg[s] = out_deg.get(s, 0) + 1
                is_true = int((k == true_idx) and not unreachable)
                dist = 2.0 + 0.01 * ci if is_true else 6.0 + 0.5 * k
                rows.append({
                    "crop": crop, "target": target, "source": s,
                    "prob": probs[k], "rank": k + 1, "margin_to_best": best - probs[k],
                    "n_candidates": n_cand, "dist_um": dist,
                    "dz_um": 0.0, "dy_um": dist, "dx_um": 0.0,
                    "src_out_degree": 0,
                    "is_true_parent": is_true,
                    "target_has_true_parent": 1,
                    "true_parent_is_candidate": int(not unreachable),
                    "target_matched_gt": 1,
                })
        for r in rows:
            if r["crop"] == crop:
                r["src_out_degree"] = out_deg.get(r["source"], 1)
    return pl.DataFrame(rows)


def chain_rows(tp, fp, fn, n_pred, n_est, div=(1, 1, 1)):
    denom = tp + fp + fn
    ratio = (n_pred - n_est) / n_est
    jac = tp / denom
    return {"edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
            "division_tp": div[0], "division_fp": div[1], "division_fn": div[2],
            "num_pred_nodes": n_pred, "node_recall": 0.99, "total_node_ratio": ratio,
            "edge_jaccard": jac, "adj_edge_jaccard": max(0.0, jac * (1 - 0.1 * ratio))}


def fake_summarise(rows):
    import assoc_report as ar

    tp = sum(r["edge_tp"] for r in rows)
    fp = sum(r["edge_fp"] for r in rows)
    fn = sum(r["edge_fn"] for r in rows)
    dtp = sum(r["division_tp"] for r in rows)
    dfp = sum(r["division_fp"] for r in rows)
    dfn = sum(r["division_fn"] for r in rows)
    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rows]
    adj = sum(a * r["adj_edge_jaccard"] for a, r in zip(w, rows)) / sum(w)
    dj = dtp / (dtp + dfp + dfn) if (dtp + dfp + dfn) else 0.0
    return {"edge_jaccard": tp / (tp + fp + fn), "adj_edge_jaccard": adj, "division_jaccard": dj,
            "division_tp": dtp, "division_fp": dfp, "division_fn": dfn,
            "node_recall": sum(r["node_recall"] for r in rows) / len(rows),
            "score": adj + ar.SCORE_DIVISION_WEIGHT * dj}


@pytest.fixture(scope="module")
def world(tmp_path_factory, repro_module):
    return build_world(tmp_path_factory.mktemp("world"), repro_module) | {"repro": repro_module}


@pytest.fixture(scope="module")
def surface():
    return build_surface()


# ======================================================================================
# 1. THE CACHE GATE - constraint 1, falsifier (a)
# ======================================================================================

def test_gate_passes_on_a_faithful_cache_with_a_declared_reproducer(world):
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"],
                     ecb_dir=world["ecb_dir"], crops=None, receipt=None,
                     reproducer=world["repro"])
    assert r["passed"], [c["reasons"] for c in r["crops"] if not c["passed"]]
    assert all(c["band_b"]["checked"] > 0 for c in r["crops"]), "band B was never compared"
    assert all(c["band_a"]["missing"] == 0 and c["band_a"]["extra"] == 0 for c in r["crops"])
    # the payload must say, in the artifact, that this is NOT the GPU-side numeric proof
    assert r["probability_parity_proof"].startswith("in_process_reproducer")


def test_gate_fails_when_the_cache_coordinates_are_not_the_recorded_nodes(tmp_path, world):
    """A cache that describes different nodes must not license a head. This is the CPU-side teeth."""
    bad = tmp_path / "cache"
    bad.mkdir()
    for p in sorted(world["cache_dir"].glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["coords"] = d["coords"].copy()
        d["coords"][3, 2] += 7          # move one node
        np.savez_compressed(bad / p.name, **d)
    r = H.gate_cache(cache_dir=bad, preilp=world["preilp"], ecb_dir=world["ecb_dir"],
                     crops=None, receipt=None, reproducer=world["repro"])
    assert not r["passed"]
    assert any("coordinates disagree" in reason
               for c in r["crops"] for reason in c["reasons"])


def test_gate_fails_when_a_frame_is_short(tmp_path, world):
    """A truncated detection silently shrinks the denominator; the gate must refuse, not pass."""
    bad = tmp_path / "cache"
    bad.mkdir()
    p = sorted(world["cache_dir"].glob("*.npz"))[0]
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["coords"] = d["coords"][:-1]
    d["ends"] = np.asarray([N_SRC, N_SRC + N_TGT - 1], dtype=np.int64)
    d["feat_1"] = d["feat_1"][:-1]
    np.savez_compressed(bad / p.name, **d)
    r = H.gate_cache(cache_dir=bad, preilp=world["preilp"], ecb_dir=world["ecb_dir"],
                     crops=[p.stem], receipt=None, reproducer=world["repro"])
    assert not r["passed"]
    assert any("node count" in reason for c in r["crops"] for reason in c["reasons"])


def test_gate_fails_when_the_sub_threshold_band_disagrees(tmp_path, world):
    """FACT-0382: all the contested errors live below 0.5, so band B is the band that matters."""
    ecb = tmp_path / "ecb"
    ecb.mkdir()
    for p in sorted(world["ecb_dir"].glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        prob = d["edge_prob"].copy()
        low = np.nonzero(prob <= H.DEPLOYED_FLOOR)[0]
        prob[low[0]] = float(prob[low[0]]) + 0.01
        d["edge_prob"] = prob
        np.savez_compressed(ecb / p.name, **d)
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"], ecb_dir=ecb,
                     crops=None, receipt=None, reproducer=world["repro"])
    assert not r["passed"]
    assert any("band B probability delta" in reason
               for c in r["crops"] for reason in c["reasons"])


def test_gate_refuses_without_a_gate1_receipt(world):
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"],
                     ecb_dir=world["ecb_dir"], crops=None, receipt=None, reproducer="receipt")
    assert not r["passed"]
    assert any("no Gate-1 receipt" in x for x in r["refusals"])


def test_gate_refuses_a_failed_receipt_and_a_zero_band_b_receipt(tmp_path, world):
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"all_passed": False, "crops": []}), encoding="utf-8")
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"],
                     ecb_dir=world["ecb_dir"], crops=None, receipt=failed, reproducer="receipt")
    assert not r["passed"]

    hollow = tmp_path / "hollow.json"
    hollow.write_text(json.dumps({
        "all_passed": True,
        "crops": [{"crop": c, "passed": True, "node_count_mismatches": [],
                   "band_b": {"checked": 0}} for c in CROPS]}), encoding="utf-8")
    r2 = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"],
                      ecb_dir=world["ecb_dir"], crops=None, receipt=hollow, reproducer="receipt")
    assert not r2["passed"]
    assert any("band-B" in reason for c in r2["crops"] for reason in c["reasons"])


def test_licence_pins_bytes_and_a_changed_cache_is_refused(tmp_path, world):
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"],
                     ecb_dir=world["ecb_dir"], crops=None, receipt=world["receipt"],
                     reproducer="receipt")
    assert r["passed"]
    assert r["probability_parity_proof"] == "gate1_receipt"
    lic = tmp_path / "licence.json"
    H.write_licence(r, lic)
    assert H.verify_licence(lic, world["cache_dir"], CROPS)["verified"] == len(CROPS)

    moved = tmp_path / "cache2"
    moved.mkdir()
    for p in sorted(world["cache_dir"].glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["feat_1"] = d["feat_1"] + 0.5
        np.savez_compressed(moved / p.name, **d)
    with pytest.raises(H.HarnessRefusal, match="cache bytes changed"):
        H.verify_licence(lic, moved, CROPS)


def test_write_licence_refuses_for_a_failed_gate():
    with pytest.raises(H.HarnessRefusal):
        H.write_licence({"passed": False}, Path("nowhere.json"))


def test_training_refuses_when_the_gate_did_not_pass(surface):
    models = [H.ModelSpec(tag="linear", model_class="linear", features=["prob", "dist_um"])]
    with pytest.raises(H.HarnessRefusal, match="cache gate did not pass"):
        H.run_harness(table=surface, models=models, fold=0, n_splits=2,
                      cache_gate={"passed": False, "crops": []})


def test_contextual_model_without_a_passing_gate_is_refused(surface):
    models = [H.ModelSpec(tag="ctx", model_class="contextual",
                          contract={"name": "declared", "dim": DIM})]
    with pytest.raises(H.HarnessRefusal, match="measures the cache"):
        H.run_harness(table=surface, models=models, fold=0, n_splits=2, cache_gate=None)


# ======================================================================================
# 2. GROUPING - constraint 2, falsifier (b)
# ======================================================================================

def test_no_crop_and_no_target_straddles_any_split(surface):
    from sklearn.model_selection import GroupKFold

    dec = surface.filter(pl.col("true_parent_is_candidate") == 1)
    crops = dec["crop"].to_numpy()
    targets = dec["target"].to_numpy().astype(np.int64)
    x = dec.select(["prob"]).to_numpy()
    y = dec["is_true_parent"].to_numpy()
    seen = 0
    for i, (tr, te) in enumerate(GroupKFold(n_splits=4).split(x, y, groups=crops)):
        H.assert_group_integrity(crops, targets, tr, te, f"t/fold{i}")
        seen += 1
    assert seen == 4


def test_group_integrity_catches_a_target_that_straddles():
    """Both halves of the guard must be reported, not whichever was checked first."""
    crops = np.array(["a", "a", "b", "b"])
    targets = np.array([1, 1, 2, 2])
    with pytest.raises(H.HarnessRefusal) as exc:
        H.assert_group_integrity(crops, targets, np.array([0, 2]), np.array([1, 3]), "t")
    assert "straddle" in str(exc.value) and "crop on both sides" in str(exc.value)


def test_group_integrity_refuses_a_degenerate_split():
    with pytest.raises(H.HarnessRefusal, match="degenerate split"):
        H.assert_group_integrity(np.array(["a"]), np.array([1]),
                                 np.array([0]), np.array([], dtype=int), "t")


def test_folds_report_that_they_are_not_embryo_held_out(surface):
    g = H.grouping_block(surface["crop"].to_numpy(), "GroupKFold", 2)
    assert g["embryos"] == ["44b6"]
    assert g["embryo_held_out"] is False
    assert "WITHIN-embryo" in g["note"]


# ======================================================================================
# 3. ABSTENTION AND SURFACE EQUIVALENCE - constraint 3
# ======================================================================================

def test_with_abstention_off_the_harness_is_the_frozen_surface(surface):
    eq = H.assert_surface_equivalence(surface, "prob")
    assert eq["heartbeat"] == "SURFACE_EQUIVALENT"
    frozen = evaluate(surface, "prob")
    mine, ledger, agg = H.decide(surface, "prob", None, preserve_single=False)
    assert mine == frozen["per_target_correct"]
    assert agg["contested"]["top1"] == frozen["contested"]["top1"]
    assert ledger.summary()["n_regressions"] == 0


def test_surface_equivalence_refuses_when_the_decision_diverges(surface):
    """A guard that cannot fire is decoration - plant a divergence and require the refusal."""
    original = H.decide

    def wrong(table, score_col, abstain_col, preserve_single):
        per_target, ledger, agg = original(table, score_col, abstain_col, preserve_single)
        first = next(iter(per_target))
        per_target[first] = 1 - per_target[first]
        return per_target, ledger, agg

    H.decide = wrong
    try:
        with pytest.raises(H.HarnessRefusal, match="differs from the frozen surface"):
            H.assert_surface_equivalence(surface, "prob")
    finally:
        H.decide = original


def test_abstention_ties_resolve_to_the_null_like_the_deployed_threshold(surface):
    """The deployed rule keeps pairs strictly above the threshold, so an exact tie abstains."""
    tbl = surface.with_columns(pl.col("prob").max().over(["crop", "target"]).alias("_null"))
    per_target, _ledger, agg = H.decide(tbl, "prob", "_null", preserve_single=False)
    assert sum(per_target.values()) == 0
    assert agg["abstentions"]["contested"] == agg["contested"]["n"]


# ======================================================================================
# 4. SINGLE-CANDIDATE ACCOUNTING - constraint 4, falsifier (c). THE DESIGN RISK.
# ======================================================================================

def _abstain_on_one_single_candidate(surface):
    """A score column plus a null column that abstains on exactly one single-candidate target."""
    counts = (surface.filter(pl.col("true_parent_is_candidate") == 1)
                     .group_by(["crop", "target"]).agg(pl.len().alias("n")))
    single = counts.filter(pl.col("n") == 1).sort(["crop", "target"]).row(0)
    tbl = surface.with_columns([
        pl.col("prob").alias("_score"),
        pl.when((pl.col("crop") == single[0]) & (pl.col("target") == single[1]))
          .then(pl.lit(1.0)).otherwise(pl.lit(H.NEVER)).alias("_null"),
    ])
    return tbl, (single[0], int(single[1]))


def test_a_single_candidate_regression_is_reported_by_identity_not_netted(surface):
    tbl, victim = _abstain_on_one_single_candidate(surface)
    per_target, ledger, _agg = H.decide(tbl, "_score", "_null", preserve_single=False)
    s = ledger.summary()
    assert s["n_regressions"] == 1
    assert (s["regressions"][0]["crop"], s["regressions"][0]["target"]) == victim
    assert s["regressions"][0]["reason"] == "abstained"
    assert len(s["regressions"]) == s["n_regressions"]
    assert s["side_b_of_the_two_sided_bar"].startswith("LIVE")
    assert per_target[victim] == 0


def test_preservation_by_construction_prevents_the_loss_and_still_reports_the_shadow(surface):
    """FACT-0386's blind spot closed: the constraint is priced, not hidden."""
    tbl, victim = _abstain_on_one_single_candidate(surface)
    per_target, ledger, _agg = H.decide(tbl, "_score", "_null", preserve_single=True)
    s = ledger.summary()
    assert s["n_regressions"] == 0
    assert s["top1"] == 1.0
    assert s["n_shadow_regressions"] == 1
    assert (s["shadow_regressions"][0]["crop"], s["shadow_regressions"][0]["target"]) == victim
    assert s["side_b_of_the_two_sided_bar"].startswith("INERT")
    assert per_target[victim] == 1


def test_a_count_without_its_identities_cannot_be_emitted():
    led = H.SingleCandidateLedger(preserved_by_construction=False, n_targets=10)
    led.regressions = [{"crop": "a", "target": 1, "source": 0, "reason": "abstained"}]
    assert led.summary()["n_regressions"] == 1
    led.n_targets = 10
    led.regressions = []
    # a ledger claiming preservation while holding losses is a contract violation
    bad = H.SingleCandidateLedger(preserved_by_construction=True, n_targets=3)
    bad.regressions = [{"crop": "a", "target": 1, "source": 0, "reason": "abstained"}]
    with pytest.raises(H.HarnessRefusal, match="constraint is not being applied"):
        bad.summary()


def test_verdict_blocks_on_single_candidate_regressions_even_when_contested_improves(surface):
    model = {
        "surface": {"contested": {"top1": 1.0, "n": 10}},
        "single_candidate": {"n_regressions": 2, "n_shadow_regressions": 0,
                             "regressions": [{"crop": "a", "target": 1},
                                             {"crop": "a", "target": 2}],
                             "side_b_of_the_two_sided_bar": "LIVE"},
        "conversions": {"contested": {"crop_paired_bootstrap": {"favourable": True}}},
    }
    v = H.harness_verdict(model, 0.8338, {"verdict": {"promotable": True}}, degenerate=False)
    assert not v["promotable"]
    assert any("single-candidate target(s) regressed" in b for b in v["blockers"])


# ======================================================================================
# 5. THE THREE CLASSES, ONE REPORT - falsifier (d)
# ======================================================================================

def _gate(world):
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"],
                     ecb_dir=world["ecb_dir"], crops=None, receipt=world["receipt"],
                     reproducer="receipt")
    assert r["passed"]
    return r


def _node_feats(world):
    return {c: H.node_features(H.load_cache(world["cache_dir"] / f"{c}.npz")) for c in CROPS}


def test_all_three_model_classes_run_and_emit_identical_channels(surface, world):
    models = [
        H.ModelSpec(tag="linear.geom", model_class="linear", features=["prob", "dist_um"]),
        H.ModelSpec(tag="tree.geom", model_class="tree", features=["prob", "dist_um"]),
        H.ModelSpec(tag="ctx.declared", model_class="contextual",
                    contract={"name": "declared_v1", "dim": DIM, "pair_builder": "diff",
                              "extra_features": ["prob", "dist_um"]}),
    ]
    payload = H.run_harness(table=surface, models=models, fold=0, n_splits=2,
                            cache_gate=_gate(world), node_feat_by_crop=_node_feats(world))
    assert payload["heartbeat"] == "ASSOC_TRAIN_HARNESS_COMPLETE"
    keys = [set(m.keys()) for m in payload["models"]]
    assert all(k == keys[0] for k in keys)
    assert {m["model_class"] for m in payload["models"]} == {"linear", "tree", "contextual"}
    for m in payload["models"]:
        assert m["surface"]["contested"]["n"] > 0
        assert m["single_candidate"]["preserved_by_construction"] is True
        assert set(m["conversions"]) == {"all_decidable", "contested", "single_candidate"}
    assert payload["grouping"]["embryo_held_out"] is False


def test_a_geometry_reading_model_converts_contested_targets_by_identity(surface, world):
    """The harness must show WHICH targets moved, not only that a number rose."""
    models = [H.ModelSpec(tag="linear.geom", model_class="linear",
                          features=["prob", "dist_um"])]
    payload = H.run_harness(table=surface, models=models, fold=0, n_splits=2,
                            cache_gate=_gate(world), node_feat_by_crop=_node_feats(world))
    m = payload["models"][0]
    conv = m["conversions"]["contested"]
    assert conv["gained"] > 0
    assert conv["net"] == conv["gained"] - conv["lost"]
    assert conv["churn"] == conv["gained"] + conv["lost"]
    assert m["surface"]["contested"]["top1"] > payload["deployed_baseline"]["contested"]["top1"]
    assert "ci95" in conv["crop_paired_bootstrap"]


def test_every_class_reports_through_assoc_report_build_report(surface, world):
    arms = {
        tag: {"control": [chain_rows(900, 60, 60, 1000, 1000), chain_rows(800, 90, 90, 990, 1000)],
              "candidate": [chain_rows(930, 50, 40, 1002, 1000),
                            chain_rows(830, 80, 70, 992, 1000)]}
        for tag in ("linear.geom", "ctx.declared")
    }
    models = [
        H.ModelSpec(tag="linear.geom", model_class="linear", features=["prob", "dist_um"]),
        H.ModelSpec(tag="ctx.declared", model_class="contextual",
                    contract={"name": "declared_v1", "dim": DIM,
                              "extra_features": ["prob", "dist_um"]}),
    ]
    payload = H.run_harness(table=surface, models=models, fold=0, n_splits=2,
                            cache_gate=_gate(world), node_feat_by_crop=_node_feats(world),
                            chain_arms=arms, summarise=fake_summarise)
    for m in payload["models"]:
        fc = m["full_chain"]
        assert fc is not None and fc["heartbeat"] == "ASSOC_REPORT_COMPLETE"
        assert set(fc["channels"]) >= {"edge_jaccard_raw", "count_adjustment", "node_recall",
                                       "division_counts", "final_graph_edges", "score",
                                       "parent_conversions"}
        assert abs(fc["channels"]["score"]["identity_check"]) < 1e-9


def _always_abstains():
    """A null scored at 1.0 dominates every calibrated probability, so abstention is certain."""
    return H.ModelSpec(tag="linear.abstain", model_class="linear",
                       features=["prob", "dist_um"],
                       abstain=H.AbstainPolicy(kind="fixed", tau=1.0))


def test_abstention_end_to_end_enumerates_every_single_candidate_regression(surface, world):
    """Abstention live on the whole surface: each loss must arrive with its identity."""
    payload = H.run_harness(table=surface, models=[_always_abstains()], fold=0, n_splits=2,
                            preserve_single=False,
                            cache_gate=_gate(world), node_feat_by_crop=_node_feats(world))
    m = payload["models"][0]
    s = m["single_candidate"]
    assert s["n_regressions"] > 0, "the abstention policy did not fire; the test proves nothing"
    assert s["n_regressions"] == s["n"], "every single-candidate target should have been lost"
    assert len(s["regressions"]) == s["n_regressions"]
    assert all(r["reason"] == "abstained" for r in s["regressions"])
    assert not m["verdict"]["promotable"]
    assert any("single-candidate" in b for b in m["verdict"]["blockers"])
    assert m["conversions"]["single_candidate"]["lost"] == s["n_regressions"]


def test_the_same_policy_under_preservation_moves_the_losses_into_the_shadow(surface, world):
    payload = H.run_harness(table=surface, models=[_always_abstains()], fold=0, n_splits=2,
                            preserve_single=True,
                            cache_gate=_gate(world), node_feat_by_crop=_node_feats(world))
    s = payload["models"][0]["single_candidate"]
    assert s["n_regressions"] == 0 and s["top1"] == 1.0
    assert s["n_shadow_regressions"] == s["n"]
    assert payload["models"][0]["verdict"]["single_candidate_shadow_regressions"] == s["n"]
    assert s["side_b_of_the_two_sided_bar"].startswith("INERT")


def test_train_quantile_tau_is_fitted_per_fold_and_never_on_validation(surface, world):
    models = [H.ModelSpec(tag="linear.q", model_class="linear", features=["prob", "dist_um"],
                          abstain=H.AbstainPolicy(kind="train_quantile", q=0.5))]
    payload = H.run_harness(table=surface, models=models, fold=0, n_splits=2,
                            preserve_single=True,
                            cache_gate=_gate(world), node_feat_by_crop=_node_feats(world))
    folds = payload["models"][0]["folds"]
    assert len(folds) == 2
    assert all(f["abstain_tau"] is not None and 0.0 <= f["abstain_tau"] <= 1.0 for f in folds)
    # the validation crops of the two folds partition the surface: no crop is scored by a model
    # that saw it, and therefore no tau is fitted on the targets it is applied to
    a, b = (set(f["val_crops"]) for f in folds)
    assert not (a & b) and (a | b) == set(CROPS)


# ======================================================================================
# 6. THE DECLARED CONTRACT, AND THE DEGENERATE FOLD
# ======================================================================================

def test_an_undeclared_contextual_contract_is_refused():
    with pytest.raises(H.HarnessRefusal, match="no declared contract"):
        H.ContextContract.from_dict(None)


def test_a_contract_whose_dimension_does_not_match_the_cache_is_refused(world):
    c = H.ContextContract.from_dict({"name": "wrong", "dim": DIM + 1})
    feats = _node_feats(world)[CROPS[0]]
    with pytest.raises(H.HarnessRefusal, match="does not describe this cache"):
        c.validate_against_cache(feats)


def test_a_contract_naming_features_outside_the_frozen_surface_is_refused():
    with pytest.raises(H.HarnessRefusal, match="outside the frozen surface"):
        H.ContextContract.from_dict({"name": "x", "dim": 4, "extra_features": ["invented"]})


def test_a_degenerate_fold_refuses_a_ranking_claim():
    """FACT-0381 / FACT-0382: a metric that cannot fail is not evidence."""
    single_only = build_surface(crops=CROPS[:4], groups=4).filter(pl.col("n_candidates") == 1)
    with pytest.raises(H.HarnessRefusal, match="ZERO contested targets"):
        H.run_harness(table=single_only, models=[], fold=1, n_splits=2)


# ======================================================================================
# 7. THE CLI, END TO END
# ======================================================================================

def test_cli_gate_then_train_runs_the_whole_path(tmp_path, world, surface):
    table = tmp_path / "surface.parquet"
    surface.write_parquet(table)
    lic = tmp_path / "licence.json"
    rc = H.main(["gate", "--cache-dir", str(world["cache_dir"]), "--preilp", str(world["preilp"]),
                 "--ecb-dir", str(world["ecb_dir"]), "--receipt", str(world["receipt"]),
                 "--licence", str(lic), "--out", str(tmp_path / "gate.json")])
    assert rc == 0 and lic.is_file()

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "name": "synthetic", "fold": 0, "table": str(table),
        "cache": {"dir": str(world["cache_dir"]), "receipt": str(world["receipt"]),
                  "preilp": str(world["preilp"]), "ecb_dir": str(world["ecb_dir"]),
                  "licence": str(lic), "reproducer": "receipt", "crops": CROPS},
        "cv": {"kind": "GroupKFold", "n_splits": 2},
        "preserve_single_candidate": True,
        "models": [{"tag": "linear.geom", "class": "linear", "features": ["prob", "dist_um"]},
                   {"tag": "ctx", "class": "contextual",
                    "contract": {"name": "declared_v1", "dim": DIM,
                                 "extra_features": ["prob", "dist_um"]}}],
        "out_dir": str(tmp_path / "out"),
    }), encoding="utf-8")
    assert H.main(["train", "--spec", str(spec)]) == 0
    payload = json.loads((tmp_path / "out" / "harness_f0.json").read_text(encoding="utf-8"))
    assert payload["heartbeat"] == "ASSOC_TRAIN_HARNESS_COMPLETE"
    assert payload["cache_gate"]["passed"] is True
    assert len(payload["models"]) == 2


def test_cli_train_refuses_without_a_licence(tmp_path, world, surface):
    table = tmp_path / "surface.parquet"
    surface.write_parquet(table)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "fold": 0, "table": str(table),
        "cache": {"dir": str(world["cache_dir"]), "receipt": str(world["receipt"]),
                  "preilp": str(world["preilp"]), "ecb_dir": str(world["ecb_dir"]),
                  "licence": str(tmp_path / "missing.json"), "crops": CROPS},
        "cv": {"kind": "GroupKFold", "n_splits": 2},
        "models": [{"tag": "ctx", "class": "contextual",
                    "contract": {"name": "d", "dim": DIM}}],
        "out_dir": str(tmp_path / "out"),
    }), encoding="utf-8")
    with pytest.raises(H.HarnessRefusal, match="no cache licence"):
        H.main(["train", "--spec", str(spec)])


def test_prepared_fold_specs_are_valid_and_declare_the_fold1_guard():
    d = ROOT / "scripts" / "win_bet" / "assoc_specs"
    f0 = json.loads((d / "harness_f0.json").read_text(encoding="utf-8"))
    f1 = json.loads((d / "harness_f1.json").read_text(encoding="utf-8"))
    for spec in (f0, f1):
        models = [H.ModelSpec.from_dict(m) for m in spec["models"]]
        assert models and spec["cv"]["kind"] in {"GroupKFold", "LeaveOneCropOut"}
        assert spec["preserve_single_candidate"] is True
        assert "licence" in spec["cache"]
        for m in models:
            if m.model_class == "contextual":
                H.ContextContract.from_dict(m.contract)
    assert f0["fold"] == 0 and f1["fold"] == 1
    # FACT-0381 / FACT-0382: fold 1 is degenerate for ranking today and must make no claim.
    assert f1["no_claim"] is True
    assert "FACT-0381" in f1["note"] and "FACT-0382" in f1["note"]


def test_module_docstring_carries_the_one_real_cache_smoke_command():
    doc = H.__doc__
    assert "assoc_train_harness.py gate" in doc
    assert "--receipt" in doc and "--licence" in doc
    assert textwrap.dedent(doc).count("ASSOC_CACHE_GATE_PASSED") >= 1
