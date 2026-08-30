r"""BIND a frozen-trunk feature cache to everything that could make a null uninterpretable, and
INDEPENDENTLY re-check that binding afterwards.

WHY THIS EXISTS (PKT-0037)
--------------------------
The expensive failure on this lane is not a wrong number. It is a NULL RESULT nobody can read.
If a head trained on a cached feature surface scores at chance, there are two explanations and
the campaign cannot currently tell them apart:

    (1) the head does not work; or
    (2) we fed it the wrong features.

Every ingredient of (2) is invisible in the artifact today. `FACT-0392` names the sharpest case:
WHICH TRUNK produced the 32-dim features is a ``--weights`` command-line argument in the
publisher's own cache script (``cache_official_hoct_features.py:499``) and is recorded NOWHERE in
the checkpoint or in the per-crop cache. Feeding official-trunk features to a head trained on
StableDet-trunk features would look EXACTLY like a wrong feature contract. `FACT-0391` makes the
same point one level down: a strict state-dict load is necessary and demonstrably not sufficient
to know what function ran.

So a cache is not trusted because it was written carefully. It is trusted because a SEPARATE
command, run later and reading only what is on disk, can still say which trunk produced it, which
fold and embryo it belongs to, how its features were normalised, what its node ordering and ids
mean, which candidate rule built its probability bands, and which notebook and commit ran.

WHAT THE MANIFEST BINDS
-----------------------
==========================  ==========================================================
trunk                       checkpoint sha256, byte size, role, provenance, and the
                            digest of the FEATURES IT PRODUCED - so a cache swapped
                            between two trunk directories stops matching its manifest
fold and embryo             fold id, held-out embryo, and every crop stem, cross-checked
                            against the fold<->embryo map (fold 0 = 44b6, fold 1 = 6bba)
feature normalisation       the transform applied after ``_index_features``, its dtype
                            and its per-crop statistics
node ordering and ids       the id convention, the frame partition, and a digest that
                            pairs each coordinate row with its own feature row, so a
                            reorder of one against the other is detectable
candidate graph             rule, floor, cap, gate, det threshold, pool kernel, softmax
                            axis, abstain mass - and the per-band pair counts
notebook and commit         notebook path + sha256, spec, git commit, kernel + version
==========================  ==========================================================

FAIL CLOSED, EVERYWHERE
-----------------------
A missing manifest, an unreadable cache, a crop present in one and absent from the other, an
unknown schema version, a band with zero pairs, a checkpoint whose bytes no longer hash to the
recorded value - every one of these is a REJECT, never a warning and never a skip. `FACT-0387`
is the standing lesson: the gate that compared zero crops is the gate that would have licensed
two wasted GPU sessions had it defaulted to a pass.

AND IT IS PROVEN BY MUTATION, NOT BY ASSERTION
----------------------------------------------
``self-test`` manufactures four defects - a swapped trunk, reordered nodes, an empty probability
band, and wrong-fold weights - and requires the auditor to REJECT each one while accepting the
clean cache it started from. A checker that has not been shown to reject is not a checker, and
this project has already been burned by a test that grepped source instead of running it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

# fold <-> held-out embryo. AGENTS.md and the upstream audit item U10 both fix this direction:
# fold 0 is the 71-crop 44b6 direction, fold 1 the 128-crop 6bba direction.
FOLD_EMBRYO = {"0": "44b6", "1": "6bba"}

# The support pack ships only ``split_0`` (trained on 6bba). It is LOEO-clean on fold 0 and
# LEAKY on fold 1 - the EXP-0019 defect. Mirrored from tests/test_loeo_weights_hygiene.py.
PACK_WEIGHTS_TOKEN = "split_0"

TRUNK_ROLES = {
    "pack_split0",     # support-pack primary, legitimate on fold 0 only
    "oof_split1",      # our out-of-fold split_1, required on fold 1
    "official",        # HOCT publisher DEFAULT_WEIGHTS, loeo_official_f0_e3/split_0
    "stabledet",       # the StableDet trunk shipped in the HOCT bundle
}

# Cache keys that carry node identity. Without the frame partition a node id is not provable, so
# a cache lacking it is REJECTED rather than audited loosely.
REQUIRED_NODE_KEYS = ("coords", "frames", "starts", "ends", "feat_frames")
REQUIRED_BAND_KEYS = ("source_id", "target_id", "edge_prob")


class Reject(Exception):
    """Any condition that makes the cache uninterpretable. Never downgraded to a warning."""


# --------------------------------------------------------------------------------------------
# digests
# --------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def load_cache(path: Path) -> dict:
    """Read one crop cache, refusing anything whose node identity cannot be established."""
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as err:                                    # pragma: no cover - I/O shape
        raise Reject(f"{path.name}: unreadable cache ({type(err).__name__}: {err})") from err
    missing = [k for k in REQUIRED_NODE_KEYS if k not in z.files]
    if missing:
        raise Reject(
            f"{path.name}: cache is missing {missing}. Without the frame partition a node id "
            "is not provable, so this cache cannot be bound to an ordering and must not be "
            "trained on."
        )
    missing_band = [k for k in REQUIRED_BAND_KEYS if k not in z.files]
    if missing_band:
        raise Reject(
            f"{path.name}: cache carries no candidate surface ({missing_band}). FACT-0382 puts "
            "the entire learnable population BELOW the deployed 0.5 floor, so a cache that "
            "cannot show its probability bands cannot be checked for an empty one."
        )
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------------------------
# per-crop binding
# --------------------------------------------------------------------------------------------
def crop_binding(crop: str, data: dict, deployed_floor: float) -> dict:
    coords = np.asarray(data["coords"])
    frames = np.asarray(data["frames"]).astype(np.int64)
    starts = np.asarray(data["starts"]).astype(np.int64)
    ends = np.asarray(data["ends"]).astype(np.int64)
    feat_frames = [int(t) for t in np.asarray(data["feat_frames"]).tolist()]

    if not (len(frames) == len(starts) == len(ends)):
        raise Reject(f"{crop}: frames/starts/ends disagree in length - the partition is corrupt")

    # NODE ORDERING. The id convention is "positional index into coords_so_far", which is only
    # meaningful if the frame blocks partition [0, N) contiguously in increasing frame order.
    order_ok = bool(np.all(np.diff(frames) > 0)) and bool(np.all(starts[1:] == ends[:-1])) \
        and (len(starts) == 0 or (int(starts[0]) == 0 and int(ends[-1]) == len(coords)))

    # TWO SEPARATE DIGESTS, because the two defects they catch have DIFFERENT remedies and an
    # auditor that cannot tell them apart sends the operator to the wrong place.
    #   coords_digest    changes when the NODES are reordered - ids now name different cells,
    #                    while every id recorded elsewhere still assumes the old order.
    #   features_digest  changes when the same nodes carry different FEATURE VALUES - which is
    #                    what a swapped trunk looks like, the FACT-0392 confound.
    # The pair digest over both is what the manifest binds to the trunk checkpoint.
    coords_digest = _digest(coords)
    feat_parts = [np.asarray(data[f"feat_{t}"]).astype(np.float64) for t in sorted(feat_frames)]
    features_digest = _digest(*feat_parts) if feat_parts else _digest(np.empty(0))
    pair_digest = _digest(coords.astype(np.float64), *feat_parts)

    src = np.asarray(data["source_id"]).astype(np.int64)
    tgt = np.asarray(data["target_id"]).astype(np.int64)
    prob = np.asarray(data["edge_prob"]).astype(np.float64)
    if not (len(src) == len(tgt) == len(prob)):
        raise Reject(f"{crop}: candidate surface arrays disagree in length")

    return {
        "crop": crop,
        "embryo": crop.split("_")[0],
        "nodes": int(len(coords)),
        "frames": [int(t) for t in frames.tolist()],
        "feature_frames": sorted(feat_frames),
        "feature_dim": int(np.asarray(data[f"feat_{feat_frames[0]}"]).shape[1]) if feat_frames else 0,
        "node_order": {
            "convention": "positional index into coords_so_far, blocks in increasing frame order",
            "partition_contiguous_and_increasing": order_ok,
            "coord_feature_pair_digest": pair_digest,
            "coords_digest": coords_digest,
            "features_digest": features_digest,
        },
        "bands": {
            "deployed_floor": deployed_floor,
            "above_floor": int((prob > deployed_floor).sum()),
            "sub_floor": int((prob <= deployed_floor).sum()),
            "total": int(len(prob)),
            "min_prob": float(prob.min()) if len(prob) else None,
            "max_prob": float(prob.max()) if len(prob) else None,
            "surface_digest": _digest(src, tgt, prob),
        },
        "feature_stats": _feature_stats(data, feat_frames),
    }


def _feature_stats(data: dict, feat_frames: list[int]) -> dict:
    if not feat_frames:
        return {"dtype": None, "min": None, "max": None, "mean": None}
    stacked = np.concatenate([np.asarray(data[f"feat_{t}"]) for t in sorted(feat_frames)])
    return {
        "dtype": str(stacked.dtype),
        "min": float(stacked.min()),
        "max": float(stacked.max()),
        "mean": float(stacked.mean()),
        "abs_max": float(np.abs(stacked).max()),
    }


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except Exception:                                            # pragma: no cover - no git
        return None


# --------------------------------------------------------------------------------------------
# bind
# --------------------------------------------------------------------------------------------
def build_manifest(args) -> dict:
    cache_dir = Path(args.cache_dir)
    caches = sorted(cache_dir.glob("*.npz"))
    if not caches:
        raise Reject(f"{cache_dir} holds no *.npz cache - there is nothing to bind")

    trunk = Path(args.trunk)
    if not trunk.is_file():
        raise Reject(f"trunk checkpoint {trunk} is not on disk; its identity cannot be recorded")
    if args.trunk_role not in TRUNK_ROLES:
        raise Reject(f"unknown trunk role {args.trunk_role!r}; known: {sorted(TRUNK_ROLES)}")
    fold = str(args.fold).strip()
    if fold not in FOLD_EMBRYO:
        raise Reject(f"fold must be one of {sorted(FOLD_EMBRYO)}, got {fold!r}")

    crops = []
    for path in caches:
        crops.append(crop_binding(path.stem, load_cache(path), args.deployed_floor))

    notebook = Path(args.notebook) if args.notebook else None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "assoc_feature_cache_manifest",
        "packet": "PKT-0037",
        "cache_dir": str(cache_dir),
        "trunk": {
            "role": args.trunk_role,
            "path": str(trunk),
            "sha256": sha256_file(trunk),
            "bytes": trunk.stat().st_size,
            "provenance": args.trunk_provenance,
            # The cache-intrinsic half of the binding. A cache swapped between two trunk
            # directories keeps its manifest's trunk sha256 and stops matching this.
            "features_digest": _digest(
                *[np.frombuffer(bytes.fromhex(c["node_order"]["coord_feature_pair_digest"]),
                                dtype=np.uint8) for c in crops]
            ),
        },
        "fold": {
            "fold": fold,
            "held_out_embryo": FOLD_EMBRYO[fold],
            "weights_glob": args.weights_glob,
            "crops": [c["crop"] for c in crops],
        },
        "feature_normalisation": {
            "transform": args.feature_normalisation,
            "applied_after": "model._index_features(unet_out[:, f_idx], coords, mask)",
            "note": (
                "'none' is the deployed contract: the head consumes _index_features output "
                "directly. Any other value must name code, because a head trained on a "
                "differently scaled cache is the wrong-features branch of an uninterpretable null."
            ),
        },
        "candidate_graph": {
            "rule": args.candidate_rule,
            "deployed_floor": args.deployed_floor,
            "acquisition_floor": args.acquisition_floor,
            "rank_cap": args.rank_cap,
            "gate_um": args.gate_um,
            "det_threshold": args.det_threshold,
            "pool_kernel_um": args.pool_kernel_um,
            "softmax_axis": args.softmax_axis,
            "abstain_mass": bool(args.abstain_mass),
        },
        "provenance": {
            "notebook": str(notebook) if notebook else None,
            "notebook_sha256": sha256_file(notebook) if notebook and notebook.is_file() else None,
            "spec": args.spec,
            "kernel": args.kernel,
            "kernel_version": args.kernel_version,
            "source_commit": args.commit or git_commit(Path(__file__).resolve().parents[2]),
        },
        "crops": crops,
    }
    return manifest


# --------------------------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------------------------
def audit(cache_dir: Path, manifest_path: Path, *, trunk: Path | None = None,
          expect_trunk_sha: str | None = None, expect_fold: str | None = None,
          expect_role: str | None = None) -> dict:
    """Re-derive the binding from what is on disk. Returns a report; raises Reject on failure."""
    if not manifest_path.is_file():
        raise Reject(
            f"no manifest at {manifest_path}. An unbound cache cannot distinguish 'the head does "
            "not work' from 'we fed it the wrong features', so it is not licensed for training."
        )
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("schema_version") != SCHEMA_VERSION:
        raise Reject(f"manifest schema {m.get('schema_version')!r} != {SCHEMA_VERSION}")

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            raise Reject(f"{name}: {detail}")

    # --- 1. TRUNK -----------------------------------------------------------------------
    trunk_path = Path(trunk) if trunk else Path(m["trunk"]["path"])
    if trunk_path.is_file():
        actual = sha256_file(trunk_path)
        check("trunk_checkpoint_sha256", actual == m["trunk"]["sha256"],
              f"{trunk_path} hashes {actual[:16]}..., manifest records "
              f"{m['trunk']['sha256'][:16]}...")
    else:
        check("trunk_checkpoint_present", False,
              f"trunk {trunk_path} is not on disk, so the cache's producing trunk cannot be "
              "re-verified - FACT-0392's third risk is exactly this")

    check("trunk_role_known", m["trunk"]["role"] in TRUNK_ROLES,
          f"role {m['trunk']['role']!r}")
    if expect_role:
        check("trunk_role_matches_expectation", m["trunk"]["role"] == expect_role,
              f"cache was produced by {m['trunk']['role']!r}, consumer expects {expect_role!r}")
    if expect_trunk_sha:
        check("trunk_sha_matches_expectation", m["trunk"]["sha256"] == expect_trunk_sha,
              f"cache trunk {m['trunk']['sha256'][:16]}..., consumer expects "
              f"{expect_trunk_sha[:16]}...")

    # --- 2. FOLD, EMBRYO AND WEIGHTS ------------------------------------------------------
    fold = str(m["fold"]["fold"])
    check("fold_known", fold in FOLD_EMBRYO, f"fold {fold!r}")
    expected_embryo = FOLD_EMBRYO[fold]
    check("fold_declares_the_right_embryo", m["fold"]["held_out_embryo"] == expected_embryo,
          f"fold {fold} holds out {expected_embryo}, manifest says "
          f"{m['fold']['held_out_embryo']!r}")
    if expect_fold is not None:
        check("fold_matches_expectation", fold == str(expect_fold),
              f"cache is fold {fold}, consumer expects fold {expect_fold}")

    glob = m["fold"].get("weights_glob") or ""
    if fold == "1":
        check("fold1_does_not_use_the_leaky_pack_weights",
              bool(glob) and PACK_WEIGHTS_TOKEN not in glob,
              f"fold 1 weights_glob {glob!r} - the pack ships only {PACK_WEIGHTS_TOKEN}, which "
              "was TRAINED ON 6bba, the very embryo fold 1 holds out. This is the EXP-0019 defect")
    else:
        check("fold0_does_not_load_split1_weights", "split_1" not in glob,
              f"fold 0 weights_glob {glob!r} loads split_1")

    # --- 3. THE CACHE ON DISK STILL IS THE CACHE THAT WAS BOUND ---------------------------
    on_disk = {p.stem: p for p in sorted(cache_dir.glob("*.npz"))}
    recorded = {c["crop"]: c for c in m["crops"]}
    check("crop_sets_agree", set(on_disk) == set(recorded),
          f"on disk {sorted(set(on_disk) - set(recorded))} unbound, manifest "
          f"{sorted(set(recorded) - set(on_disk))} absent")
    check("cache_is_not_empty", bool(on_disk), "no crop caches on disk")

    pair_digests = []
    for crop in sorted(on_disk):
        rec = recorded[crop]
        got = crop_binding(crop, load_cache(on_disk[crop]),
                           m["candidate_graph"]["deployed_floor"])
        pair_digests.append(got["node_order"]["coord_feature_pair_digest"])

        check(f"{crop}:embryo_matches_fold", got["embryo"] == expected_embryo,
              f"crop {crop} is embryo {got['embryo']}, but fold {fold} evaluates "
              f"{expected_embryo}")
        check(f"{crop}:node_ordering_is_a_contiguous_increasing_partition",
              got["node_order"]["partition_contiguous_and_increasing"],
              "node ids are positional indices into coords_so_far; a non-contiguous or "
              "out-of-order frame partition makes every id ambiguous")
        check(f"{crop}:node_order_unchanged",
              got["node_order"]["coords_digest"] == rec["node_order"]["coords_digest"],
              "the coordinate rows no longer digest to the bound value - the NODES HAVE BEEN "
              "REORDERED, so every positional id recorded against this crop (pre-ILP export, "
              "candidate surface, any trained head) now names a different cell")
        check(f"{crop}:features_unchanged",
              got["node_order"]["features_digest"] == rec["node_order"]["features_digest"],
              "the same nodes carry DIFFERENT FEATURE VALUES than were bound. On an unchanged "
              "node set that is a different producing trunk - FACT-0392's confound, where "
              "official-trunk features fed to a StableDet-trained head look exactly like a "
              "wrong feature contract")
        check(f"{crop}:candidate_surface_unchanged",
              got["bands"]["surface_digest"] == rec["bands"]["surface_digest"],
              "the candidate surface differs from the one that was bound")
        check(f"{crop}:node_count_unchanged", got["nodes"] == rec["nodes"],
              f"{got['nodes']} nodes on disk, {rec['nodes']} bound")

        # --- 4. NO EMPTY PROBABILITY BAND -------------------------------------------------
        check(f"{crop}:deployed_band_is_not_empty", got["bands"]["above_floor"] > 0,
              "zero pairs above the deployed floor: nothing in this crop can be checked "
              "against the deployed surface, so a pass here certifies nothing")
        check(f"{crop}:learnable_band_is_not_empty", got["bands"]["sub_floor"] > 0,
              "zero pairs at or below the deployed floor. FACT-0382 measured that ALL 691 "
              "fold-0 contested errors have their true parent below that floor, so a cache "
              "with an empty sub-threshold band holds none of the learnable population")

    # --- 5. THE TRUNK <-> FEATURES BINDING ------------------------------------------------
    features_digest = _digest(*[np.frombuffer(bytes.fromhex(d), dtype=np.uint8)
                                for d in pair_digests])
    check("features_still_bound_to_the_recorded_trunk",
          features_digest == m["trunk"]["features_digest"],
          "the features on disk do not digest to the value bound against trunk "
          f"{m['trunk']['role']} ({m['trunk']['sha256'][:16]}...). Either the cache was "
          "swapped between trunk directories or it was rewritten without rebinding - both are "
          "FACT-0392's confound, where an official-trunk cache fed to a StableDet-trained head "
          "looks exactly like a wrong feature contract")

    # --- 6. PROVENANCE COMPLETENESS -------------------------------------------------------
    prov = m.get("provenance", {})
    for field in ("notebook", "spec", "source_commit"):
        check(f"provenance_declares_{field}", bool(prov.get(field)),
              f"provenance.{field} is empty; the run that produced this cache is not identifiable")
    check("feature_normalisation_declared", bool(m["feature_normalisation"]["transform"]),
          "feature_normalisation.transform is empty")
    check("candidate_rule_declared", bool(m["candidate_graph"]["rule"]),
          "candidate_graph.rule is empty")

    return {"passed": True, "checks": checks, "crops": len(on_disk), "fold": fold,
            "trunk_role": m["trunk"]["role"], "trunk_sha256": m["trunk"]["sha256"]}


def audit_dual_trunk(dir_a: Path, dir_b: Path, man_a: Path | None = None,
                     man_b: Path | None = None) -> dict:
    """Check that two sibling caches form a VALID DUAL-TRUNK PAIR.

    FACT-0392's third risk is that trunk identity is a command-line argument recorded nowhere in
    the artifact, so an official-trunk cache fed to a StableDet-trained head looks exactly like a
    wrong feature contract. The remedy is to cache both trunks and compare them - but only if the
    pair is well formed, and "well formed" has a precise, checkable meaning:

        SAME node set        identical coordinate digests, crop for crop. Both trunks must index
                             ONE detector pass; otherwise trunk identity is confounded with node
                             identity and the pair cannot separate the two.
        DIFFERENT features   the feature digests must differ. Two caches that agree here are not
                             two trunks - they are one trunk written twice, and a null measured
                             against them would be a null about nothing.
        SAME fold            comparing trunks across folds re-introduces the embryo confound the
                             contract already forbids pooling away.
        DIFFERENT trunks     distinct checkpoint hashes and distinct declared roles.

    Both caches must independently pass `audit` first. A pair of broken caches is not a pair.
    """
    man_a = man_a or dir_a / "cache_manifest.json"
    man_b = man_b or dir_b / "cache_manifest.json"
    rep_a, rep_b = audit(dir_a, man_a), audit(dir_b, man_b)
    ma = json.loads(man_a.read_text(encoding="utf-8"))
    mb = json.loads(man_b.read_text(encoding="utf-8"))

    if ma["trunk"]["sha256"] == mb["trunk"]["sha256"]:
        raise Reject(
            "dual_trunk_checkpoints_differ: both caches name the same trunk checkpoint, so this "
            "is one trunk written twice and resolves nothing about FACT-0392's ambiguity"
        )
    if ma["trunk"]["role"] == mb["trunk"]["role"]:
        raise Reject(f"dual_trunk_roles_differ: both caches declare role {ma['trunk']['role']!r}")
    if str(ma["fold"]["fold"]) != str(mb["fold"]["fold"]):
        raise Reject(
            f"dual_trunk_same_fold: fold {ma['fold']['fold']} against fold {mb['fold']['fold']}. "
            "Comparing trunks across folds confounds trunk identity with the embryo direction, "
            "which the contract requires be reported separately in the first place"
        )

    a_crops = {c["crop"]: c for c in ma["crops"]}
    b_crops = {c["crop"]: c for c in mb["crops"]}
    if set(a_crops) != set(b_crops):
        raise Reject(
            "dual_trunk_same_crops: the two caches cover different crops "
            f"({sorted(set(a_crops) ^ set(b_crops))[:5]}), so any difference between them is "
            "confounded with which crops each one saw"
        )
    for crop in sorted(a_crops):
        an, bn = a_crops[crop]["node_order"], b_crops[crop]["node_order"]
        if an["coords_digest"] != bn["coords_digest"]:
            raise Reject(
                f"dual_trunk_shares_the_node_set: {crop} has different coordinates in the two "
                "caches. Both trunks must index ONE detector pass - otherwise trunk identity is "
                "confounded with node identity and the pair cannot separate them"
            )
        if an["features_digest"] == bn["features_digest"]:
            raise Reject(
                f"dual_trunk_features_differ: {crop} has byte-identical features under two "
                "different checkpoints. Either the second trunk was never loaded or the cache "
                "was copied - either way the pair proves nothing"
            )
    return {"passed": True, "fold": str(ma["fold"]["fold"]),
            "roles": [ma["trunk"]["role"], mb["trunk"]["role"]],
            "crops": len(a_crops), "a": rep_a["trunk_sha256"], "b": rep_b["trunk_sha256"]}


# --------------------------------------------------------------------------------------------
# self-test: the four mutations, manufactured
# --------------------------------------------------------------------------------------------
def _synth_cache(path: Path, *, crop: str, trunk_seed: int, n_frames: int = 3,
                 n_per_frame: int = 4, feat_dim: int = 32, reorder: bool = False,
                 empty_sub_band: bool = False) -> None:
    # The coordinate grid is a property of the DETECTOR and is identical across trunks in this
    # fixture; only the 32-dim features depend on `trunk_seed`. That is deliberate: it makes the
    # swapped-trunk mutation a pure feature change on an unchanged node set, which is exactly
    # the shape FACT-0392 warns about and the shape a node-count or shape check cannot see.
    rng = np.random.default_rng(trunk_seed)
    coords, starts, ends, feats = [], [], [], {}
    total = 0
    for t in range(n_frames):
        arr = np.stack([np.full(n_per_frame, t), np.arange(n_per_frame),
                        np.arange(n_per_frame) * 2, np.arange(n_per_frame) * 3], axis=1)
        starts.append(total); total += n_per_frame; ends.append(total)
        coords.append(arr.astype(np.int16))
        feats[t] = rng.normal(size=(n_per_frame, feat_dim)).astype(np.float32)
    coords_arr = np.concatenate(coords)

    if reorder:
        # A REORDER that leaves the frame partition, the node COUNT and every feature value
        # untouched: within the first frame, reverse the coordinate rows. Nothing about the
        # file's shape changes - only which cell each positional id names.
        block = coords_arr[starts[0]:ends[0]][::-1].copy()
        coords_arr = coords_arr.copy()
        coords_arr[starts[0]:ends[0]] = block

    src, tgt, prob = [], [], []
    for t in range(n_frames - 1):
        for i in range(n_per_frame):
            for j in range(n_per_frame):
                src.append(starts[t] + i); tgt.append(starts[t + 1] + j)
                prob.append(0.9 if i == j else 0.05)
    prob_arr = np.asarray(prob, dtype=np.float32)
    if empty_sub_band:
        prob_arr = np.full_like(prob_arr, 0.9)     # every pair above the deployed floor

    np.savez_compressed(
        path,
        coords=coords_arr,
        frames=np.arange(n_frames, dtype=np.int64),
        starts=np.asarray(starts, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
        feat_frames=np.arange(n_frames, dtype=np.int64),
        source_id=np.asarray(src, dtype=np.int64),
        target_id=np.asarray(tgt, dtype=np.int64),
        edge_prob=prob_arr,
        **{f"feat_{t}": v for t, v in feats.items()},
    )


def _bind_args(cache_dir: Path, trunk: Path, *, fold: str, role: str, weights_glob: str,
               notebook: Path):
    ns = argparse.Namespace(
        cache_dir=str(cache_dir), trunk=str(trunk), trunk_role=role,
        trunk_provenance="synthetic self-test trunk", fold=fold, weights_glob=weights_glob,
        feature_normalisation="none", candidate_rule="softmax over source axis, floor 0.1, cap 4",
        deployed_floor=0.5, acquisition_floor=0.1, rank_cap=4, gate_um=None,
        det_threshold=0.96875, pool_kernel_um=3.0, softmax_axis="source", abstain_mass=False,
        notebook=str(notebook), spec="selftest_spec", kernel=None, kernel_version=None,
        commit="0" * 40,
    )
    return ns


def self_test(out: Path | None) -> int:
    results = []

    def record(name: str, expect_reject: bool, fn) -> None:
        try:
            fn()
            outcome, detail = "accepted", ""
        except Reject as err:
            outcome, detail = "rejected", str(err)
        ok = (outcome == "rejected") if expect_reject else (outcome == "accepted")
        results.append({"mutation": name, "expected": "reject" if expect_reject else "accept",
                        "outcome": outcome, "ok": ok, "detail": detail[:300]})
        print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} expected="
              f"{'reject' if expect_reject else 'accept':8s} got={outcome}")
        if detail and expect_reject:
            print(f"          -> {detail[:200]}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        nb = root / "notebook.ipynb"; nb.write_text('{"cells": []}', encoding="utf-8")
        trunk_a = root / "trunk_official.pth"; trunk_a.write_bytes(b"OFFICIAL-TRUNK-BYTES" * 64)
        trunk_b = root / "trunk_stabledet.pth"; trunk_b.write_bytes(b"STABLEDET-TRUNK-BYTE" * 64)

        clean = root / "clean"; clean.mkdir()
        _synth_cache(clean / "44b6_aaaaaaaa.npz", crop="44b6_aaaaaaaa", trunk_seed=1)
        man = root / "clean" / "cache_manifest.json"
        args = _bind_args(clean, trunk_a, fold="0", role="official",
                          weights_glob="/kaggle/input/*/split_0/edge_predictor_best.pth",
                          notebook=nb)
        man.write_text(json.dumps(build_manifest(args), indent=2), encoding="utf-8")

        print("\nSELF-TEST: the auditor must accept the clean cache and reject four mutations\n")
        record("control_clean_cache", False, lambda: audit(clean, man))

        # (1) SWAPPED TRUNK. Same manifest; the cache is re-produced by the other trunk. This is
        #     FACT-0392's third risk in its realistic accidental form: two trunk caches in one
        #     session and the wrong directory consumed.
        swapped = root / "swapped"; swapped.mkdir()
        _synth_cache(swapped / "44b6_aaaaaaaa.npz", crop="44b6_aaaaaaaa", trunk_seed=2)
        man_sw = swapped / "cache_manifest.json"
        man_sw.write_text(man.read_text(encoding="utf-8"), encoding="utf-8")
        record("swapped_trunk_features", True, lambda: audit(swapped, man_sw))

        # (1b) and the other direction: the manifest points at a different checkpoint file.
        man_sw2 = root / "clean" / "manifest_wrong_trunk.json"
        m2 = json.loads(man.read_text(encoding="utf-8"))
        m2["trunk"]["path"] = str(trunk_b)
        man_sw2.write_text(json.dumps(m2), encoding="utf-8")
        record("swapped_trunk_checkpoint", True, lambda: audit(clean, man_sw2))

        # (2) REORDERED NODES. Identical shapes, identical node count, identical frame partition;
        #     only the correspondence between coordinates and feature rows is permuted.
        reordered = root / "reordered"; reordered.mkdir()
        _synth_cache(reordered / "44b6_aaaaaaaa.npz", crop="44b6_aaaaaaaa", trunk_seed=1,
                     reorder=True)
        man_ro = reordered / "cache_manifest.json"
        man_ro.write_text(man.read_text(encoding="utf-8"), encoding="utf-8")
        record("reordered_nodes", True, lambda: audit(reordered, man_ro))

        # (3) EMPTY PROBABILITY BAND. Every pair above the deployed floor, so the sub-0.5 band -
        #     where FACT-0382 puts the whole learnable population - holds nothing.
        empty = root / "empty_band"; empty.mkdir()
        _synth_cache(empty / "44b6_aaaaaaaa.npz", crop="44b6_aaaaaaaa", trunk_seed=1,
                     empty_sub_band=True)
        args_e = _bind_args(empty, trunk_a, fold="0", role="official",
                            weights_glob="/kaggle/input/*/split_0/edge_predictor_best.pth",
                            notebook=nb)
        man_e = empty / "cache_manifest.json"
        man_e.write_text(json.dumps(build_manifest(args_e), indent=2), encoding="utf-8")
        record("empty_probability_band", True, lambda: audit(empty, man_e))

        # (4) WRONG-FOLD WEIGHTS. A fold-1 cache carrying the pack's split_0 glob - the EXP-0019
        #     defect - and, separately, a fold-1 declaration over fold-0 embryo crops.
        f1 = root / "fold1"; f1.mkdir()
        _synth_cache(f1 / "6bba_bbbbbbbb.npz", crop="6bba_bbbbbbbb", trunk_seed=1)
        args_f1 = _bind_args(f1, trunk_a, fold="1", role="oof_split1",
                             weights_glob="/kaggle/input/*/split_0/edge_predictor_best.pth",
                             notebook=nb)
        man_f1 = f1 / "cache_manifest.json"
        man_f1.write_text(json.dumps(build_manifest(args_f1), indent=2), encoding="utf-8")
        record("wrong_fold_weights_split0_on_fold1", True, lambda: audit(f1, man_f1))

        man_f1b = f1 / "manifest_wrong_embryo.json"
        args_f1b = _bind_args(f1, trunk_a, fold="1", role="oof_split1",
                              weights_glob="/kaggle/input/*/edge_predictor_best_split_1.pth",
                              notebook=nb)
        good_f1 = build_manifest(args_f1b)
        man_f1b.write_text(json.dumps(good_f1), encoding="utf-8")
        record("control_correct_fold1_cache", False, lambda: audit(f1, man_f1b))

        # Declared fold 0, with a glob that is perfectly legal for fold 0, over 6bba crops. The
        # weights check cannot fire here; only the fold<->embryo binding can.
        mismatch = json.loads(json.dumps(good_f1))
        mismatch["fold"]["fold"] = "0"
        mismatch["fold"]["held_out_embryo"] = "44b6"
        mismatch["fold"]["weights_glob"] = "/kaggle/input/*/split_0/edge_predictor_best.pth"
        man_f1c = f1 / "manifest_fold0_over_6bba.json"
        man_f1c.write_text(json.dumps(mismatch), encoding="utf-8")
        record("wrong_fold_declared_over_the_other_embryo", True, lambda: audit(f1, man_f1c))

        # (5) and the floor case that started this packet: no manifest at all.
        record("no_manifest_at_all", True,
               lambda: audit(clean, root / "does_not_exist.json"))

        # (6) THE DUAL-TRUNK PAIR (FACT-0392 risk three). A well-formed pair shares the node set
        #     and differs in features. The two degenerate shapes are one trunk written twice, and
        #     two trunks that were given different node sets.
        pair_b = root / "pair_b"; pair_b.mkdir()
        _synth_cache(pair_b / "44b6_aaaaaaaa.npz", crop="44b6_aaaaaaaa", trunk_seed=7)
        args_b = _bind_args(pair_b, trunk_b, fold="0", role="stabledet",
                            weights_glob="/kaggle/input/*/split_0/edge_predictor_best.pth",
                            notebook=nb)
        (pair_b / "cache_manifest.json").write_text(
            json.dumps(build_manifest(args_b), indent=2), encoding="utf-8")
        record("control_valid_dual_trunk_pair", False, lambda: audit_dual_trunk(clean, pair_b))

        same = root / "pair_same"; same.mkdir()
        _synth_cache(same / "44b6_aaaaaaaa.npz", crop="44b6_aaaaaaaa", trunk_seed=1)
        args_s = _bind_args(same, trunk_b, fold="0", role="stabledet",
                            weights_glob="/kaggle/input/*/split_0/edge_predictor_best.pth",
                            notebook=nb)
        (same / "cache_manifest.json").write_text(
            json.dumps(build_manifest(args_s), indent=2), encoding="utf-8")
        record("dual_trunk_one_trunk_written_twice", True, lambda: audit_dual_trunk(clean, same))

        diffnodes = root / "pair_diffnodes"; diffnodes.mkdir()
        _synth_cache(diffnodes / "44b6_aaaaaaaa.npz", crop="44b6_aaaaaaaa", trunk_seed=7,
                     n_per_frame=5)
        args_d = _bind_args(diffnodes, trunk_b, fold="0", role="stabledet",
                            weights_glob="/kaggle/input/*/split_0/edge_predictor_best.pth",
                            notebook=nb)
        (diffnodes / "cache_manifest.json").write_text(
            json.dumps(build_manifest(args_d), indent=2), encoding="utf-8")
        record("dual_trunk_node_sets_disagree", True, lambda: audit_dual_trunk(clean, diffnodes))

    passed = all(r["ok"] for r in results)
    payload = {"schema_version": SCHEMA_VERSION, "self_test": "audit_feature_cache",
               "all_passed": passed, "results": results}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSELF-TEST {'PASSED' if passed else 'FAILED'} - "
          f"{sum(r['ok'] for r in results)}/{len(results)} behaved as required")
    return 0 if passed else 1


# --------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bind", help="write a manifest that binds a cache to its producer")
    b.add_argument("--cache-dir", required=True)
    b.add_argument("--manifest", required=True)
    b.add_argument("--trunk", required=True)
    b.add_argument("--trunk-role", required=True, choices=sorted(TRUNK_ROLES))
    b.add_argument("--trunk-provenance", required=True)
    b.add_argument("--fold", required=True, choices=sorted(FOLD_EMBRYO))
    b.add_argument("--weights-glob", default="")
    b.add_argument("--feature-normalisation", default="none")
    b.add_argument("--candidate-rule", required=True)
    b.add_argument("--deployed-floor", type=float, default=0.5)
    b.add_argument("--acquisition-floor", type=float, default=0.1)
    b.add_argument("--rank-cap", type=int, default=4)
    b.add_argument("--gate-um", type=float)
    b.add_argument("--det-threshold", type=float, default=0.96875)
    b.add_argument("--pool-kernel-um", type=float, default=3.0)
    b.add_argument("--softmax-axis", default="source")
    b.add_argument("--abstain-mass", action="store_true")
    b.add_argument("--notebook")
    b.add_argument("--spec")
    b.add_argument("--kernel")
    b.add_argument("--kernel-version")
    b.add_argument("--commit")

    a = sub.add_parser("audit", help="independently re-check a cache against its manifest")
    a.add_argument("--cache-dir", required=True)
    a.add_argument("--manifest")
    a.add_argument("--trunk")
    a.add_argument("--expect-trunk-sha256")
    a.add_argument("--expect-fold", choices=sorted(FOLD_EMBRYO))
    a.add_argument("--expect-role", choices=sorted(TRUNK_ROLES))
    a.add_argument("--out")

    d = sub.add_parser("audit-pair", help="check two caches form a valid dual-trunk pair")
    d.add_argument("--cache-a", required=True)
    d.add_argument("--cache-b", required=True)
    d.add_argument("--manifest-a")
    d.add_argument("--manifest-b")
    d.add_argument("--out")

    s = sub.add_parser("self-test", help="prove the auditor rejects each manufactured defect")
    s.add_argument("--out")

    args = ap.parse_args()

    if args.cmd == "self-test":
        return self_test(Path(args.out) if args.out else None)

    if args.cmd == "audit-pair":
        try:
            report = audit_dual_trunk(
                Path(args.cache_a), Path(args.cache_b),
                Path(args.manifest_a) if args.manifest_a else None,
                Path(args.manifest_b) if args.manifest_b else None)
        except Reject as err:
            if args.out:
                Path(args.out).write_text(
                    json.dumps({"passed": False, "reject": str(err)}, indent=2), encoding="utf-8")
            print("DUAL-TRUNK PAIR REJECTED", file=sys.stderr)
            print(f"  {err}", file=sys.stderr)
            return 1
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"DUAL-TRUNK PAIR OK  fold={report['fold']} roles={report['roles']} "
              f"crops={report['crops']}")
        return 0

    if args.cmd == "bind":
        manifest = build_manifest(args)
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"BOUND {args.cache_dir} -> {args.manifest}\n"
              f"  trunk {manifest['trunk']['role']} {manifest['trunk']['sha256'][:16]}...\n"
              f"  fold {manifest['fold']['fold']} ({manifest['fold']['held_out_embryo']}), "
              f"{len(manifest['crops'])} crops")
        return 0

    cache_dir = Path(args.cache_dir)
    manifest = Path(args.manifest) if args.manifest else cache_dir / "cache_manifest.json"
    try:
        report = audit(cache_dir, manifest,
                       trunk=Path(args.trunk) if args.trunk else None,
                       expect_trunk_sha=args.expect_trunk_sha256,
                       expect_fold=args.expect_fold, expect_role=args.expect_role)
    except Reject as err:
        payload = {"passed": False, "reject": str(err)}
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"CACHE AUDIT REJECTED\n  {err}", file=sys.stderr)
        return 1
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CACHE AUDIT PASSED  fold={report['fold']} trunk={report['trunk_role']} "
          f"crops={report['crops']} checks={len(report['checks'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
