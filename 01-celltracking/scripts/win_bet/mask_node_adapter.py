r"""THE NUCLEI-MASK -> NODE ADAPTER, and the tiny node/mask evaluation fixture (PKT-0048 task 2).

WHAT IT IS FOR
--------------
``FACT-0447``'s scope correction leaves exactly one detector-shaped route open: a detector that
produces new FEATURES, MASKS, CANDIDATES and ASSOCIATIONS jointly. The node-only thesis is dead
permanently and nothing here reopens it. What a mask source buys is the FIRST factor of the
winning function - the 14 of 19 HOCT input slots that ``hoct_feature_contract.py`` currently
refuses (``FACT-0454``, corrected from 15). This module is the piece that turns instance masks
into those slots, and the fixture is what says whether the instances are any good BEFORE a GPU
is spent finding out.

THE THREE THINGS IT REFUSES TO DO
---------------------------------
1.  It will not load FOCUS-3D weights without asserting BOTH key counts.
    ``FACT-0425``: the publisher's loader is ``load_state_dict(..., strict=False)`` at
    ``inference_win.py:141`` with its missing/unexpected prints COMMENTED OUT at ``:143-149``.
    A key mismatch runs on random weights and looks exactly like "the foreign model does not
    join". ``load_focus3d_strict`` fails closed on an absent file, on a nonzero missing-key
    count and on a nonzero unexpected-key count, separately, each with its own reason.
2.  It will not assume isotropy. Centroids are produced in a NAMED unit convention taken from
    ``hoct_scale_gate.CONVENTIONS``; there is no default.
3.  It will not report a number computed from ``general_v1`` without ``FACT-0451`` attached -
    every payload carries ``offline_scoreable: false`` mechanically.

THE FIXTURE DIFFERS FROM PRODUCTION IN COST, NEVER IN KIND
-----------------------------------------------------------
It reduces exactly one thing - the NUMBER OF CROPS - and it says so in its own payload under
``reduction``. Everything that decides a verdict is imported from the production path rather
than restated: the matcher, the 7 um threshold and the ``(1.625, 0.40625, 0.40625)`` scale all
come from ``ceiling_ladder`` / ``detpeak_curve``, which is the same code that produced
``FACT-0368``, ``FACT-0371`` and ``FACT-0447``.

AND IT NAMES THE TRAP IT WAS BUILT TO AVOID. The deployed detector runs on a ``(1, 4, 4)``
downsampled grid (``detpeak_curve.DOWNSAMPLE``, from ``predict_unet_transformer.py:157``).
``(1.625, 0.40625, 0.40625) * (1, 4, 4) = (1.625, 1.625, 1.625)``: **at that one downsample the
voxel is isotropic**, so a fixture that works in the detector's own grid can assume isotropy and
be right there and wrong everywhere else. A mask source works at FULL RESOLUTION and inherits
none of that. ``assert_not_the_identity_downsample()`` is executed, not remembered.

THE INSTRUMENT CALIBRATES BEFORE IT REPORTS
--------------------------------------------
Two arms run before any new number is read, in the ``revladder`` pattern:
  ``gt_identity``   GT nodes offered as the candidate set. Must return recall 1.0 and residual
                    0.0, or the fixture is measuring itself wrong.
  ``deployed``      the shipped champion node set. Must reproduce the registry - node recall
                    ``FACT-0447`` (0.98679 f0 / 0.86357 f1) and the division residual share
                    ``FACT-0380`` (0.455 f0 / 0.746 f1).
A third arm, ``focus3d``, is RECORDED AS REFUSED rather than skipped, because a silently missing
arm is invisible in a panel.

USAGE
-----
    python scripts/win_bet/mask_node_adapter.py fixture --fold 0 --max-crops 12 --out <json>
    python scripts/win_bet/mask_node_adapter.py adapter-selftest --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# PRODUCTION CONSTANTS AND THE PRODUCTION MATCHER - imported, never restated. This is the
# "cost, not kind" guarantee expressed as an import rather than as a comment.
from ceiling_ladder import SCALE, match_gt_to_pred  # noqa: E402
from detpeak_curve import DOWNSAMPLE, MAX_DISTANCE_UM, SCALE_UM  # noqa: E402
from hoct_feature_contract import FeatureRefusal, audit_matrix, build_node_feats  # noqa: E402
from hoct_scale_gate import CONVENTIONS  # noqa: E402

HEARTBEAT_OK = "MASK_NODE_FIXTURE_COMPLETE"
HEARTBEAT_REFUSED = "MASK_NODE_FIXTURE_REFUSED"

FOLDS = {
    0: {"prefix": "44b6", "slices": Path(r"C:\temp\revladder\slices"), "n_crops": 71,
        "node_recall_anchor": 0.98679, "div_share_anchor": 0.455,
        "unmatched_gt_anchor": 223, "n_pred_anchor": 1905264},
    1: {"prefix": "6bba", "slices": Path(r"C:\temp\revladder\slices"), "n_crops": 128,
        "node_recall_anchor": 0.86357, "div_share_anchor": 0.746,
        "unmatched_gt_anchor": 14970, "n_pred_anchor": 1931622},
}

DIVISION_RESIDUAL_UM = 3.0    # FACT-0380's threshold, inside the 7 um matcher
FOCUS3D_WEIGHTS_ENV = "FOCUS3D_WEIGHTS"


# ---------------------------------------------------------------------------------------------
# 0. The guard the fixture exists to carry
# ---------------------------------------------------------------------------------------------
def assert_not_the_identity_downsample() -> dict:
    """The deployed detector's grid is isotropic in microns. A mask source's is not.

    Executed so the fixture cannot inherit an assumption that happens to be true only on the
    surface the deployed detector runs on.
    """
    ds = np.asarray(DOWNSAMPLE, dtype=np.float64)
    scale = np.asarray(SCALE_UM, dtype=np.float64)
    detector_grid_um = scale * ds
    isotropic_there = bool(np.allclose(detector_grid_um, detector_grid_um[0]))
    isotropic_at_full_res = bool(np.allclose(scale, scale[0]))
    return {
        "deployed_downsample_zyx": ds.tolist(),
        "full_res_um_per_voxel_zyx": scale.tolist(),
        "detector_grid_um_per_voxel_zyx": detector_grid_um.tolist(),
        "detector_grid_is_isotropic": isotropic_there,
        "full_res_grid_is_isotropic": isotropic_at_full_res,
        "passes": bool(isotropic_there and not isotropic_at_full_res),
        "why_this_is_checked": "(1.625, 0.40625, 0.40625) * (1, 4, 4) = (1.625, 1.625, 1.625). "
                               "The deployed detector's own grid is the ONE downsample at which "
                               "isotropy is true. A mask adapter runs at full resolution and "
                               "must not inherit it",
    }


# ---------------------------------------------------------------------------------------------
# 1. THE LOADER GUARD - FACT-0425, fail-closed on three separate grounds
# ---------------------------------------------------------------------------------------------
class LoaderRefusal(RuntimeError):
    pass


def load_focus3d_strict(model, weights: Path | None, *, allow_missing: int = 0,
                        allow_unexpected: int = 0) -> dict:
    """Load FOCUS-3D weights and REFUSE unless both key counts clear their budgets.

    The publisher's ``build_predictor`` (``inference_win.py:131-152``) calls
    ``load_state_dict(state_dict, strict=False)`` and has its diagnostics commented out, so a
    complete key mismatch produces a model that runs on its random initialisation. This wrapper
    turns each of the three failure modes into a distinct refusal:

      * the checkpoint is absent           -> LoaderRefusal("weights_absent")
      * missing keys exceed the budget     -> LoaderRefusal("missing_keys")
      * unexpected keys exceed the budget  -> LoaderRefusal("unexpected_keys")

    ``allow_missing``/``allow_unexpected`` default to ZERO. A caller who needs a nonzero budget
    must say so in the call, where it is visible in a diff.
    """
    if weights is None or not Path(weights).exists():
        raise LoaderRefusal(
            f"weights_absent: {weights}. FACT-0425 - the publisher's loader is strict=False "
            "with its warnings commented out, so proceeding here would run FOCUS-3D on its "
            "RANDOM INITIALISATION and the null result would be uninterpretable"
        )
    import torch

    blob = torch.load(str(weights), map_location="cpu", weights_only=False)
    state = blob.get("model", blob) if isinstance(blob, dict) else blob
    state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    report = {
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "first_missing": list(missing)[:10],
        "first_unexpected": list(unexpected)[:10],
        "n_params_in_checkpoint": len(state),
    }
    if len(missing) > allow_missing:
        raise LoaderRefusal(f"missing_keys: {len(missing)} > {allow_missing}. {report}")
    if len(unexpected) > allow_unexpected:
        raise LoaderRefusal(f"unexpected_keys: {len(unexpected)} > {allow_unexpected}. {report}")
    return report


# ---------------------------------------------------------------------------------------------
# 2. THE ADAPTER - instances to nodes, in a NAMED unit convention
# ---------------------------------------------------------------------------------------------
def instances_to_nodes(labels: np.ndarray, t: int, *, convention: str,
                       intensity: np.ndarray | None = None,
                       volume_shape_zyx: tuple[int, int, int] | None = None) -> dict:
    """One frame of an instance label volume -> the HOCT node columns, no zero fills.

    ``labels``  : (Z, Y, X) integer instance labels, 0 = background, at FULL RESOLUTION.
    ``convention``: a key of ``hoct_scale_gate.CONVENTIONS``. There is no default: the caller
      states the unit or the adapter refuses, because ``FACT-0454`` blocker 1 is precisely a
      unit that was never stated and never applied.

    Returns the column dict ``hoct_feature_contract.build_node_feats`` consumes. Mask-derived
    columns are present only if they were actually computed; ``intensity`` being None means the
    four intensity columns are ABSENT, which makes the builder refuse - it does not make them
    zero, which is what ``graph.py:184-187`` would do.

    THE INERTIA TENSOR IS COMPUTED IN INDEX SPACE, deliberately. ``graph.py:175-177`` hands
    ``RegionPropsNodes`` the raw label array and never rescales it, so the publisher's shipped
    ``inertia_tensor`` statistics are index-space second moments. Computing ours in microns
    would put a different quantity in the same slot.
    """
    if convention not in CONVENTIONS:
        raise FeatureRefusal(
            f"unknown unit convention {convention!r}; state one of "
            f"{sorted(CONVENTIONS)} - there is no default (FACT-0454 blocker 1)", ["z", "y", "x"]
        )
    factors = np.asarray(CONVENTIONS[convention]["factors_zyx"], dtype=np.float64)

    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise FeatureRefusal(f"labels must be (Z, Y, X), got {labels.shape}", ["z", "y", "x"])
    ids = np.unique(labels)
    ids = ids[ids != 0]
    n = len(ids)
    if n == 0:
        raise FeatureRefusal("no instances in this frame", ["z", "y", "x"])

    cen = np.empty((n, 3), dtype=np.float64)
    eqd = np.empty(n, dtype=np.float64)
    inert = np.empty((n, 3, 3), dtype=np.float64)
    imin = np.empty(n); imax = np.empty(n); imean = np.empty(n); istd = np.empty(n)

    for k, lab in enumerate(ids):
        vox = np.argwhere(labels == lab).astype(np.float64)   # (M, 3) in index space
        c = vox.mean(axis=0)
        cen[k] = c
        m = len(vox)
        # skimage's equivalent_diameter_area in 3D: diameter of the sphere of equal volume.
        eqd[k] = (6.0 * m / np.pi) ** (1.0 / 3.0)
        d = vox - c
        inert[k] = (d.T @ d) / m
        if intensity is not None:
            vals = intensity[labels == lab].astype(np.float64)
            imin[k], imax[k] = vals.min(), vals.max()
            imean[k], istd[k] = vals.mean(), vals.std()

    scaled = cen * factors
    cols: dict[str, np.ndarray] = {
        "t": np.full(n, float(t)),
        "z": scaled[:, 0], "y": scaled[:, 1], "x": scaled[:, 2],
        "equivalent_diameter_area": eqd,
        "inertia_tensor": inert,
    }
    if intensity is not None:
        cols["intensity_min"] = imin
        cols["intensity_max"] = imax
        cols["intensity_mean"] = imean
        cols["intensity_std"] = istd
    if volume_shape_zyx is not None:
        shape = np.asarray(volume_shape_zyx, dtype=np.float64)[None, :]
        dist = np.minimum(cen, shape - cen).min(axis=1)
        cols["border_dist"] = 1.0 - np.minimum(1.0, dist / 5.0)   # features.py:34, cutoff=5
    return {
        "columns": cols,
        "n_instances": int(n),
        "convention": convention,
        "centroids_index_space": cen,
        "centroids_in_convention_units": scaled,
        "notes": {
            "border_dist_units": "INDEX space with cutoff=5, per features.py:8-52. Feeding it "
                                 "micron coordinates against a voxel shape would be wrong",
            "inertia_units": "INDEX space, per graph.py:175-177",
        },
    }


# ---------------------------------------------------------------------------------------------
# 3. THE EVALUATION FIXTURE
# ---------------------------------------------------------------------------------------------
def _load_gt(geff: Path):
    from biotrack.metric import load_graph

    g = load_graph(geff)
    n = g.node_attrs().to_pandas()
    ids = n["node_id"].to_numpy().astype(np.int64)
    row = {int(v): i for i, v in enumerate(ids)}
    arr = n[["t", "z", "y", "x"]].to_numpy().astype(np.float64)
    e = g.edge_attrs().to_pandas()
    edges = [(row[int(s)], row[int(t)]) for s, t in zip(e["source_id"], e["target_id"])
             if int(s) in row and int(t) in row]
    return arr, edges


def _division_tuples(gt: np.ndarray, edges: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Mother rows with out-degree 2, returned as (mother, daughter_a, daughter_b)."""
    kids: dict[int, list[int]] = {}
    for u, v in edges:
        kids.setdefault(u, []).append(v)
    return [(u, k[0], k[1]) for u, k in kids.items() if len(k) == 2]


def evaluate_nodes(gt: np.ndarray, edges: list[tuple[int, int]], pred: np.ndarray) -> dict:
    """The node/mask comparison. Every number goes through the PRODUCTION matcher."""
    mapping = match_gt_to_pred(gt, pred)              # gt row -> pred row, one-to-one, 7 um
    n_gt, n_pred = len(gt), len(pred)
    matched = np.array(sorted(mapping.keys()), dtype=np.int64)

    if len(matched):
        resid = np.linalg.norm(
            (gt[matched, 1:] - pred[[mapping[int(g)] for g in matched], 1:]) * SCALE, axis=1
        )
    else:
        resid = np.zeros(0)

    # duplicate / split / merge, per frame, at the same 7 um the matcher uses
    n_pred_near_gt = np.zeros(n_gt, dtype=np.int64)
    n_gt_near_pred = np.zeros(n_pred, dtype=np.int64)
    dup_pairs = 0
    for frame in np.unique(gt[:, 0]).astype(np.int64):
        gm = np.nonzero(gt[:, 0] == frame)[0]
        pm = np.nonzero(pred[:, 0] == frame)[0]
        if not len(pm) or not len(gm):
            continue
        d = np.linalg.norm(
            (gt[gm, 1:][:, None, :] - pred[pm, 1:][None, :, :]) * SCALE, axis=-1
        )
        near = d <= MAX_DISTANCE_UM
        n_pred_near_gt[gm] = near.sum(axis=1)
        n_gt_near_pred[pm] = near.sum(axis=0)
        if len(pm) > 1:
            pp = np.linalg.norm(
                (pred[pm, 1:][:, None, :] - pred[pm, 1:][None, :, :]) * SCALE, axis=-1
            )
            iu = np.triu_indices(len(pm), k=1)
            dup_pairs += int((pp[iu] <= MAX_DISTANCE_UM).sum())

    # FACT-0380's division-sensitive residual: fully matched division tuples whose WORST member
    # sits beyond 3 um while every member still passes the 7 um matcher.
    tuples = _division_tuples(gt, edges)
    worst = []
    for m, a, b in tuples:
        if m in mapping and a in mapping and b in mapping:
            r = [
                float(np.linalg.norm((gt[i, 1:] - pred[mapping[i], 1:]) * SCALE))
                for i in (m, a, b)
            ]
            worst.append(max(r))
    worst_arr = np.asarray(worst, dtype=np.float64)

    return {
        "n_gt_nodes": int(n_gt),
        "n_pred_nodes": int(n_pred),
        "n_matched": int(len(mapping)),
        "node_recall": float(len(mapping) / n_gt) if n_gt else 0.0,
        "matched_share_of_pred": float(len(mapping) / n_pred) if n_pred else 0.0,
        "localisation": {
            "median_um": float(np.median(resid)) if len(resid) else None,
            "p90_um": float(np.percentile(resid, 90)) if len(resid) else None,
            "share_over_3um": float(np.mean(resid > DIVISION_RESIDUAL_UM)) if len(resid) else None,
        },
        "split_merge": {
            "gt_with_multiple_pred_within_7um": int((n_pred_near_gt > 1).sum()),
            "gt_with_no_pred_within_7um": int((n_pred_near_gt == 0).sum()),
            "pred_with_multiple_gt_within_7um": int((n_gt_near_pred > 1).sum()),
            "duplicate_pred_pairs_within_7um": dup_pairs,
            "split_rate": float((n_pred_near_gt > 1).mean()) if n_gt else None,
            "merge_rate": float((n_gt_near_pred > 1).mean()) if n_pred else None,
        },
        "division_residual": {
            "definition": "FACT-0380: fully matched GT division tuples whose WORST member "
                          "residual exceeds 3 um while all three still pass the 7 um matcher",
            "n_tuples": int(len(tuples)),
            "n_fully_matched": int(len(worst_arr)),
            "median_worst_member_um": float(np.median(worst_arr)) if len(worst_arr) else None,
            "share_over_3um": float(np.mean(worst_arr > DIVISION_RESIDUAL_UM))
            if len(worst_arr) else None,
        },
    }


def _deployed_nodes(fold: int, crop: str) -> np.ndarray:
    src = FOLDS[fold]["slices"] / f"f{fold}_{crop}.parquet"
    if not src.exists():
        raise SystemExit(f"deployed slice missing: {src} (run revladder.py prep --fold {fold})")
    frame = pl.read_parquet(src).filter(pl.col("row_type") == "node")
    return frame.select(["t", "z", "y", "x"]).to_numpy().astype(np.float64)


def _accumulate(per_crop: list[dict]) -> dict:
    """Pool the crop-level counts. Rates are recomputed from totals, never averaged."""
    tot = {
        "n_gt_nodes": 0, "n_pred_nodes": 0, "n_matched": 0,
        "gt_with_multiple_pred_within_7um": 0, "gt_with_no_pred_within_7um": 0,
        "pred_with_multiple_gt_within_7um": 0, "duplicate_pred_pairs_within_7um": 0,
        "n_div_tuples": 0, "n_div_matched": 0, "n_div_over_3um": 0,
    }
    resid_med, div_worst = [], []
    for c in per_crop:
        for k in ("n_gt_nodes", "n_pred_nodes", "n_matched"):
            tot[k] += c[k]
        for k in ("gt_with_multiple_pred_within_7um", "gt_with_no_pred_within_7um",
                  "pred_with_multiple_gt_within_7um", "duplicate_pred_pairs_within_7um"):
            tot[k] += c["split_merge"][k]
        d = c["division_residual"]
        tot["n_div_tuples"] += d["n_tuples"]
        tot["n_div_matched"] += d["n_fully_matched"]
        if d["share_over_3um"] is not None:
            tot["n_div_over_3um"] += int(round(d["share_over_3um"] * d["n_fully_matched"]))
            div_worst.append(d["median_worst_member_um"])
        if c["localisation"]["median_um"] is not None:
            resid_med.append(c["localisation"]["median_um"])
    return {
        **tot,
        "node_recall": tot["n_matched"] / tot["n_gt_nodes"] if tot["n_gt_nodes"] else None,
        "median_of_crop_median_localisation_um": float(np.median(resid_med)) if resid_med else None,
        "division_share_over_3um": (
            tot["n_div_over_3um"] / tot["n_div_matched"] if tot["n_div_matched"] else None
        ),
        "median_of_crop_median_division_worst_um": (
            float(np.median(div_worst)) if div_worst else None
        ),
    }


def run_fixture(fold: int, max_crops: int, gt_dir: Path) -> dict:
    cfg = FOLDS[fold]
    geffs = sorted(gt_dir.glob(f"{cfg['prefix']}_*.geff"))
    if len(geffs) != cfg["n_crops"]:
        raise SystemExit(f"fold {fold}: found {len(geffs)} GT crops, expected {cfg['n_crops']}")
    if max_crops:
        geffs = geffs[:max_crops]

    arms: dict[str, list[dict]] = {"gt_identity": [], "deployed": []}
    for i, g in enumerate(geffs, 1):
        gt, edges = _load_gt(g)
        arms["gt_identity"].append(evaluate_nodes(gt, edges, gt.copy()))
        arms["deployed"].append(evaluate_nodes(gt, edges, _deployed_nodes(fold, g.stem)))
        if i % 5 == 0 or i == len(geffs):
            print(f"  [{i}/{len(geffs)}] {g.stem}", flush=True)

    pooled = {k: _accumulate(v) for k, v in arms.items()}

    ident = pooled["gt_identity"]
    self_check = {
        "arm": "gt_identity",
        "expect": "recall 1.0 and localisation 0.0 - the fixture measuring itself",
        "node_recall": ident["node_recall"],
        "median_localisation_um": ident["median_of_crop_median_localisation_um"],
        "passes": bool(
            ident["node_recall"] is not None and abs(ident["node_recall"] - 1.0) < 1e-12
            and (ident["median_of_crop_median_localisation_um"] or 0.0) < 1e-9
        ),
    }

    dep = pooled["deployed"]
    full_fold = not (max_crops and max_crops < cfg["n_crops"])
    unmatched = dep["n_gt_nodes"] - dep["n_matched"]
    anchors = [
        {
            # The SHARPEST available cross-check, and it is exact rather than approximate:
            # FACT-0447's perfect_membership_additive arm inserted exactly this many GT cells.
            # If our matcher agreed with theirs only approximately, this would not land.
            "anchor": "FACT-0447 unmatched GT cells (exact count)",
            "expected": cfg["unmatched_gt_anchor"],
            "measured": int(unmatched),
            "tolerance": 0,
            "subsampled": not full_fold,
            "passes": bool(full_fold and unmatched == cfg["unmatched_gt_anchor"]),
        },
        {
            "anchor": "FACT-0449 emitted-graph node count (exact)",
            "expected": cfg["n_pred_anchor"],
            "measured": int(dep["n_pred_nodes"]),
            "tolerance": 0,
            "subsampled": not full_fold,
            "passes": bool(full_fold and dep["n_pred_nodes"] == cfg["n_pred_anchor"]),
        },
        {
            "anchor": "FACT-0447 node recall",
            "expected": cfg["node_recall_anchor"],
            "measured": dep["node_recall"],
            "tolerance": 0.01,
            "subsampled": not full_fold,
            "definition_warning": "OURS is matched-node COVERAGE under the official one-to-one "
                                  "7 um matcher. FACT-0447's is tracking_cellmot.node_recall, a "
                                  "different quantity from the scorer. They agree to ~0.2 pp on "
                                  "fold 0 and must not be substituted for one another - the "
                                  "exact-count anchor above is the one that binds",
            "passes": bool(
                dep["node_recall"] is not None
                and abs(dep["node_recall"] - cfg["node_recall_anchor"]) <= 0.01
            ),
        },
        {
            "anchor": "FACT-0380 division worst-member share over 3 um",
            "expected": cfg["div_share_anchor"],
            "measured": dep["division_share_over_3um"],
            "tolerance": 0.10,
            "subsampled": not full_fold,
            "passes": bool(
                dep["division_share_over_3um"] is not None
                and abs(dep["division_share_over_3um"] - cfg["div_share_anchor"]) <= 0.10
            ),
        },
    ]

    focus_refusal = None
    try:
        load_focus3d_strict(None, None)
    except LoaderRefusal as exc:
        focus_refusal = str(exc)

    return {
        "fold": fold,
        "prefix": cfg["prefix"],
        "reduction": {
            "what_is_reduced": "the NUMBER OF CROPS, and nothing else",
            "crops_used": len(geffs),
            "crops_in_fold": cfg["n_crops"],
            "what_is_identical_to_production": {
                "matcher": "ceiling_ladder.match_gt_to_pred - the same function that produced "
                           "FACT-0368, FACT-0371 and FACT-0447",
                "scale_um_zyx": list(SCALE_UM),
                "max_distance_um": MAX_DISTANCE_UM,
                "division_definition": "FACT-0380, verbatim",
            },
        },
        "identity_downsample_guard": assert_not_the_identity_downsample(),
        "instrument_self_check": self_check,
        "registry_anchors": anchors,
        "pooled": pooled,
        "arms_recorded_as_refused": [
            {
                "arm": "focus3d",
                "refusal": focus_refusal,
                "why_recorded_not_skipped": "a silently missing arm is invisible in a panel - "
                                            "the all_passed trap FACT-0417 records",
                "what_would_unblock_it": (
                    f"the FOCUS-3D checkpoint on disk (env {FOCUS3D_WEIGHTS_ENV}); "
                    "C:/temp/focus3d/hfcache holds only the HF README, no weights. "
                    "FACT-0425 puts the blob at 4.469 GB and CPU throughput at 20.2 s/patch"
                ),
            }
        ],
    }


# ---------------------------------------------------------------------------------------------
# 4. ADAPTER SELF-TEST - the mask route turning 14 refused slots into computed ones
# ---------------------------------------------------------------------------------------------
def _synthetic_instance_frame(centroids_zyx: np.ndarray, shape=(64, 256, 256),
                              radius_um=4.0, seed=7) -> tuple[np.ndarray, np.ndarray]:
    """Ellipsoidal instances at given full-res centroids, drawn ANISOTROPICALLY.

    A sphere of ``radius_um`` in PHYSICAL space is an ellipsoid in index space with semi-axes
    ``radius_um / (1.625, 0.40625, 0.40625)``. Drawing a sphere in index space instead would be
    the isotropy mistake this module exists to refuse, so it is not done here either.
    """
    rng = np.random.default_rng(seed)
    lab = np.zeros(shape, dtype=np.int32)
    img = rng.normal(0.15, 0.02, shape).astype(np.float32)
    semi = radius_um / np.asarray(SCALE_UM, dtype=np.float64)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for k, c in enumerate(centroids_zyx, start=1):
        m = (((zz - c[0]) / semi[0]) ** 2 + ((yy - c[1]) / semi[1]) ** 2
             + ((xx - c[2]) / semi[2]) ** 2) <= 1.0
        lab[m] = k
        # TEXTURED, not flat. A flat fill makes intensity_std exactly zero, which the auditor
        # correctly reads as a zero fill - and it would be a fixture differing in KIND.
        img[m] = rng.uniform(0.5, 0.95) + rng.normal(0.0, 0.04, int(m.sum()))
    return lab, img


def adapter_selftest() -> dict:
    """Instances -> 19-vector, audited by the SAME auditor that refuses a zero fill.

    Two arms. With intensity images every one of the 19 slots is computed and the audit must
    PASS. Without them the builder must REFUSE the four intensity slots rather than fill them -
    which is the exact difference between this adapter and ``graph.py:184-187``.
    """
    rng = np.random.default_rng(20260831)

    def centres(n_interior: int, n_border: int) -> np.ndarray:
        interior = np.stack([
            rng.uniform(10, 54, n_interior),
            rng.uniform(20, 236, n_interior),
            rng.uniform(20, 236, n_interior),
        ], axis=1)
        # BORDER-ADJACENT instances. Without them border_dist is constant 0 for every node and
        # the auditor reads it as a zero fill - which it would be right to do.
        border = np.stack([
            rng.uniform(1, 4, n_border),
            rng.uniform(20, 236, n_border),
            rng.uniform(20, 236, n_border),
        ], axis=1)
        return np.concatenate([interior, border])

    # TWO TIME POINTS. A single frame makes slot 0 constant, and the batcher's window always
    # spans several frames, so a one-frame fixture would differ in KIND. All three of these
    # corrections were found BY THE AUDITOR, not by inspection - see `auditor_found` below.
    frames = []
    per_frame = []
    for t in (0, 1):
        cen_t = centres(20, 4)
        lab_t, img_t = _synthetic_instance_frame(cen_t, seed=7 + t)
        out_t = instances_to_nodes(lab_t, t=t, convention="voxel_raw", intensity=img_t,
                                   volume_shape_zyx=lab_t.shape)
        frames.append(build_node_feats(out_t["columns"], out_t["n_instances"]))
        per_frame.append((cen_t, lab_t, img_t, out_t))

    feats = np.concatenate(frames, axis=0)
    n = int(feats.shape[0])
    audit = audit_matrix(feats)

    cen, lab, img, out = per_frame[0]
    # centroid fidelity: does the adapter recover the centres it was given?
    order = np.argsort(out["centroids_index_space"][:, 1])
    ref = cen[np.argsort(cen[:, 1])]
    cen_err_um = float(np.median(np.linalg.norm(
        (out["centroids_index_space"][order] - ref) * np.asarray(SCALE_UM), axis=1
    )))

    # the anisotropy witness: a physically spherical instance must have an index-space inertia
    # tensor whose z variance is (1.625/0.40625)^-2 of its xy variance. If the fixture had drawn
    # spheres in INDEX space this ratio would be 1.0 and the check would fail.
    it = out["columns"]["inertia_tensor"]
    ratio = float(np.median(it[:, 0, 0] / ((it[:, 1, 1] + it[:, 2, 2]) / 2.0)))
    expected = float((SCALE_UM[1] / SCALE_UM[0]) ** 2)

    no_intensity = instances_to_nodes(lab, t=0, convention="voxel_raw", intensity=None,
                                      volume_shape_zyx=lab.shape)
    try:
        build_node_feats(no_intensity["columns"], no_intensity["n_instances"])
        refused, named = False, []
    except FeatureRefusal as exc:
        refused, named = True, exc.slots

    unknown_convention_refused = False
    try:
        instances_to_nodes(lab, t=0, convention="isotropic_please")
    except FeatureRefusal:
        unknown_convention_refused = True

    return {
        "n_instances": n,
        "with_intensity": {
            "feature_matrix_shape": list(feats.shape),
            "audit_verdict": audit["verdict"],
            "offending_slots": audit["offending_slots"],
            "passes": audit["verdict"] == "PASS",
            "meaning": "all 19 slots vary. The 14 slots hoct_feature_contract refuses on a "
                       "points-only pipeline are COMPUTED here - that is what a mask source buys",
        },
        "without_intensity": {
            "refused": refused,
            "named_slots": sorted(named),
            "passes": bool(refused and len(named) == 4),
            "meaning": "four slots refused, not filled. graph.py:184-187 would have filled them",
        },
        "centroid_fidelity": {
            "median_error_um": cen_err_um,
            "passes": bool(cen_err_um < 0.5),
        },
        "anisotropy_witness": {
            "measured_izz_over_mean_ixx_iyy": ratio,
            "expected_for_a_physically_spherical_instance": expected,
            "relative_miss": abs(ratio - expected) / expected,
            "passes": bool(abs(ratio - expected) / expected < 0.10),
            "meaning": "a fixture that drew spheres in INDEX space would return 1.0 here. "
                       "This is the fixture proving it differs from production in COST, not KIND",
        },
        "unknown_convention_refused": unknown_convention_refused,
        "identity_downsample_guard": assert_not_the_identity_downsample(),
        "auditor_found": {
            "why_this_is_recorded": "the first version of this self-test was written as ONE "
                                    "frame of flat-intensity interior instances. The auditor "
                                    "refused it and named three slots. Every one was a way the "
                                    "fixture differed from production in KIND, and none was "
                                    "visible by inspection - which is the whole argument for "
                                    "auditing the array instead of the bookkeeping",
            "slots_it_named": ["t", "intensity_std", "border_dist"],
            "what_each_meant": {
                "t": "a one-frame fixture. The batcher's window always spans several frames",
                "intensity_std": "a flat intensity fill inside each mask gives std exactly 0",
                "border_dist": "every instance was far from the volume edge, so features.py:34 "
                               "returned 0 for all of them",
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("fixture", "adapter-selftest"))
    ap.add_argument("--fold", type=int, default=0, choices=(0, 1))
    ap.add_argument("--max-crops", type=int, default=12)
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    payload: dict = {
        "schema_version": 1,
        "packet": "PKT-0048",
        "instrument": "scripts/win_bet/mask_node_adapter.py",
        "binding_restriction": {
            "fact": "FACT-0451",
            "text": "general_v1's training data CANNOT BE ESTABLISHED and overlap with the "
                    "competition movies cannot be excluded; submission-only judgement",
            "offline_scoreable": False,
        },
    }
    if args.mode == "adapter-selftest":
        r = payload["adapter_selftest"] = adapter_selftest()
        ok = all([
            r["with_intensity"]["passes"], r["without_intensity"]["passes"],
            r["centroid_fidelity"]["passes"], r["anisotropy_witness"]["passes"],
            r["unknown_convention_refused"], r["identity_downsample_guard"]["passes"],
        ])
    else:
        r = payload["fixture"] = run_fixture(args.fold, args.max_crops, args.gt_dir)
        ok = r["instrument_self_check"]["passes"] and r["identity_downsample_guard"]["passes"]

    payload["heartbeat"] = HEARTBEAT_OK if ok else HEARTBEAT_REFUSED
    payload["passes"] = ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    print(f"\n{payload['heartbeat']}")
    if args.mode == "adapter-selftest":
        r = payload["adapter_selftest"]
        print(f"  instances={r['n_instances']}  19-vector audit -> "
              f"{r['with_intensity']['audit_verdict']}")
        print(f"  without intensity -> refused {r['without_intensity']['named_slots']}")
        print(f"  centroid median error {r['centroid_fidelity']['median_error_um']:.4f} um")
        aw = r["anisotropy_witness"]
        print(f"  anisotropy witness {aw['measured_izz_over_mean_ixx_iyy']:.4f} vs expected "
              f"{aw['expected_for_a_physically_spherical_instance']:.4f} "
              f"({aw['relative_miss'] * 100:.1f}% miss)")
    else:
        r = payload["fixture"]
        sc = r["instrument_self_check"]
        print(f"  self-check gt_identity: recall {sc['node_recall']}, "
              f"localisation {sc['median_localisation_um']} -> "
              f"{'PASS' if sc['passes'] else 'FAIL'}")
        for a in r["registry_anchors"]:
            print(f"  anchor {a['anchor']:<48} expected {a['expected']}  measured "
                  f"{a['measured']}  {'PASS' if a['passes'] else 'MISS'}"
                  f"{'  (subsampled)' if a['subsampled'] else ''}")
        d = r["pooled"]["deployed"]
        print(f"  deployed: recall {d['node_recall']:.5f}  split_rate "
              f"{d['gt_with_multiple_pred_within_7um']}/{d['n_gt_nodes']}  merge "
              f"{d['pred_with_multiple_gt_within_7um']}/{d['n_pred_nodes']}  dup_pairs "
              f"{d['duplicate_pred_pairs_within_7um']}")
        print(f"  refused arms: {[a['arm'] for a in r['arms_recorded_as_refused']]}")
    print(f"  -> {args.out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
