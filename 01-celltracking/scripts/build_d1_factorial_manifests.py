"""Build the D1 2x2 checkpoint x family factorial manifests (decision-package correction C1).

WHY THIS EXISTS
---------------
The v5 LOEO export routes ``44b6 -> split_0`` and ``6bba -> split_1``, so *family* and
*checkpoint* are perfectly confounded.  The two checkpoints' 32-D channel bases are
independent (median per-tensor conv-weight cosine -0.00065, relative L2 1.41542 against
sqrt(2) = 1.41421, ``cos(detect_head_w0, detect_head_w1) = -0.155``).  A head fitted in one
basis cannot be applied in the other, so cross-family transfer is **not measurable at all** on
the v5 export.

The fix is a 2x2 checkpoint x family factorial in which the SAME checkpoint encodes both the
source and the target family::

                 encodes 44b6            encodes 6bba
    split 0      held-out  (TARGET)      training  (SOURCE)
    split 1      training  (SOURCE)      held-out  (TARGET)

Fit and select EXCLUSIVELY on the source family inside one checkpoint basis, freeze, then open
the target family ONCE.  Split 1 stages first at full scale: its held-out family is 6bba, which
carries ~85% of the pooled objective.

ONE CANONICAL SOURCE OF TRUTH
-----------------------------
Nothing here globs a directory.  ``census`` reads the committed LOEO roster
(``data/dataset_splits.json``) for the crop set, measures per-crop statistics from the real GT
geffs and the LOEO OOF predictions, and writes ``crop_census.json``.  ``build`` then derives
all three tiers from that census alone -- it never touches the dataset, so regeneration is
byte-reproducible on any machine that has the census file.

USAGE
-----
    python scripts/build_d1_factorial_manifests.py census \
        --splits data/dataset_splits.json \
        --data-dir data/train \
        --oof-dir artifacts/kaggle/oof_clean \
        --ckpt-dir artifacts/kaggle/weights_dataset \
        --out data/d1_factorial/crop_census.json

    python scripts/build_d1_factorial_manifests.py build \
        --census data/d1_factorial/crop_census.json \
        --out-dir data/d1_factorial
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# Frozen facts.  Every one of these is verified by tests/test_d1_factorial.py.
# --------------------------------------------------------------------------------------

SCHEMA_VERSION = 1

FAMILIES: tuple[str, ...] = ("44b6", "6bba")

#: LOEO fold -> checkpoint file name and sha256.  Verified against the real files by
#: ``census`` and re-asserted by the test suite.
CHECKPOINTS: dict[int, dict[str, str]] = {
    0: {
        "weights_file": "edge_predictor_best_split_0.pth",
        "config_file": "config_split_0.json",
        "sha256": "d3e89eb361eeadef06d18159d834594776174a9f8cabd81f65271abbb42a492f",
        "weights_glob": "/kaggle/input/*/edge_predictor_best_split_0.pth",
        "config_glob": "/kaggle/input/*/config_split_0.json",
    },
    1: {
        "weights_file": "edge_predictor_best_split_1.pth",
        "config_file": "config_split_1.json",
        "sha256": "2e4ebf616b3d4fb53881eadf42c7972e04978c28f74f229c5cb30bee835063de",
        "weights_glob": "/kaggle/input/*/edge_predictor_best_split_1.pth",
        "config_glob": "/kaggle/input/*/config_split_1.json",
    },
}

#: LOEO semantics, read off data/dataset_splits.json and re-verified by ``census``:
#: split 0 trains on 6bba and holds out 44b6; split 1 trains on 44b6 and holds out 6bba.
FOLD_TRAIN_FAMILY: dict[int, str] = {0: "6bba", 1: "44b6"}
FOLD_HELDOUT_FAMILY: dict[int, str] = {0: "44b6", 1: "6bba"}

#: Role assignment for the factorial.  The family the checkpoint was TRAINED on supplies the
#: fitting rows (SOURCE); the family it never saw is the measurement (TARGET).
ROLE_SOURCE = "source"
ROLE_TARGET = "target"

#: The three v5 smoke crops.
SMOKE_CROPS: tuple[str, ...] = ("44b6_0113de3b", "6bba_57b7cc1e", "6bba_6feb10f0")

#: Scorer match radius and voxel scale (biotrack.metric.MAX_DISTANCE / DEFAULT_SCALE).
MATCH_UM = 7.0
SCALE_ZYX: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

# --- measured cost anchors -------------------------------------------------------------
# (a) v5 D1 smoke, `predict_minutes_total` from run_stats.csv:
#       fold 0, 1 crop  (44b6_0113de3b)                 -> 2.459 min  = 147.5 s
#       fold 1, 2 crops (6bba_57b7cc1e, 6bba_6feb10f0)  -> 3.917 min  = 235.0 s
#     Two points, two unknowns: fixed 60.0 s + 87.5 s marginal per crop.  Both crop sets are
#     tiny and unrepresentative, so this is used only as an optimistic bracket.
# (b) 199-crop production coupled cache, stock 4-view TTA, LOEO routing:
#       shard0 100 crops in 3.51 h, shard1 99 crops in 3.53 h -> 126.5 / 128.4 s per crop,
#       peak GPU 601/599 MB.
# (c) the decision package plans on "fold 1 consumes 76.6% of a 9 h session" at v6 settings:
#       0.766 * 9 * 3600 / 128 = 193.9 s per crop.  That is 1.53x (b), consistent with the
#       deployed 8-view TTA against production's 4-view.  (c) is the planning number.
SEC_PER_CROP_PLANNING = 193.9  # v6, 8-view TTA (basis: decision-package fold-1 budget)
SEC_PER_CROP_PRODUCTION = 126.5  # measured, 4-view TTA production cache (optimistic bracket)
SEC_PER_CROP_SMOKE_MARGINAL = 87.5  # measured v5 D1 smoke marginal (unrepresentative crops)
SEC_FIXED_PER_KERNEL = 900.0  # model load + dataset mount + wrapper + export + upload
SESSION_SECONDS = 9 * 3600.0
SESSION_USABLE_FRACTION = 0.90  # leave headroom for the non-predict tail

#: The decision package's own GPU budget, against which feasibility is judged.
ROADMAP_GPU_HOURS_FULL_EXPORT = 7.0  # "20-28 h GPU: full-199 v6 export, 2 shards ~7 T4-h"
ROADMAP_GPU_HOURS_TOTAL = 11.5  # "Total GPU ~ 11.5 T4-hours"

# --- measured storage anchors ----------------------------------------------------------
# From the v5 smoke d1_audit pull (agent4/v5_pull), exact file sizes:
#   44b6_0113de3b   gt_rows    52  feat_gt 1,235,584  feat_max 1,235,584  rows.parquet 179,139
#   6bba_57b7cc1e   gt_rows 1,659  feat_gt 1,441,280  feat_max 1,441,280  rows.parquet 376,774
#   6bba_6feb10f0   gt_rows 1,368  feat_gt 1,404,032  feat_max 1,404,032  rows.parquet 342,132
# n_rows == gt_rows + T*(N_UNIFORM + N_SUBTHR) == gt_rows + 100*(64+32) == gt_rows + 9600,
# and each feat npy is float32 (n_rows, 32) => 128 + 128*n_rows bytes.  Fitting the parquet
# residual gives bytes ~= 2,636,600 + 375*gt_rows, which reproduces all three measured crops
# to within 0.25%.
BYTES_FIXED_PER_CROP = 2_636_600
BYTES_PER_GT_ROW = 375
D1_N_UNIFORM = 64
D1_N_SUBTHR = 32

#: PILOT design.  Two families x three GT-mass tertiles x three miss-burden tertiles.
PILOT_MASS_BINS = 3
PILOT_MISS_BINS = 3
PILOT_PER_CELL = 1

#: "Crop size" axis.  MEASURED: all 199 crops are exactly 100 x 64 x 256 x 256 voxels, so the
#: literal voxel crop size has ZERO variance and cannot stratify anything.  The variance-
#: bearing size of a crop is how much work is inside it, i.e. the post-wrapper predicted node
#: load, which is also what drives runtime and association-graph size.  `gt_bbox_um3` (the
#: spatial extent of the annotated region) is reported alongside it in the balance table.
PILOT_SIZE_KEY = "pred_nodes"
PILOT_Z_KEYS = ("gt_nodes", "miss_rate", PILOT_SIZE_KEY)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def family_of(crop_id: str) -> str:
    fam = crop_id.split("_", 1)[0]
    if fam not in FAMILIES:
        raise ValueError(f"crop {crop_id!r} has unknown family {fam!r}")
    return fam


def role_of(fold: int, family: str) -> str:
    if family == FOLD_TRAIN_FAMILY[fold]:
        return ROLE_SOURCE
    if family == FOLD_HELDOUT_FAMILY[fold]:
        return ROLE_TARGET
    raise ValueError(f"family {family!r} is not part of fold {fold}")


def provenance_path(path: Path) -> str:
    """Last two path components -- a stable human label that never embeds a machine.

    The census must be byte-identical whether it is built from the repo or from a worktree
    that points at the repo's data, so absolute paths cannot appear in it.  The bytes that
    matter (the LOEO roster, the two checkpoints) are pinned by sha256 instead.
    """
    parts = Path(path).resolve().parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else Path(path).as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> str:
    """Deterministic, atomic JSON write.  Returns the sha256 of the bytes written."""
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    raw = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:  # binary: never let the platform rewrite newlines
        fh.write(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------------------
# census: the single canonical source of truth
# --------------------------------------------------------------------------------------


def _read_geff_nodes(geff: Path):
    import numpy as np
    import zarr

    root = zarr.open(str(geff), mode="r")
    t = np.asarray(root["nodes/props/t/values"][:])
    z = np.asarray(root["nodes/props/z/values"][:])
    y = np.asarray(root["nodes/props/y/values"][:])
    x = np.asarray(root["nodes/props/x/values"][:])
    n_edges = int(root["edges/ids"].shape[0])
    return t, z, y, x, n_edges


def _miss_burden(gt_geff: Path, pred_geff: Path) -> tuple[int, int]:
    """(#GT nodes with no predicted node within MATCH_UM at the same t, #pred nodes).

    Basis: LOEO.  The predicted graph is the post-wrapper deployed graph, so this is the
    deployed miss burden, not the raw accepted-peak C/T/L/D partition.  It is a stratification
    variable only; it never enters a score.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    gt_t, gt_z, gt_y, gt_x, _ = _read_geff_nodes(gt_geff)
    p_t, p_z, p_y, p_x, _ = _read_geff_nodes(pred_geff)

    sz, sy, sx = SCALE_ZYX
    gt_um = np.stack([gt_z * sz, gt_y * sy, gt_x * sx], axis=1)
    p_um = np.stack([p_z * sz, p_y * sy, p_x * sx], axis=1)

    missed = 0
    for tv in np.unique(gt_t):
        gsel = gt_t == tv
        psel = p_t == tv
        if not psel.any():
            missed += int(gsel.sum())
            continue
        tree = cKDTree(p_um[psel])
        d, _ = tree.query(gt_um[gsel], k=1)
        missed += int((d > MATCH_UM).sum())
    return missed, int(p_t.shape[0])


def cmd_census(args: argparse.Namespace) -> int:
    import numpy as np

    splits_path = Path(args.splits)
    data_dir = Path(args.data_dir)
    oof_dir = Path(args.oof_dir)
    ckpt_dir = Path(args.ckpt_dir)
    out = Path(args.out)

    splits = json.loads(splits_path.read_text())
    if len(splits) != 2:
        raise SystemExit(f"expected 2 LOEO splits, got {len(splits)}")

    # --- verify the LOEO semantics this whole design rests on --------------------------
    roster: list[str] = []
    for fold, spec in enumerate(splits):
        train_fams = {family_of(c) for c in spec["train"]}
        test_fams = {family_of(c) for c in spec["test"]}
        if train_fams != {FOLD_TRAIN_FAMILY[fold]}:
            raise SystemExit(
                f"fold {fold} train families {train_fams} != {FOLD_TRAIN_FAMILY[fold]}"
            )
        if test_fams != {FOLD_HELDOUT_FAMILY[fold]}:
            raise SystemExit(
                f"fold {fold} test families {test_fams} != {FOLD_HELDOUT_FAMILY[fold]}"
            )
        roster.extend(spec["train"])
        roster.extend(spec["test"])
    crops = sorted(set(roster))
    # each crop must appear exactly twice across the two splits (once train, once test)
    if len(roster) != 2 * len(crops):
        raise SystemExit("LOEO roster is not a clean 2x partition of the crop set")

    # --- verify the pinned checkpoint hashes against the real files --------------------
    ckpt_verified: dict[str, dict[str, str]] = {}
    for fold, spec in CHECKPOINTS.items():
        wp = ckpt_dir / spec["weights_file"]
        if not wp.exists():
            raise SystemExit(f"missing checkpoint {wp}")
        got = sha256_file(wp)
        if got != spec["sha256"]:
            raise SystemExit(f"fold {fold} checkpoint sha mismatch: {got} != {spec['sha256']}")
        ckpt_verified[str(fold)] = {
            "weights_file": spec["weights_file"],
            "sha256": got,
            "size_bytes": str(wp.stat().st_size),
        }

    rows = []
    for crop in crops:
        gt_geff = data_dir / f"{crop}.geff"
        vol_meta = data_dir / f"{crop}.zarr" / "0" / "zarr.json"
        if not gt_geff.exists():
            raise SystemExit(f"missing GT geff for {crop}: {gt_geff}")
        if not vol_meta.exists():
            raise SystemExit(f"missing volume metadata for {crop}: {vol_meta}")

        gt_t, gt_z, gt_y, gt_x, gt_edges = _read_geff_nodes(gt_geff)
        shape = [int(v) for v in json.loads(vol_meta.read_text())["shape"]]
        sz, sy, sx = SCALE_ZYX
        gt_bbox_um3 = float(
            max(float(gt_z.max() - gt_z.min()) * sz, 1.0)
            * max(float(gt_y.max() - gt_y.min()) * sy, 1.0)
            * max(float(gt_x.max() - gt_x.min()) * sx, 1.0)
        )

        fam = family_of(crop)
        # LOEO routing of the v5 export: 44b6 -> split_0, 6bba -> split_1
        oof_fold = 0 if fam == "44b6" else 1
        pred_geff = oof_dir / f"pred_geffs_split_{oof_fold}" / f"{crop}.geff"
        if not pred_geff.exists():
            raise SystemExit(f"missing LOEO OOF prediction for {crop}: {pred_geff}")
        missed, pred_nodes = _miss_burden(gt_geff, pred_geff)

        gt_nodes = int(gt_t.shape[0])
        rows.append(
            {
                "crop_id": crop,
                "family": fam,
                "gt_nodes": gt_nodes,
                "gt_edges": int(gt_edges),
                "gt_t_min": int(np.min(gt_t)),
                "gt_t_max": int(np.max(gt_t)),
                "gt_frames_annotated": int(np.unique(gt_t).shape[0]),
                "vol_shape_tzyx": shape,
                "vol_voxels": int(shape[0] * shape[1] * shape[2] * shape[3]),
                "gt_bbox_um3": round(gt_bbox_um3, 3),
                "gt_span_t": int(gt_t.max() - gt_t.min()) + 1,
                "loeo_oof_fold": oof_fold,
                "pred_nodes": pred_nodes,
                "miss_nodes": missed,
                "miss_rate": round(missed / gt_nodes, 6) if gt_nodes else 0.0,
            }
        )
        if args.verbose:
            print(f"  {crop} gt={gt_nodes} miss={missed}", file=sys.stderr)

    vol_shapes = sorted({tuple(r["vol_shape_tzyx"]) for r in rows})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "d1_factorial_crop_census",
        "provenance": {
            "roster_source": provenance_path(splits_path),
            "roster_sha256": sha256_file(splits_path),
            "gt_dir": provenance_path(data_dir),
            "oof_pred_dir": provenance_path(oof_dir),
            "match_um": MATCH_UM,
            "scale_zyx": list(SCALE_ZYX),
            "miss_burden_basis": "LOEO (post-wrapper OOF prediction graphs, 7 um, per-frame NN)",
        },
        "checkpoints": ckpt_verified,
        "fold_train_family": {str(k): v for k, v in FOLD_TRAIN_FAMILY.items()},
        "fold_heldout_family": {str(k): v for k, v in FOLD_HELDOUT_FAMILY.items()},
        "vol_shapes_observed": [list(s) for s in vol_shapes],
        "n_crops": len(rows),
        "crops": sorted(rows, key=lambda r: r["crop_id"]),
    }
    digest = write_json(out, payload)
    fam_counts = {f: sum(1 for r in rows if r["family"] == f) for f in FAMILIES}
    print(f"census -> {out}  n={len(rows)}  {fam_counts}  sha256={digest}")
    return 0


# --------------------------------------------------------------------------------------
# tier construction
# --------------------------------------------------------------------------------------


def _tertile_bins(values: list[float], n_bins: int) -> list[int]:
    """Rank-based equal-count binning.  Deterministic; ties broken by input order."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    bins = [0] * len(values)
    n = len(values)
    for rank, idx in enumerate(order):
        bins[idx] = min(n_bins - 1, (rank * n_bins) // n)
    return bins


def select_pilot(crops: list[dict]) -> dict:
    """Stratified PILOT selection: family x GT-mass tertile x miss-burden tertile.

    Tertiles are formed WITHIN family, so the design is balanced in each family's own
    distribution rather than in the pooled one (the two embryos differ by an order of
    magnitude in GT mass).  Crop size enters as the within-cell selection criterion: the
    representative of a cell is the crop closest to the cell centroid in standardised
    (gt_nodes, miss_rate, vol_voxels) space, so the chosen crops are typical of their cell in
    all three variables and not only in the two that define it.

    The three v5 smoke crops are force-included so that SMOKE is a strict subset of PILOT and
    their already-paid encoder passes stay reusable; each one pins its natural cell.
    """
    chosen: list[str] = []
    strata: list[dict] = []
    forced = set(SMOKE_CROPS)
    bin_of: dict[str, tuple[int, int]] = {}

    for fam in FAMILIES:
        fam_rows = [c for c in crops if c["family"] == fam]
        fam_rows.sort(key=lambda c: c["crop_id"])
        mass_bin = _tertile_bins([float(c["gt_nodes"]) for c in fam_rows], PILOT_MASS_BINS)
        miss_bin = _tertile_bins([float(c["miss_rate"]) for c in fam_rows], PILOT_MISS_BINS)

        def _std(key: str) -> list[float]:
            vals = [float(c[key]) for c in fam_rows]
            mu = sum(vals) / len(vals)
            var = sum((v - mu) ** 2 for v in vals) / len(vals)
            sd = math.sqrt(var) if var > 0 else 1.0
            return [(v - mu) / sd for v in vals]

        zs = {k: _std(k) for k in PILOT_Z_KEYS}

        cells: dict[tuple[int, int], list[int]] = {}
        for i in range(len(fam_rows)):
            cells.setdefault((mass_bin[i], miss_bin[i]), []).append(i)
            bin_of[fam_rows[i]["crop_id"]] = (mass_bin[i], miss_bin[i])

        for mb in range(PILOT_MASS_BINS):
            for sb in range(PILOT_MISS_BINS):
                members = cells.get((mb, sb), [])
                stratum = {
                    "family": fam,
                    "gt_mass_tertile": mb,
                    "miss_burden_tertile": sb,
                    "n_available": len(members),
                    "selected": [],
                }
                if members:
                    cx = {k: sum(zs[k][i] for i in members) / len(members) for k in zs}
                    forced_here = [i for i in members if fam_rows[i]["crop_id"] in forced]
                    if forced_here:
                        # a smoke crop pins its own cell; keep the first by crop_id
                        pick = sorted(forced_here, key=lambda i: fam_rows[i]["crop_id"])[0]
                        stratum["forced_smoke_crop"] = True
                    else:
                        pick = min(
                            members,
                            key=lambda i: (
                                round(sum((zs[k][i] - cx[k]) ** 2 for k in zs) ** 0.5, 9),
                                fam_rows[i]["crop_id"],
                            ),
                        )
                    stratum["selected"] = [fam_rows[pick]["crop_id"]]
                    chosen.append(fam_rows[pick]["crop_id"])
                strata.append(stratum)

    # A smoke crop whose cell representative is another smoke crop still has to be in the
    # pilot -- SMOKE must be a strict subset of PILOT.
    for c in SMOKE_CROPS:
        if c not in chosen:
            chosen.append(c)
            mb, sb = bin_of[c]
            strata.append(
                {
                    "family": family_of(c),
                    "gt_mass_tertile": mb,
                    "miss_burden_tertile": sb,
                    "n_available": 1,
                    "selected": [c],
                    "forced_smoke_crop": True,
                    "co_tenant": True,
                    "note": (
                        "smoke crop appended as a second occupant of an already-represented "
                        "stratum, so that SMOKE stays a strict subset of PILOT"
                    ),
                }
            )

    return {"crops": sorted(set(chosen)), "strata": strata}


def _balance_report(pilot_ids: list[str], crops: list[dict]) -> dict:
    by_id = {c["crop_id"]: c for c in crops}
    pilot = [by_id[c] for c in pilot_ids]

    def stats(rows: list[dict], key: str) -> dict:
        vals = sorted(float(r[key]) for r in rows)
        n = len(vals)
        mean = sum(vals) / n
        med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        return {
            "n": n,
            "min": round(vals[0], 6),
            "median": round(med, 6),
            "mean": round(mean, 6),
            "max": round(vals[-1], 6),
            "total": round(sum(vals), 6),
        }

    out: dict = {}
    for scope, sel in (("all", None), ("44b6", "44b6"), ("6bba", "6bba")):
        p = [r for r in pilot if sel is None or r["family"] == sel]
        f = [r for r in crops if sel is None or r["family"] == sel]
        if not p:
            continue
        out[scope] = {
            key: {"pilot": stats(p, key), "full": stats(f, key)}
            for key in (
                "gt_nodes",
                "gt_edges",
                "miss_nodes",
                "miss_rate",
                "pred_nodes",
                "gt_bbox_um3",
                "vol_voxels",
            )
        }
        out[scope]["crop_share"] = round(len(p) / len(f), 6)
        out[scope]["gt_mass_share_of_full"] = round(
            sum(r["gt_nodes"] for r in p) / sum(r["gt_nodes"] for r in f), 6
        )
    return out


# --------------------------------------------------------------------------------------
# cost model
# --------------------------------------------------------------------------------------


def crop_bytes(row: dict) -> int:
    return BYTES_FIXED_PER_CROP + BYTES_PER_GT_ROW * int(row["gt_nodes"])


def cost_block(rows: list[dict], n_kernels: int) -> dict:
    n = len(rows)
    total_bytes = sum(crop_bytes(r) for r in rows)
    return {
        "crop_inferences": n,
        "sec_per_crop_planning": SEC_PER_CROP_PLANNING,
        "predict_hours": round(n * SEC_PER_CROP_PLANNING / 3600.0, 4),
        "wall_hours_incl_fixed": round(
            (n * SEC_PER_CROP_PLANNING + n_kernels * SEC_FIXED_PER_KERNEL) / 3600.0, 4
        ),
        "wall_hours_production_bracket": round(
            (n * SEC_PER_CROP_PRODUCTION + n_kernels * SEC_FIXED_PER_KERNEL) / 3600.0, 4
        ),
        "wall_hours_smoke_marginal_bracket": round(
            (n * SEC_PER_CROP_SMOKE_MARGINAL + n_kernels * SEC_FIXED_PER_KERNEL) / 3600.0, 4
        ),
        "n_kernels": n_kernels,
        "bytes": total_bytes,
        "mib": round(total_bytes / 1048576.0, 2),
    }


# --------------------------------------------------------------------------------------
# manifest assembly
# --------------------------------------------------------------------------------------


def _cells() -> dict:
    out = {}
    for fold in (0, 1):
        for fam in FAMILIES:
            role = role_of(fold, fam)
            out[f"split_{fold}__{fam}"] = {
                "fold": fold,
                "family": fam,
                "role": role,
                "checkpoint_sha256": CHECKPOINTS[fold]["sha256"],
                "checkpoint_weights_file": CHECKPOINTS[fold]["weights_file"],
                "checkpoint_trained_on_family": FOLD_TRAIN_FAMILY[fold],
                "checkpoint_heldout_family": FOLD_HELDOUT_FAMILY[fold],
                "encoder_saw_this_family_in_training": fam == FOLD_TRAIN_FAMILY[fold],
                "fit_and_select_here": role == ROLE_SOURCE,
                "open_once_after_freeze": role == ROLE_TARGET,
            }
    return out


#: Stage order: split 1 first (held-out 6bba ~ 85% of the objective), and within a fold the
#: SOURCE cell before the TARGET cell so the head is fitted and frozen before the target
#: family is opened.
STAGE_ORDER: tuple[tuple[int, str], ...] = (
    (1, ROLE_SOURCE),
    (1, ROLE_TARGET),
    (0, ROLE_SOURCE),
    (0, ROLE_TARGET),
)


def build_tier(tier: str, crop_ids: list[str], census: dict, max_shard_seconds: float) -> dict:
    by_id = {c["crop_id"]: c for c in census["crops"]}
    crop_ids = sorted(set(crop_ids))
    missing = [c for c in crop_ids if c not in by_id]
    if missing:
        raise SystemExit(f"tier {tier}: crops absent from the census: {missing}")

    rows = []
    for fold in (0, 1):
        for crop in crop_ids:
            fam = family_of(crop)
            rows.append(
                {
                    "row_id": f"split_{fold}::{crop}",
                    "crop_id": crop,
                    "family": fam,
                    "fold": fold,
                    "cell": f"split_{fold}__{fam}",
                    "role": role_of(fold, fam),
                    "checkpoint_sha256": CHECKPOINTS[fold]["sha256"],
                    "gt_nodes": by_id[crop]["gt_nodes"],
                    "miss_nodes": by_id[crop]["miss_nodes"],
                    "est_bytes": crop_bytes(by_id[crop]),
                }
            )
    rows.sort(key=lambda r: r["row_id"])

    # --- shards: fold-pure (the LOEO retarget binds one fold at module level and rebinds
    # --- TEST_DIR globally) and role-pure (a source shard must land and its head be frozen
    # --- before the matching target shard is opened).
    shards = []
    per_crop = SEC_PER_CROP_PLANNING
    for stage, (fold, role) in enumerate(STAGE_ORDER, start=1):
        cell_rows = [r for r in rows if r["fold"] == fold and r["role"] == role]
        if not cell_rows:
            continue
        capacity = max(1.0, max_shard_seconds - SEC_FIXED_PER_KERNEL)
        n_shard = max(1, math.ceil((len(cell_rows) * per_crop) / capacity))
        buckets: list[list[dict]] = [[] for _ in range(n_shard)]
        load = [0] * n_shard
        # balance by GT mass, largest first (production balanced by node count to 0.6% wall)
        for r in sorted(cell_rows, key=lambda r: (-r["gt_nodes"], r["crop_id"])):
            i = min(range(n_shard), key=lambda k: (load[k], k))
            buckets[i].append(r)
            load[i] += r["gt_nodes"]
        fam = FOLD_TRAIN_FAMILY[fold] if role == ROLE_SOURCE else FOLD_HELDOUT_FAMILY[fold]
        for k, bucket in enumerate(buckets):
            stems = sorted(r["crop_id"] for r in bucket)
            shards.append(
                {
                    "shard_id": f"{tier}__s{stage}__split_{fold}__{fam}__{role}__{k}",
                    "stage": stage,
                    "fold": fold,
                    "family": fam,
                    "role": role,
                    "cell": f"split_{fold}__{fam}",
                    "checkpoint_sha256": CHECKPOINTS[fold]["sha256"],
                    "weights_glob": CHECKPOINTS[fold]["weights_glob"],
                    "config_glob": CHECKPOINTS[fold]["config_glob"],
                    "stems": stems,
                    "expected_crops": len(stems),
                    "gt_nodes_total": sum(r["gt_nodes"] for r in bucket),
                    "est_predict_hours": round(len(stems) * per_crop / 3600.0, 4),
                    "est_wall_hours": round(
                        (len(stems) * per_crop + SEC_FIXED_PER_KERNEL) / 3600.0, 4
                    ),
                    "est_bytes": sum(r["est_bytes"] for r in bucket),
                    # VERIFIED against scripts/kaggle_edits/loeo_retarget.py: the retarget
                    # mounts EVERY train .zarr it finds under /kaggle/input and then selects
                    # purely by the declared BIOHUB_LOEO_STEMS list -- there is no
                    # fold-membership guard.  A SOURCE shard therefore runs on the unchanged
                    # kernel; only the stem list and the weights glob differ.
                    "stems_are_in_checkpoint_training_set": role == ROLE_SOURCE,
                    "kernel_edit_change_required": False,
                    "env": {
                        "BIOHUB_LOEO_FOLD": str(fold),
                        "BIOHUB_LOEO_ARM": "strict",
                        "BIOHUB_LOEO_LIMIT": "0",
                        "BIOHUB_LOEO_STEMS": json.dumps(stems),
                        "BIOHUB_LOEO_WEIGHTS_GLOB": CHECKPOINTS[fold]["weights_glob"],
                        "BIOHUB_LOEO_CONFIG_GLOB": CHECKPOINTS[fold]["config_glob"],
                        "BIOHUB_D1_FOLD": str(fold),
                        "BIOHUB_D1_CKPT_SHA": CHECKPOINTS[fold]["sha256"],
                        "BIOHUB_D1_EXPECTED_CROPS": str(len(stems)),
                        "BIOHUB_D1_N_UNIFORM": str(D1_N_UNIFORM),
                        "BIOHUB_D1_N_SUBTHR": str(D1_N_SUBTHR),
                    },
                }
            )
    shards.sort(key=lambda s: (s["stage"], s["shard_id"]))

    census_rows = [by_id[c] for c in crop_ids]
    budget_block: dict = {
        "total": cost_block(rows, len(shards)),
        "by_cell": {},
        "by_stage": {},
        "session_seconds": SESSION_SECONDS,
        "session_usable_fraction": SESSION_USABLE_FRACTION,
        "max_shard_seconds": max_shard_seconds,
    }
    for cell in sorted(_cells()):
        cr = [r for r in rows if r["cell"] == cell]
        if cr:
            nk = len([s for s in shards if s["cell"] == cell])
            blk = cost_block(cr, nk)
            blk["session_fraction_per_kernel"] = round(
                (len(cr) * SEC_PER_CROP_PLANNING + nk * SEC_FIXED_PER_KERNEL)
                / (SESSION_SECONDS * max(1, nk)),
                4,
            )
            budget_block["by_cell"][cell] = blk
    for stage, (fold, role) in enumerate(STAGE_ORDER, start=1):
        sr = [r for r in rows if r["fold"] == fold and r["role"] == role]
        if sr:
            budget_block["by_stage"][str(stage)] = cost_block(
                sr, len([s for s in shards if s["stage"] == stage])
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "d1_factorial_manifest",
        "tier": tier,
        "design": {
            "name": "2x2 checkpoint x family factorial",
            "rationale": (
                "The v5 export confounds family with checkpoint; the two 32-D bases are "
                "independent (rel-L2 1.41542 vs sqrt(2)=1.41421), so a head fitted in one "
                "basis cannot be applied in the other.  Here the SAME checkpoint encodes both "
                "source and target family, so transfer is measured inside one basis."
            ),
            "fit_rule": "fit and select ONLY on the source family within the same checkpoint basis",
            "freeze_rule": "freeze the head, then open the target family exactly once",
            "stage_rule": "split 1 stages first at full scale (held-out 6bba ~ 85% of the objective)",
            "stage_order": [
                {"stage": i + 1, "fold": f, "role": r} for i, (f, r) in enumerate(STAGE_ORDER)
            ],
        },
        "census": {
            "roster_sha256": census["provenance"]["roster_sha256"],
            "n_crops_total": census["n_crops"],
        },
        "cells": _cells(),
        "n_crops": len(crop_ids),
        "n_rows": len(rows),
        "crops": crop_ids,
        "rows": rows,
        "shards": shards,
        "budget": budget_block,
        "storage_model": {
            "bytes_per_crop": "2_636_600 + 375 * gt_nodes",
            "derivation": (
                "feat_gt and feat_max are float32 (n_rows, 32) npy with n_rows = gt_nodes + "
                "T*(N_UNIFORM+N_SUBTHR) = gt_nodes + 9600; the rows.parquet residual fits "
                "179_000 + 119*gt_nodes.  Reproduces all three measured v5 smoke crops to "
                "within 0.25%."
            ),
            "assumes": {"T": 100, "N_UNIFORM": D1_N_UNIFORM, "N_SUBTHR": D1_N_SUBTHR},
        },
        "gt_nodes_total": sum(r["gt_nodes"] for r in census_rows),
    }


def feasibility_block(manifest: dict, census: dict, reduced_source_crops: list[str]) -> dict:
    """Does this tier fit the GPU budget, and if not, what has to give?

    The factorial doubles crop-inferences, so the honest answer for FULL is no.  The escape is
    asymmetric: the TARGET cells are the measurement and must stay at full scale, while the
    SOURCE cells only have to supply enough rows to fit a 33-parameter pointwise head in a
    32-D basis -- a handful of stratified crops already gives O(10^5) rows.  This block prices
    that reduced plan explicitly so the trade is a decision and not an accident.
    """
    by_id = {c["crop_id"]: c for c in census["crops"]}
    # the reduced source set can never exceed the tier it is reducing
    tier_crops = set(manifest["crops"])
    reduced_source_crops = [c for c in reduced_source_crops if c in tier_crops]
    total = manifest["budget"]["total"]
    required = total["wall_hours_incl_fixed"]
    session_h = SESSION_SECONDS / 3600.0
    usable_h = session_h * SESSION_USABLE_FRACTION

    # reduced plan: full TARGET cells, SOURCE cells cut to the stratified subsample
    reduced_rows = []
    for fold in (0, 1):
        tgt_fam = FOLD_HELDOUT_FAMILY[fold]
        src_fam = FOLD_TRAIN_FAMILY[fold]
        reduced_rows += [c for c in manifest["crops"] if family_of(c) == tgt_fam]
        reduced_rows += [c for c in reduced_source_crops if family_of(c) == src_fam]
    n_reduced = len(reduced_rows)
    reduced_bytes = sum(crop_bytes(by_id[c]) for c in reduced_rows)
    reduced_kernels = 4
    reduced_h = (
        n_reduced * SEC_PER_CROP_PLANNING + reduced_kernels * SEC_FIXED_PER_KERNEL
    ) / 3600.0

    # split-1-only plan: stage 1 + stage 2 of the reduced plan
    s1_rows = [c for c in reduced_source_crops if family_of(c) == FOLD_TRAIN_FAMILY[1]]
    s2_rows = [c for c in manifest["crops"] if family_of(c) == FOLD_HELDOUT_FAMILY[1]]
    split1_h = (
        (len(s1_rows) + len(s2_rows)) * SEC_PER_CROP_PLANNING + 2 * SEC_FIXED_PER_KERNEL
    ) / 3600.0

    fits_alloc = required <= ROADMAP_GPU_HOURS_FULL_EXPORT
    fits_total = required <= ROADMAP_GPU_HOURS_TOTAL
    every_shard_fits = all(s["est_wall_hours"] <= usable_h for s in manifest["shards"])

    if fits_alloc:
        verdict = "FITS the allocated full-export budget"
    elif fits_total:
        verdict = "does NOT fit the allocated full-export budget, but fits the total remaining GPU"
    else:
        verdict = "does NOT fit: exceeds the entire remaining GPU budget"

    return {
        "required_t4_hours": round(required, 3),
        "roadmap_allocated_t4_hours_for_full_export": ROADMAP_GPU_HOURS_FULL_EXPORT,
        "roadmap_total_remaining_t4_hours": ROADMAP_GPU_HOURS_TOTAL,
        "fits_allocated_full_export_budget": fits_alloc,
        "fits_total_remaining_budget": fits_total,
        "every_shard_fits_one_session": every_shard_fits,
        "min_kernel_runs": len(manifest["shards"]),
        "kernels_cannot_be_merged_because": (
            "the LOEO retarget binds one fold at module level and rebinds TEST_DIR globally, "
            "so a single kernel cannot serve both checkpoints"
        ),
        "verdict": verdict,
        "what_has_to_give": {
            "principle": (
                "TARGET cells are the measurement and stay at full scale; SOURCE cells only "
                "have to supply fitting rows for a 33-parameter pointwise head in a 32-D "
                "basis, so they can be cut to the stratified subsample"
            ),
            "reduced_plan_crop_inferences": n_reduced,
            "reduced_plan_t4_hours": round(reduced_h, 3),
            "reduced_plan_mib": round(reduced_bytes / 1048576.0, 2),
            "reduced_plan_saving_t4_hours": round(required - reduced_h, 3),
            "split_1_only_first_window_t4_hours": round(split1_h, 3),
            "split_1_only_covers": (
                "the primary experiment: held-out 6bba, ~85% of the pooled objective"
            ),
        },
    }


def cmd_build(args: argparse.Namespace) -> int:
    census_path = Path(args.census)
    census = json.loads(census_path.read_text())
    if census.get("kind") != "d1_factorial_crop_census":
        raise SystemExit("--census does not point at a d1_factorial crop census")
    out_dir = Path(args.out_dir)
    crops = census["crops"]
    all_ids = [c["crop_id"] for c in crops]

    pilot = select_pilot(crops)
    tiers = {"smoke": list(SMOKE_CROPS), "pilot": pilot["crops"], "full": all_ids}

    written = {}
    for tier, ids in tiers.items():
        manifest = build_tier(tier, ids, census, args.max_shard_seconds)
        if tier == "pilot":
            manifest["stratification"] = {
                "variables": [
                    "family",
                    "gt_nodes (GT mass)",
                    f"{PILOT_SIZE_KEY} (crop size / occupied load)",
                    "miss_rate (miss burden)",
                ],
                "design": (
                    f"within each family, {PILOT_MASS_BINS} equal-count GT-mass tertiles x "
                    f"{PILOT_MISS_BINS} equal-count miss-burden tertiles, "
                    f"{PILOT_PER_CELL} representative per non-empty cell, chosen as the crop "
                    "closest to the cell centroid in standardised "
                    f"({', '.join(PILOT_Z_KEYS)}) space"
                ),
                "crop_size_note": (
                    "MEASURED: every one of the 199 crops is exactly 100x64x256x256 voxels, so "
                    "literal voxel crop size has zero variance and cannot stratify.  The size "
                    f"axis is therefore {PILOT_SIZE_KEY}, the post-wrapper predicted node load, "
                    "which is what actually varies and what drives runtime and storage."
                ),
                "smoke_forced": list(SMOKE_CROPS),
                "strata": pilot["strata"],
                "achieved_balance": _balance_report(pilot["crops"], crops),
            }
        manifest["feasibility"] = feasibility_block(manifest, census, pilot["crops"])
        path = out_dir / f"manifest_{tier}.json"
        digest = write_json(path, manifest)
        written[tier] = {
            # file name only: the index must not depend on where it was written
            "file": path.name,
            "sha256": digest,
            "n_crops": manifest["n_crops"],
            "n_rows": manifest["n_rows"],
        }
        b = manifest["budget"]["total"]
        print(
            f"{tier:6s} crops={manifest['n_crops']:3d} rows={manifest['n_rows']:3d} "
            f"shards={len(manifest['shards'])} predict={b['predict_hours']:.2f}h "
            f"wall={b['wall_hours_incl_fixed']:.2f}h storage={b['mib']:.1f}MiB "
            f"| {manifest['feasibility']['verdict']}"
        )
    write_json(
        out_dir / "manifest_index.json",
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "d1_factorial_index",
            # LF-normalised: the repo runs core.autocrlf=true, so the checked-out bytes are
            # not the written bytes and a raw hash would be checkout-dependent.
            "census_sha256": hashlib.sha256(
                census_path.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
            "tiers": written,
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("census", help="scan the dataset once and write the canonical census")
    c.add_argument("--splits", default="data/dataset_splits.json")
    c.add_argument("--data-dir", default="data/train")
    c.add_argument("--oof-dir", default="artifacts/kaggle/oof_clean")
    c.add_argument("--ckpt-dir", default="artifacts/kaggle/weights_dataset")
    c.add_argument("--out", required=True)
    c.add_argument("--verbose", action="store_true")
    c.set_defaults(func=cmd_census)

    b = sub.add_parser("build", help="derive all three tiers from the census (no dataset access)")
    b.add_argument("--census", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument(
        "--max-shard-seconds",
        type=float,
        default=SESSION_SECONDS * SESSION_USABLE_FRACTION,
        help="wall-clock budget for one kernel shard (default 90%% of a 9 h session)",
    )
    b.set_defaults(func=cmd_build)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
