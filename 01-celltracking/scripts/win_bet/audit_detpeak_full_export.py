r"""Fail-closed audit of a FULL DetPeak export against its paired P28 champion control.

WHY A SECOND INSTRUMENT
-----------------------
``validate_detpeak_export.py`` is the SMOKE gate: sidecar fields, threshold reconstruction, and
whole-graph equality on the crops it was given. It is binary and it stops there. A full two-fold
export needs three things the smoke never had to answer:

1. **Is the detection surface itself identical to the control?** This is the only question
   ``LEVER-0028`` depends on. Node budgeting re-thresholds the candidate peak set; if the export
   run's detector differed at all from the P28 champion run, the sidecars describe a substrate
   that is not the one the 0.928 lineage was measured on. The predictor writes its per-crop node
   count to ``run_stats.csv`` as ``raw_nodes``, so this is directly checkable and it is the gate.

2. **If the final graph diverges, WHERE?** ``run_stats.csv`` carries a counter per pipeline stage.
   A divergence confined to stages strictly downstream of detection is a property of the pipeline,
   not of the exporter, and does not invalidate the sidecars. A divergence at or before detection
   invalidates them. The audit reports the earliest differing stage rather than a pass/fail bit.

3. **Can a champion node be located in the candidate export?** ``PKT-0024`` asks for this
   fail-closed. It cannot be an identity test: the deployed pipeline inserts synthetic gap nodes
   and then line-fit smooths EVERY surviving node, so champion coordinates are not peak
   coordinates. What is checkable is the RESIDUAL to the nearest surviving peak, whose scale is
   set by the downsampled grid (peaks sit on a 4-voxel y/x lattice = 1.625 um isotropic), plus an
   accounting of how many nodes the pipeline's own counters say were synthesised.

STAGE ORDER
-----------
The counters are grouped in pipeline order so "earliest differing stage" is meaningful. Detection
is ``raw_nodes``; everything after it is post-processing inside the notebook.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

SCALE_UM = (1.625, 0.40625, 0.40625)   # src/biotrack/metric.py:28
DOWNSAMPLE = (1, 4, 4)                 # predict_unet_transformer.py:157
PEAK_GRID_UM = 1.625                   # the downsampled peak lattice is isotropic 1.625 um

# run_stats.csv counters in pipeline order. `raw_nodes` is the DETECTION stage and the only
# one LEVER-0028 re-thresholds; every later group is notebook post-processing.
STAGE_ORDER: list[tuple[str, tuple[str, ...]]] = [
    ("detection", ("raw_nodes",)),
    ("edge_prediction", ("raw_edges",)),
    ("motion_relink", (
        "motion_relink_edges", "motion_relink_tight_edges", "motion_relink_relaxed_edges",
        "motion_relink_replaced_raw_edges", "motion_relink_fallback_raw",
    )),
    ("gap_close", (
        "gap_candidates", "gap_pairs_selected", "gap_inserted_synthetic", "gap_added_nodes",
        "gap_added_edges", "gap_refined_synthetic", "gap2_added_nodes", "gap2_added_edges",
    )),
    ("deepcenter", (
        "deepcenter_gap_checked", "deepcenter_gap_accepted", "deepcenter_gap_rejected",
        "deepcenter_safe_div_checked", "deepcenter_safe_div_accepted",
        "deepcenter_safe_div_rejected",
    )),
    ("safe_division", ("safe_division_candidates", "safe_divisions_added")),
    ("short_track_filter", (
        "short_track_components_removed", "short_track_nodes_removed",
        "short_track_edges_removed", "short_track_rescue_components",
    )),
    ("linefit", ("linefit_smoothed_nodes",)),
    ("final", ("nodes", "edges")),
]

SYNTHETIC_NODE_COUNTERS = ("gap_added_nodes", "gap2_added_nodes", "safe_divisions_added")


def audit_sidecars(peaks_dir: Path) -> dict:
    """Field, non-emptiness and threshold-reconstruction contract for every sidecar."""
    files = sorted(peaks_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"no sidecars under {peaks_dir}")
    required = {"t", "zyx", "logit", "pipeline_threshold", "pipeline_peak_count"}
    per_crop: dict[str, dict] = {}
    thresholds: set[float] = set()
    for path in files:
        with np.load(path, allow_pickle=False) as sidecar:
            missing = required - set(sidecar.files)
            if missing:
                raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
            logits = sidecar["logit"].astype(np.float64)
            zyx = sidecar["zyx"]
            times = sidecar["t"]
            threshold = float(sidecar["pipeline_threshold"])
            recorded = int(sidecar["pipeline_peak_count"])
        if logits.size == 0:
            raise ValueError(f"{path.name}: empty sidecar")
        if not (len(logits) == len(zyx) == len(times)):
            raise ValueError(f"{path.name}: ragged arrays")
        if not np.isfinite(logits).all():
            raise ValueError(f"{path.name}: non-finite logits")
        probs = 1.0 / (1.0 + np.exp(-logits))
        reconstructed = int(np.count_nonzero(probs > threshold))
        if reconstructed != recorded:
            raise ValueError(
                f"{path.name}: pipeline peak count mismatch {reconstructed} != {recorded}"
            )
        thresholds.add(threshold)
        per_crop[path.stem] = {
            "exported_peaks": int(len(logits)),
            "pipeline_peak_count": recorded,
            "pipeline_threshold": threshold,
            "min_prob": float(probs.min()),
            "n_frames": int(len(np.unique(times))),
        }
    if len(thresholds) != 1:
        raise ValueError(f"sidecars disagree on the pipeline threshold: {sorted(thresholds)}")
    return {
        "n_sidecars": len(per_crop),
        "pipeline_threshold": thresholds.pop(),
        "exported_peaks_total": sum(v["exported_peaks"] for v in per_crop.values()),
        "pipeline_peaks_total": sum(v["pipeline_peak_count"] for v in per_crop.values()),
        "min_exported_prob": min(v["min_prob"] for v in per_crop.values()),
        "per_crop": per_crop,
    }


def audit_stages(control_stats: Path, export_stats: Path) -> dict:
    """Earliest differing pipeline stage, per crop, from the run_stats counters."""
    a = pl.read_csv(control_stats).sort("dataset")
    b = pl.read_csv(export_stats).sort("dataset")
    if a["dataset"].to_list() != b["dataset"].to_list():
        raise ValueError("run_stats crop sets differ between control and export")
    present = set(a.columns) & set(b.columns)
    stage_diffs: dict[str, list[str]] = {}
    per_crop_earliest: dict[str, str] = {}
    for stage, columns in STAGE_ORDER:
        cols = [c for c in columns if c in present]
        if not cols:
            continue
        differing: list[str] = []
        for c in cols:
            mask = (b[c] - a[c]) != 0
            differing.extend(a.filter(mask)["dataset"].to_list())
        differing = sorted(set(differing))
        if differing:
            stage_diffs[stage] = differing
            for name in differing:
                per_crop_earliest.setdefault(name, stage)
    detection_identical = "detection" not in stage_diffs
    return {
        "n_crops": a.height,
        "detection_identical": detection_identical,
        "stages_differing": stage_diffs,
        "earliest_differing_stage_per_crop": per_crop_earliest,
        "synthetic_nodes_total": {
            c: int(a[c].sum()) for c in SYNTHETIC_NODE_COUNTERS if c in present
        },
    }


def audit_graph_parity(control_csv: Path, export_csv: Path) -> dict:
    """Per-crop exact row equality of the emitted graph, ignoring the running `id` column."""
    a = pl.read_csv(control_csv).drop("id")
    b = pl.read_csv(export_csv).drop("id")
    cols = a.columns
    crops = sorted(set(a["dataset"].unique()) | set(b["dataset"].unique()))
    divergent: dict[str, dict] = {}
    for name in crops:
        x = a.filter(pl.col("dataset") == name).sort(cols)
        y = b.filter(pl.col("dataset") == name).select(cols).sort(cols)
        if x.equals(y):
            continue
        divergent[name] = {
            "control_rows": x.height,
            "export_rows": y.height,
            "node_delta": int(
                y.filter(pl.col("row_type") == "node").height
                - x.filter(pl.col("row_type") == "node").height
            ),
            "edge_delta": int(
                y.filter(pl.col("row_type") == "edge").height
                - x.filter(pl.col("row_type") == "edge").height
            ),
        }
    return {
        "n_crops": len(crops),
        "n_identical": len(crops) - len(divergent),
        "exact_parity": not divergent,
        "divergent_crops": divergent,
        "control_rows_total": a.height,
        "export_rows_total": b.height,
    }


def audit_locatability(control_csv: Path, peaks_dir: Path, bound_um: float) -> dict:
    """Residual from every champion node to the nearest SURVIVING peak, in microns.

    This is deliberately NOT an identity test. The pipeline synthesises gap nodes and then
    line-fit smooths every surviving node, so a champion coordinate is not a peak coordinate.
    The scale that matters is the peak lattice: peaks sit 1.625 um apart isotropically, so a
    smoothed node's expected residual is a fraction of that. A node far outside the lattice
    spacing is one the candidate export cannot explain.
    """
    from scipy.spatial import cKDTree

    scale = np.asarray(SCALE_UM, dtype=np.float64)
    down = np.asarray(DOWNSAMPLE, dtype=np.float64)
    nodes = pl.read_csv(control_csv).filter(pl.col("row_type") == "node")
    residuals: list[np.ndarray] = []
    unlocatable_crops: dict[str, int] = {}
    n_nodes = 0
    for key, group in nodes.group_by("dataset"):
        name = key[0] if isinstance(key, tuple) else key
        sidecar_path = peaks_dir / f"{name}.npz"
        if not sidecar_path.exists():
            raise ValueError(f"champion crop {name} has no sidecar")
        with np.load(sidecar_path, allow_pickle=False) as sidecar:
            times = sidecar["t"].astype(np.int64)
            zyx = sidecar["zyx"].astype(np.float64)
            logits = sidecar["logit"].astype(np.float64)
            threshold = float(sidecar["pipeline_threshold"])
        keep = (1.0 / (1.0 + np.exp(-logits))) > threshold
        peak_um = (zyx * down) * scale
        node_arr = group.select(["t", "z", "y", "x"]).to_numpy().astype(np.float64)
        n_nodes += len(node_arr)
        over = 0
        for frame in np.unique(node_arr[:, 0]).astype(np.int64):
            pm = keep & (times == frame)
            nm = node_arr[:, 0] == frame
            if not pm.any():
                over += int(nm.sum())
                continue
            tree = cKDTree(peak_um[pm])
            distances, _ = tree.query(node_arr[nm, 1:] * scale)
            residuals.append(distances)
            over += int((distances > bound_um).sum())
        if over:
            unlocatable_crops[name] = over
    alld = np.concatenate(residuals) if residuals else np.zeros(0)
    beyond = int((alld > bound_um).sum())
    return {
        "bound_um": bound_um,
        "peak_lattice_um": PEAK_GRID_UM,
        "n_champion_nodes": int(n_nodes),
        "n_residuals": int(len(alld)),
        "percentiles_um": {
            f"p{q}": float(np.percentile(alld, q)) for q in (50, 75, 90, 95, 99, 99.9, 100)
        },
        "frac_within_bound": float(1.0 - beyond / max(len(alld), 1)),
        "n_beyond_bound": beyond,
        "crops_with_nodes_beyond_bound": len(unlocatable_crops),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--peaks-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--control-csv", type=Path, required=True)
    ap.add_argument("--export-csv", type=Path, required=True)
    ap.add_argument("--control-stats", type=Path, required=True)
    ap.add_argument("--export-stats", type=Path, required=True)
    ap.add_argument("--locatability-bound-um", type=float, default=PEAK_GRID_UM * 2.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("graph_unchanged") is not True:
        raise ValueError("manifest does not declare graph_unchanged=true")

    sidecars = audit_sidecars(args.peaks_dir)
    stages = audit_stages(args.control_stats, args.export_stats)
    parity = audit_graph_parity(args.control_csv, args.export_csv)
    locate = audit_locatability(args.control_csv, args.peaks_dir, args.locatability_bound_um)

    if sidecars["n_sidecars"] != stages["n_crops"]:
        raise ValueError(
            f"sidecar count {sidecars['n_sidecars']} != crop count {stages['n_crops']}"
        )

    # THE GATE. LEVER-0028 re-thresholds the detection surface and nothing else, so the
    # sidecars are usable if and only if detection was identical to the control run. Graph
    # parity is reported, but a divergence confined to post-detection stages does not
    # invalidate the candidate set.
    detection_contract = bool(stages["detection_identical"])
    result = {
        "schema_version": 1,
        "export_threshold": manifest.get("export_threshold"),
        "sidecars": sidecars,
        "stages": stages,
        "graph_parity": parity,
        "locatability": locate,
        "detection_contract_pass": detection_contract,
        "graph_exact_parity": parity["exact_parity"],
        "pass": detection_contract,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"DETPEAK_FULL_AUDIT pass={result['pass']} "
        f"detection_identical={detection_contract} "
        f"graph_exact_parity={parity['exact_parity']} "
        f"sidecars={sidecars['n_sidecars']} "
        f"divergent_crops={len(parity['divergent_crops'])} "
        f"stages_differing={sorted(stages['stages_differing'])}"
    )
    if not detection_contract:
        raise SystemExit("DETECTION CONTRACT FAILED - sidecars do not describe the P28 substrate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
