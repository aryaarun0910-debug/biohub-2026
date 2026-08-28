r"""Lost-cell atlas: why the pipeline discards annotated cells its own detector already found.

THE QUESTION THIS DECIDES
-------------------------
``FACT-0355`` measured that the P28 pipeline nets away 133 (fold 0) and 5,000 (fold 1) annotated
cells the detector had matched. ``LEVER-0035`` proposes recovering them by relaxing the solver.
``PKT-0025`` falsifier (b) says that is only worth doing if the solver is CAUSING the loss rather
than reacting correctly to weak association - if a lost cell had no viable candidate edge in the
first place, no solver weight can rescue it and the effort belongs in association instead.

So this instrument classifies every lost cell rather than counting them, on the axes that separate
those two worlds:

  detector_prob        the sigmoid of the peak that matched it - was it a confident detection?
  residual_um          how far that peak sat from the annotation - a localisation quality axis
  cand_out / cand_in   how many CANDIDATE edges the pre-ILP graph offered it, and their best
                       probability. THIS IS THE DECIDING COLUMN: zero candidates means the solver
                       had nothing to work with.
  t, t_frac            distance to the temporal boundary - track ends are where appearance and
                       disappearance costs actually bite
  frames_to_division   proximity to a GT division, because divisions are where the topology the
                       metric scores is most fragile (FACT-0349)

MATCHING
--------
Both the "did the detector find it" and "did the pipeline keep it" questions use the OFFICIAL
one-to-one bipartite matching at 7.0 um, through the same shared implementation as the recall
numbers in ``FACT-0354``/``FACT-0355`` (``detpeak_curve.match_one_to_one_pairs``), so a cell
counted as lost here is the same cell counted as lost there.

FOLD SCOPE, STATED HONESTLY
---------------------------
The candidate-edge columns need a pre-ILP graph, which exists only for fold 1. Fold 0 - the harder
promotion gate - can still be atlased on every other axis, and the instrument says which columns
are populated rather than silently emitting zeros.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detpeak_curve import (  # noqa: E402
    DOWNSAMPLE,
    MAX_DISTANCE_UM,
    SCALE_UM,
    match_one_to_one_pairs,
    sigmoid,
)

SCALE = np.asarray(SCALE_UM, dtype=np.float64)
DOWN = np.asarray(DOWNSAMPLE, dtype=np.float64)


def load_gt_graph(geff_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """GT node table (node_id, t, z, y, x), edge table (source, target), division metadata."""
    from biotrack.metric import load_graph

    graph = load_graph(geff_path)
    nodes = graph.node_attrs().to_pandas()
    edges = graph.edge_attrs().to_pandas()
    node_arr = nodes[["node_id", "t", "z", "y", "x"]].to_numpy().astype(np.float64)
    edge_arr = (
        edges[["source_id", "target_id"]].to_numpy().astype(np.int64)
        if len(edges) else np.zeros((0, 2), dtype=np.int64)
    )
    out_degree = Counter(edge_arr[:, 0].tolist())
    division_sources = {int(k) for k, v in out_degree.items() if v >= 2}
    t_by_id = {int(r[0]): int(r[1]) for r in node_arr}
    division_frames = sorted({t_by_id[d] for d in division_sources if d in t_by_id})
    return node_arr, edge_arr, {
        "division_sources": division_sources,
        "division_frames": np.asarray(division_frames, dtype=np.int64),
        "n_divisions": len(division_sources),
    }


def candidate_support(preilp_crop: pl.DataFrame | None) -> dict:
    """(t,z,y,x) -> candidate out/in degree and best probability, from the pre-ILP graph."""
    if preilp_crop is None or preilp_crop.height == 0:
        return {}
    nodes = preilp_crop.filter(pl.col("row_type") == "node")
    edges = preilp_crop.filter(pl.col("row_type") == "edge")
    key_by_id = {
        int(nid): (int(t), float(z), float(y), float(x))
        for nid, t, z, y, x in zip(
            nodes["node_id"], nodes["t"], nodes["z"], nodes["y"], nodes["x"]
        )
    }
    out_n: dict[int, int] = defaultdict(int)
    in_n: dict[int, int] = defaultdict(int)
    out_best: dict[int, float] = defaultdict(float)
    in_best: dict[int, float] = defaultdict(float)
    for s, t, p in zip(edges["source_id"], edges["target_id"], edges["edge_prob"]):
        s, t, p = int(s), int(t), float(p)
        out_n[s] += 1
        in_n[t] += 1
        out_best[s] = max(out_best[s], p)
        in_best[t] = max(in_best[t], p)
    return {
        key: {
            "cand_out": out_n.get(nid, 0),
            "cand_in": in_n.get(nid, 0),
            "cand_out_best": out_best.get(nid, 0.0),
            "cand_in_best": in_best.get(nid, 0.0),
        }
        for nid, key in key_by_id.items()
    }


def atlas_crop(
    crop: str,
    sidecar: Path,
    gt_geff: Path,
    final_nodes: np.ndarray,
    preilp_crop: pl.DataFrame | None,
) -> list[dict]:
    with np.load(sidecar, allow_pickle=False) as z:
        times = z["t"].astype(np.int64)
        zyx = z["zyx"].astype(np.float64)
        logits = z["logit"].astype(np.float64)
        threshold = float(z["pipeline_threshold"])
    keep = sigmoid(logits) > threshold
    peak_level0 = zyx * DOWN
    probs = sigmoid(logits)

    gt_nodes, _, div = load_gt_graph(gt_geff)
    support = candidate_support(preilp_crop)
    t_max = int(gt_nodes[:, 1].max()) if len(gt_nodes) else 0
    div_frames = div["division_frames"]

    rows: list[dict] = []
    for frame in np.unique(gt_nodes[:, 1]).astype(np.int64):
        gmask = gt_nodes[:, 1] == frame
        gt_um = gt_nodes[gmask, 2:] * SCALE
        gt_ids = gt_nodes[gmask, 0].astype(np.int64)

        pmask = keep & (times == frame)
        det_pairs = match_one_to_one_pairs(peak_level0[pmask] * SCALE, gt_um, MAX_DISTANCE_UM)
        if not det_pairs:
            continue
        pidx = np.nonzero(pmask)[0]

        fmask = final_nodes[:, 0] == frame
        fin_pairs = match_one_to_one_pairs(final_nodes[fmask, 1:] * SCALE, gt_um, MAX_DISTANCE_UM)
        retained_gt = {g for g, _, _ in fin_pairs}

        if len(div_frames):
            frames_to_div = int(np.min(np.abs(div_frames - frame)))
        else:
            frames_to_div = -1

        for g, p, dist in det_pairs:
            lost = g not in retained_gt
            global_peak = int(pidx[p])
            key = (
                int(frame),
                float(peak_level0[global_peak, 0]),
                float(peak_level0[global_peak, 1]),
                float(peak_level0[global_peak, 2]),
            )
            sup = support.get(key)
            rows.append({
                "crop": crop,
                "gt_node_id": int(gt_ids[g]),
                "t": int(frame),
                "t_frac": (frame / t_max) if t_max else 0.0,
                "at_temporal_boundary": bool(frame == 0 or frame == t_max),
                "detector_prob": float(probs[global_peak]),
                "residual_um": float(dist),
                "lost": bool(lost),
                "frames_to_division": frames_to_div,
                "cand_out": sup["cand_out"] if sup else None,
                "cand_in": sup["cand_in"] if sup else None,
                "cand_out_best": sup["cand_out_best"] if sup else None,
                "cand_in_best": sup["cand_in_best"] if sup else None,
                "cand_support_known": sup is not None,
            })
    return rows


def summarise(table: pl.DataFrame) -> dict:
    lost = table.filter(pl.col("lost"))
    kept = table.filter(~pl.col("lost"))
    known = lost.filter(pl.col("cand_support_known"))
    out = {
        "n_detected_annotated_cells": table.height,
        "n_lost": lost.height,
        "n_kept": kept.height,
        "lost_fraction": lost.height / max(table.height, 1),
        "candidate_support_known_for": known.height,
    }
    if lost.height:
        out["lost_profile"] = {
            "detector_prob_median": float(lost["detector_prob"].median()),
            "residual_um_median": float(lost["residual_um"].median()),
            "at_temporal_boundary_frac": float(lost["at_temporal_boundary"].mean()),
        }
        out["kept_profile"] = {
            "detector_prob_median": float(kept["detector_prob"].median()) if kept.height else None,
            "residual_um_median": float(kept["residual_um"].median()) if kept.height else None,
            "at_temporal_boundary_frac": float(kept["at_temporal_boundary"].mean()) if kept.height else None,
        }
    if known.height:
        # THE DECIDING SPLIT. A lost cell with zero candidate edges is one no solver weight can
        # rescue; a lost cell with viable candidates is one the solver declined.
        no_cand = known.filter((pl.col("cand_out") == 0) & (pl.col("cand_in") == 0))
        any_cand = known.filter((pl.col("cand_out") > 0) | (pl.col("cand_in") > 0))
        kept_known = kept.filter(pl.col("cand_support_known"))
        out["deciding_split"] = {
            "lost_with_no_candidate_edge": no_cand.height,
            "lost_with_candidate_edge": any_cand.height,
            "share_unrescuable_by_solver": no_cand.height / known.height,
            "lost_with_candidate_best_prob_median": (
                float(
                    any_cand.select(
                        pl.max_horizontal("cand_out_best", "cand_in_best")
                    ).to_series().median()
                ) if any_cand.height else None
            ),
            "kept_best_prob_median": (
                float(
                    kept_known.select(
                        pl.max_horizontal("cand_out_best", "cand_in_best")
                    ).to_series().median()
                ) if kept_known.height else None
            ),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--peaks-dir", type=Path, required=True)
    ap.add_argument("--final-csv", type=Path, required=True)
    ap.add_argument("--preilp", type=Path, help="pre-ILP parquet; enables the candidate-edge columns")
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--max-crops", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-table", type=Path)
    args = ap.parse_args()

    final = (
        pl.read_csv(args.final_csv)
        .filter(pl.col("row_type") == "node")
        .select(["dataset", "t", "z", "y", "x"])
    )
    final_by_crop = {
        (k[0] if isinstance(k, tuple) else k): g.select(["t", "z", "y", "x"]).to_numpy().astype(np.float64)
        for k, g in final.group_by("dataset")
    }

    preilp = pl.read_parquet(args.preilp) if args.preilp else None

    sidecars = sorted(args.peaks_dir.glob("*.npz"))
    if args.max_crops:
        sidecars = sidecars[: args.max_crops]

    rows: list[dict] = []
    for i, sidecar in enumerate(sidecars, 1):
        crop = sidecar.stem
        gt_geff = args.gt_dir / f"{crop}.geff"
        if not gt_geff.exists():
            raise SystemExit(f"missing GT for {crop}")
        pre_crop = preilp.filter(pl.col("dataset") == crop) if preilp is not None else None
        rows.extend(atlas_crop(crop, sidecar, gt_geff, final_by_crop[crop], pre_crop))
        print(f"  [{i}/{len(sidecars)}] {crop}", flush=True)

    table = pl.DataFrame(rows)
    if args.out_table:
        args.out_table.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(args.out_table)

    result = {
        "schema_version": 1,
        "peaks_dir": str(args.peaks_dir),
        "final_csv": str(args.final_csv),
        "preilp": str(args.preilp) if args.preilp else None,
        "n_crops": len(sidecars),
        "summary": summarise(table),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    s = result["summary"]
    line = (
        f"LOST_CELL_ATLAS crops={len(sidecars)} detected={s['n_detected_annotated_cells']:,} "
        f"lost={s['n_lost']:,} ({s['lost_fraction']:.4f})"
    )
    if "deciding_split" in s:
        d = s["deciding_split"]
        line += (
            f" | no_candidate_edge={d['lost_with_no_candidate_edge']:,} "
            f"has_candidate={d['lost_with_candidate_edge']:,} "
            f"unrescuable_share={d['share_unrescuable_by_solver']:.3f}"
        )
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
