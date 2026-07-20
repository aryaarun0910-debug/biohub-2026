"""Corrected 20-crop temporal-evidence gate for isolated missed cells.

Primary, preregistered endpoint (five frames):

    [(spatial-flow - static) at positive] - [(spatial-flow - static) at control]

Controls are DAXI peaks from the same crop/frame, matched on depth, predicted-cell
density, and finally *the identical patch-inference response*.  Cached full-frame
responses are never used for final matching or scoring.  GT motion is an oracle
signal-existence ceiling.  Deployable motion is a spatial kNN median displacement
field built only from the frozen E0c graph, with the robust per-frame median used as
fallback.  The per-frame-only field and 3/7-frame windows are diagnostics, not extra
ways to pass.

Run the coverage/power audit first:

    python scripts/win_bet/daxi_accumulation_v3.py --daxi-dir PATH --audit-only

Then run inference only if both families have adequate crop/event coverage.  Results
are written atomically as JSON summary plus event/coverage CSV files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

E0C_CACHE = ROOT / "artifacts/kaggle/e0c_cache"
SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
PRIMARY_WINDOW = 5
WINDOWS = (3, 5, 7)
WZ, WYX = 1, 3
CONTROL_GT_EXCLUSION_UM = 7.0


@dataclass
class Event:
    gid: int
    t0: int
    oracle: dict[int, np.ndarray]
    density: float


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def crop_split(crop: str) -> int:
    return 0 if crop.startswith("44b6_") else 1


def load_pred_nodes_edges(split: int, crop: str):
    df = pl.read_parquet(E0C_CACHE / "graphs" / str(split) / f"{crop}.parquet")
    nodes = df.filter(pl.col("row_type") == "node")
    edges = df.filter(pl.col("row_type") == "edge")
    pos = {
        int(row["node_id"]): np.array([row["z"], row["y"], row["x"]], dtype=np.float64)
        for row in nodes.iter_rows(named=True)
    }
    times = {int(row["node_id"]): int(row["t"]) for row in nodes.iter_rows(named=True)}
    pairs = [(int(row["source_id"]), int(row["target_id"])) for row in edges.iter_rows(named=True)]
    return pos, times, pairs


class PredDensity:
    """Label-free predicted-cell density used to match positives and controls."""

    def __init__(self, pos: dict[int, np.ndarray], times: dict[int, int]):
        by_t: dict[int, list[np.ndarray]] = defaultdict(list)
        for node_id, xyz in pos.items():
            by_t[times[node_id]].append(xyz * SCALE)
        self.trees = {t: cKDTree(np.stack(points)) for t, points in by_t.items() if points}

    def count(self, t: int, xyz: np.ndarray, radius_um: float = 12.0) -> float:
        tree = self.trees.get(int(t))
        if tree is None:
            return 0.0
        return float(len(tree.query_ball_point(np.asarray(xyz) * SCALE, radius_um)))

    def counts(self, t: int, xyz: np.ndarray, radius_um: float = 12.0) -> np.ndarray:
        tree = self.trees.get(int(t))
        if tree is None or len(xyz) == 0:
            return np.zeros(len(xyz), dtype=np.float64)
        return np.asarray([len(v) for v in tree.query_ball_point(np.asarray(xyz) * SCALE, radius_um)], dtype=float)


class E0cFlowField:
    """Spatial robust flow from unambiguous frozen-E0c edges.

    Spatial kNN median is primary.  When fewer than three nearby anchors exist,
    it falls back to the per-frame median, then the embryo-wide global median.
    """

    def __init__(
        self,
        pos: dict[int, np.ndarray],
        times: dict[int, int],
        edges: list[tuple[int, int]],
        *,
        neighbours: int = 8,
        radius_um: float = 30.0,
    ):
        self.neighbours = int(neighbours)
        self.radius_um = float(radius_um)
        outdeg: dict[int, int] = defaultdict(int)
        indeg: dict[int, int] = defaultdict(int)
        for source, target in edges:
            outdeg[source] += 1
            indeg[target] += 1

        forward: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        backward: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        all_disp: list[np.ndarray] = []
        for source, target in edges:
            if source not in pos or target not in pos:
                continue
            if times[target] != times[source] + 1 or outdeg[source] != 1 or indeg[target] != 1:
                continue
            disp = pos[target] - pos[source]
            forward[times[source]].append((pos[source] * SCALE, disp))
            backward[times[target]].append((pos[target] * SCALE, -disp))
            all_disp.append(disp)
        self.global_forward = np.median(np.stack(all_disp), axis=0) if all_disp else np.zeros(3)
        self.global_backward = -self.global_forward
        self.fields = {"forward": self._pack(forward), "backward": self._pack(backward)}

    @staticmethod
    def _pack(raw: dict[int, list[tuple[np.ndarray, np.ndarray]]]):
        packed = {}
        for t, rows in raw.items():
            points = np.stack([row[0] for row in rows])
            disp = np.stack([row[1] for row in rows])
            packed[t] = {
                "points": points,
                "disp": disp,
                "tree": cKDTree(points),
                "median": np.median(disp, axis=0),
            }
        return packed

    def displacement(self, t: int, xyz: np.ndarray, *, direction: str, spatial: bool) -> np.ndarray:
        field = self.fields[direction].get(int(t))
        global_flow = self.global_forward if direction == "forward" else self.global_backward
        if field is None:
            return global_flow.copy()
        if not spatial:
            return field["median"].copy()
        k = min(self.neighbours, len(field["points"]))
        distance, index = field["tree"].query(np.asarray(xyz) * SCALE, k=k)
        distance = np.atleast_1d(distance)
        index = np.atleast_1d(index)
        keep = np.isfinite(distance) & (distance <= self.radius_um)
        if keep.sum() < 3:
            return field["median"].copy()
        return np.median(field["disp"][index[keep]], axis=0)

    def tube(self, t0: int, center: np.ndarray, times: Iterable[int], *, spatial: bool) -> dict[int, np.ndarray]:
        wanted = sorted(set(int(t) for t in times))
        tube = {int(t0): np.asarray(center, dtype=np.float64).copy()}
        current = tube[t0]
        for t in range(t0, max(wanted) if wanted else t0):
            current = current + self.displacement(t, current, direction="forward", spatial=spatial)
            tube[t + 1] = current.copy()
        current = tube[t0]
        for t in range(t0, min(wanted) if wanted else t0, -1):
            current = current + self.displacement(t, current, direction="backward", spatial=spatial)
            tube[t - 1] = current.copy()
        return {t: tube[t] for t in wanted if t in tube}


def isolated_records(split: int, crop: str):
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph
    from biotrack.submission import submission_to_graphs

    gdf = (pl.read_parquet(E0C_CACHE / "graphs" / str(split) / f"{crop}.parquet")
           .with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))
    pred = submission_to_graphs(gdf)[crop]
    gt = load_graph(ROOT / "data" / "train" / f"{crop}.geff")
    pred.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    matched_attr = pred.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    matched = {int(value) for value in matched_attr[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list()
               if value not in (None, -1)}
    attrs = gt.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
    pos = {
        int(row[td.DEFAULT_ATTR_KEYS.NODE_ID]):
        (int(row["t"]), np.array([row["z"], row["y"], row["x"]], dtype=np.float64))
        for row in attrs.iter_rows(named=True)
    }
    gt_by_t: dict[int, list[np.ndarray]] = defaultdict(list)
    for t, xyz in pos.values():
        gt_by_t[t].append(xyz)
    isolated = []
    for gid in map(int, gt.node_ids()):
        if gid in matched:
            continue
        neighbours = [int(v) for v in gt.predecessors(gid)] + [int(v) for v in gt.successors(gid)]
        if any(node_id in matched for node_id in neighbours):
            continue
        isolated.append(gid)
    return gt, pos, gt_by_t, matched, isolated


def lineage_tube(gt, pos, center: int, radius: int = 3) -> dict[int, np.ndarray]:
    t0, _ = pos[center]
    tube = {t0: pos[center][1]}
    current = center
    for _ in range(radius):
        parents = [int(value) for value in gt.predecessors(current)]
        if len(parents) != 1:
            break
        current = parents[0]
        tube[pos[current][0]] = pos[current][1]
    current = center
    for _ in range(radius):
        children = [int(value) for value in gt.successors(current)]
        if len(children) != 1:
            break
        current = children[0]
        tube[pos[current][0]] = pos[current][1]
    return tube


def eligible_window(tube: dict[int, np.ndarray], t0: int, window: int) -> bool:
    radius = window // 2
    return all(t in tube for t in range(t0 - radius, t0 + radius + 1))


def frame_shape(arr) -> tuple[int, int, int]:
    return tuple(int(value) for value in arr.shape[1:])


def valid_point(xyz: np.ndarray, shape: tuple[int, int, int]) -> bool:
    point = np.asarray(xyz)
    return bool(np.isfinite(point).all() and np.all(point >= 0) and np.all(point < np.asarray(shape)))


def infer_queries(net, arr, queries, *, batch_size: int = 8):
    """Run the identical float32 patch path for every positive and control query."""
    by_t: dict[int, list[tuple[object, np.ndarray]]] = defaultdict(list)
    shape = frame_shape(arr)
    for key, t, xyz in queries:
        if 0 <= int(t) < arr.shape[0] and valid_point(xyz, shape):
            by_t[int(t)].append((key, np.asarray(xyz, dtype=np.float64)))
    result = {}
    pz, pyx = 8, 32
    for t, items in sorted(by_t.items()):
        frame = np.asarray(arr[t], dtype=np.float32)
        lo, hi = np.percentile(frame, (1.0, 99.9))
        frame = np.clip(
            (frame - np.float32(lo)) / np.float32(hi - lo + 1e-6), 0.0, 1.0
        ).astype(np.float32, copy=False)
        padded = np.pad(frame, ((pz, pz), (pyx, pyx), (pyx, pyx)), mode="reflect")
        patches = []
        for _, xyz in items:
            z, y, x = np.rint(xyz).astype(int) + np.array([pz, pyx, pyx])
            patches.append(np.ascontiguousarray(
                padded[z-pz:z+pz, y-pyx:y+pyx, x-pyx:x+pyx], dtype=np.float32
            ))
        for start in range(0, len(patches), batch_size):
            batch = torch.from_numpy(np.stack(patches[start:start + batch_size]))[:, None]
            if batch.dtype != torch.float32:
                raise TypeError(f"DAXI input must be float32, got {batch.dtype}")
            with torch.inference_mode():
                output = net(batch)[:, 0].numpy()
            for offset, volume in enumerate(output):
                key = items[start + offset][0]
                result[key] = float(volume[
                    pz-WZ:pz+WZ+1, pyx-WYX:pyx+WYX+1, pyx-WYX:pyx+WYX+1
                ].max())
    return result


def control_shortlist(
    cache_path: Path,
    *,
    t: int,
    positive_xyz: np.ndarray,
    positive_density: float,
    gt_at_t: list[np.ndarray],
    density: PredDensity,
    count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    data = np.load(cache_path)
    indices = np.flatnonzero(data["t"] == t)
    if not len(indices):
        return []
    xyz = np.stack([data["z"][indices], data["y"][indices], data["x"][indices]], axis=1).astype(float)
    if gt_at_t:
        gt_tree = cKDTree(np.asarray(gt_at_t) * SCALE)
        distance, _ = gt_tree.query(xyz * SCALE, k=1)
        xyz = xyz[distance > CONTROL_GT_EXCLUSION_UM]
    if not len(xyz):
        return []
    # Bound density work while retaining broad candidates. Cached response is
    # deliberately not used for final matching or the score.
    if len(xyz) > 2000:
        xyz = xyz[rng.choice(len(xyz), 2000, replace=False)]
    candidate_density = density.counts(t, xyz)
    depth_delta = np.abs((xyz[:, 0] - positive_xyz[0]) * SCALE[0]) / 8.0
    density_delta = np.abs(np.log1p(candidate_density) - math.log1p(positive_density))
    score = depth_delta + density_delta
    order = np.argsort(score, kind="stable")[:count]
    return [xyz[index] for index in order]


def select_events(
    isolated: list[int],
    gt,
    pos,
    density: PredDensity,
    *,
    max_events: int,
    seed: int,
) -> tuple[list[Event], dict[str, int]]:
    eligible5, eligible7 = [], 0
    for gid in isolated:
        t0, center = pos[gid]
        tube = lineage_tube(gt, pos, gid, radius=3)
        if eligible_window(tube, t0, 7):
            eligible7 += 1
        if eligible_window(tube, t0, PRIMARY_WINDOW):
            eligible5.append(Event(gid, t0, tube, density.count(t0, center)))
    # Deterministic time/depth interleaving rather than alphabetical/GT order.
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible5)
    eligible5.sort(key=lambda event: (event.t0 // 20, int(event.oracle[event.t0][0]) // 8, rng.random()))
    if len(eligible5) > max_events:
        bins: dict[tuple[int, int], list[Event]] = defaultdict(list)
        for event in eligible5:
            bins[(event.t0 // 20, int(event.oracle[event.t0][0]) // 8)].append(event)
        chosen = []
        while len(chosen) < max_events and any(bins.values()):
            for key in sorted(bins):
                if bins[key] and len(chosen) < max_events:
                    chosen.append(bins[key].pop())
        eligible5 = chosen
    return eligible5, {"eligible5": len([gid for gid in isolated if eligible_window(lineage_tube(gt, pos, gid), pos[gid][0], 5)]),
                       "eligible7": eligible7}


def crop_seed(crop: str, base: int) -> int:
    digest = hashlib.sha256(f"{base}:{crop}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def prepare_crop(split: int, crop: str, *, max_events: int, seed: int):
    pred_pos, pred_times, pred_edges = load_pred_nodes_edges(split, crop)
    density = PredDensity(pred_pos, pred_times)
    gt, gt_pos, gt_by_t, matched, isolated = isolated_records(split, crop)
    events, eligible = select_events(
        isolated, gt, gt_pos, density, max_events=max_events, seed=crop_seed(crop, seed)
    )
    coverage = {
        "crop": crop,
        "family": crop.split("_", 1)[0],
        "split": split,
        "gt_nodes": len(gt_pos),
        "matched_nodes": len(matched),
        "missed_nodes": len(gt_pos) - len(matched),
        "isolated_misses": len(isolated),
        **eligible,
        "selected": len(events),
        "selected_t_min": min((event.t0 for event in events), default=None),
        "selected_t_max": max((event.t0 for event in events), default=None),
        "selected_z_min": min((float(event.oracle[event.t0][0]) for event in events), default=None),
        "selected_z_max": max((float(event.oracle[event.t0][0]) for event in events), default=None),
        "selected_density_median": float(np.median([event.density for event in events])) if events else None,
    }
    return events, coverage, gt_by_t, density, E0cFlowField(pred_pos, pred_times, pred_edges)


def score_crop(
    split: int,
    crop: str,
    cache_path: Path,
    net,
    *,
    max_events: int,
    shortlist: int,
    seed: int,
):
    events, coverage, gt_by_t, density, flow = prepare_crop(
        split, crop, max_events=max_events, seed=seed
    )
    if not events:
        return [], coverage
    arr = zarr.open(ROOT / "data/train" / f"{crop}.zarr" / "0", mode="r")
    shape = frame_shape(arr)
    rng = np.random.default_rng(crop_seed(crop, seed))

    # Exact positive-center response plus depth/density-matched candidate shortlist.
    positive_queries = [(('positive', index), event.t0, event.oracle[event.t0])
                        for index, event in enumerate(events)]
    positive_response = infer_queries(net, arr, positive_queries)
    shortlists = []
    control_queries = []
    for index, event in enumerate(events):
        candidates = control_shortlist(
            cache_path,
            t=event.t0,
            positive_xyz=event.oracle[event.t0],
            positive_density=event.density,
            gt_at_t=gt_by_t.get(event.t0, []),
            density=density,
            count=shortlist,
            rng=rng,
        )
        shortlists.append(candidates)
        control_queries.extend([(('control', index, j), event.t0, xyz)
                                for j, xyz in enumerate(candidates)])
    exact_control_response = infer_queries(net, arr, control_queries)

    pairs = []
    for index, event in enumerate(events):
        key = ('positive', index)
        if key not in positive_response:
            continue
        candidates = shortlists[index]
        available = [(j, xyz, exact_control_response[('control', index, j)])
                     for j, xyz in enumerate(candidates)
                     if ('control', index, j) in exact_control_response]
        if not available:
            continue
        _, control_xyz, control_center_response = min(
            available, key=lambda item: abs(item[2] - positive_response[key])
        )
        pairs.append((event, control_xyz, positive_response[key], control_center_response))

    queries = []
    metadata = []
    for pair_index, (event, control_center, pos_center_response, ctrl_center_response) in enumerate(pairs):
        t0 = event.t0
        all_times = range(t0 - 3, t0 + 4)
        pos_spatial = flow.tube(t0, event.oracle[t0], all_times, spatial=True)
        ctrl_spatial = flow.tube(t0, control_center, all_times, spatial=True)
        pos_frame = flow.tube(t0, event.oracle[t0], all_times, spatial=False)
        ctrl_frame = flow.tube(t0, control_center, all_times, spatial=False)
        oracle_control = {t: control_center + (xyz - event.oracle[t0]) for t, xyz in event.oracle.items()}
        tubes = {
            "pos_static": {t: event.oracle[t0] for t in all_times},
            "ctrl_static": {t: control_center for t in all_times},
            "pos_oracle": event.oracle,
            "ctrl_oracle": oracle_control,
            "pos_spatial": pos_spatial,
            "ctrl_spatial": ctrl_spatial,
            "pos_frame": pos_frame,
            "ctrl_frame": ctrl_frame,
        }
        invalid = any(not valid_point(tube[t], shape) for tube in tubes.values() for t in tube)
        if invalid:
            continue
        for method, tube in tubes.items():
            for t, xyz in tube.items():
                queries.append(((pair_index, method, t), t, xyz))
        metadata.append((pair_index, event, control_center, pos_center_response, ctrl_center_response, tubes))
    response = infer_queries(net, arr, queries)

    rows = []
    for pair_index, event, control_center, pos_center_response, ctrl_center_response, tubes in metadata:
        row = {
            "crop": crop,
            "family": crop.split("_", 1)[0],
            "split": split,
            "gid": event.gid,
            "t0": event.t0,
            "z": float(event.oracle[event.t0][0]),
            "y": float(event.oracle[event.t0][1]),
            "x": float(event.oracle[event.t0][2]),
            "control_z": float(control_center[0]),
            "control_y": float(control_center[1]),
            "control_x": float(control_center[2]),
            "pred_density": event.density,
            "positive_center": pos_center_response,
            "control_center": ctrl_center_response,
            "center_match_abs": abs(pos_center_response - ctrl_center_response),
        }
        complete = True
        for window in WINDOWS:
            radius = window // 2
            times = list(range(event.t0 - radius, event.t0 + radius + 1))
            if not all(t in event.oracle for t in times):
                continue
            for method in ("static", "oracle", "spatial", "frame"):
                for kind in ("pos", "ctrl"):
                    key = f"{kind}_{method}"
                    values = [response.get((pair_index, key, t)) for t in times]
                    if any(value is None for value in values):
                        complete = False
                        break
                    row[f"{kind}_{method}_{window}"] = float(np.mean(values))
        if not complete or "pos_spatial_5" not in row:
            continue
        for method in ("oracle", "spatial", "frame"):
            for kind in ("pos", "ctrl"):
                row[f"{kind}_{method}_gain5"] = row[f"{kind}_{method}_5"] - row[f"{kind}_static_5"]
            row[f"paired_{method}_contrast5"] = (
                row[f"pos_{method}_gain5"] - row[f"ctrl_{method}_gain5"]
            )
        oracle5 = np.stack([event.oracle[t] for t in range(event.t0 - 2, event.t0 + 3)])
        spatial5 = np.stack([tubes["pos_spatial"][t] for t in range(event.t0 - 2, event.t0 + 3)])
        frame5 = np.stack([tubes["pos_frame"][t] for t in range(event.t0 - 2, event.t0 + 3)])
        row["spatial_oracle_error_um"] = float(np.mean(np.linalg.norm((spatial5 - oracle5) * SCALE, axis=1)))
        row["frame_oracle_error_um"] = float(np.mean(np.linalg.norm((frame5 - oracle5) * SCALE, axis=1)))
        rows.append(row)
    coverage["paired"] = len(rows)
    return rows, coverage


def auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    npos, nneg = int(labels.sum()), int((1 - labels).sum())
    if not npos or not nneg:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def crop_bootstrap(rows: list[dict], field: str, *, draws: int, seed: int):
    crop_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        crop_values[row["crop"]].append(float(row[field]))
    crop_means = np.asarray([np.mean(values) for values in crop_values.values()], dtype=float)
    if not len(crop_means):
        return {"n_crops": 0, "mean": None, "ci_low": None, "ci_high": None}
    if len(crop_means) == 1:
        return {"n_crops": 1, "mean": float(crop_means[0]), "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(crop_means, size=(draws, len(crop_means)), replace=True), axis=1)
    return {
        "n_crops": len(crop_means),
        "mean": float(crop_means.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def choose_threshold(rows: list[dict], *, max_control_fpr: float = 0.05):
    positive = np.asarray([row["pos_spatial_gain5"] for row in rows], dtype=float)
    control = np.asarray([row["ctrl_spatial_gain5"] for row in rows], dtype=float)
    if not len(positive) or not len(control):
        return None
    thresholds = np.unique(np.concatenate([positive, control]))
    best = None
    for threshold in thresholds:
        fpr = float(np.mean(control >= threshold))
        tpr = float(np.mean(positive >= threshold))
        if fpr <= max_control_fpr and (best is None or (tpr, threshold) > (best[1], best[0])):
            best = (float(threshold), tpr, fpr)
    return best


def evaluate_threshold(rows: list[dict], threshold: float):
    positive = np.asarray([row["pos_spatial_gain5"] for row in rows], dtype=float)
    control = np.asarray([row["ctrl_spatial_gain5"] for row in rows], dtype=float)
    return {"threshold": threshold, "tpr": float(np.mean(positive >= threshold)),
            "control_fpr": float(np.mean(control >= threshold)), "events": len(rows)}


def summarise(rows: list[dict], coverage: list[dict], *, draws: int, seed: int):
    result = {"coverage": coverage, "families": {}, "crossfit": {}}
    by_family = {family: [row for row in rows if row["family"] == family] for family in ("44b6", "6bba")}
    for family, family_rows in by_family.items():
        labels = [1] * len(family_rows) + [0] * len(family_rows)
        result["families"][family] = {
            "events": len(family_rows),
            "crops": len(set(row["crop"] for row in family_rows)),
            "primary_spatial_contrast": crop_bootstrap(
                family_rows, "paired_spatial_contrast5", draws=draws, seed=seed + (0 if family == "44b6" else 1)
            ),
            "oracle_contrast": crop_bootstrap(
                family_rows, "paired_oracle_contrast5", draws=draws, seed=seed + 10
            ),
            "frame_contrast_diagnostic": crop_bootstrap(
                family_rows, "paired_frame_contrast5", draws=draws, seed=seed + 20
            ),
            "spatial_gain_auc": auc(labels, [row["pos_spatial_gain5"] for row in family_rows]
                                      + [row["ctrl_spatial_gain5"] for row in family_rows]),
            "median_center_match_abs": float(np.median([row["center_match_abs"] for row in family_rows]))
            if family_rows else None,
            "median_spatial_oracle_error_um": float(np.median([row["spatial_oracle_error_um"] for row in family_rows]))
            if family_rows else None,
        }
    for train, test in (("44b6", "6bba"), ("6bba", "44b6")):
        selected = choose_threshold(by_family[train])
        key = f"train_{train}_test_{test}"
        result["crossfit"][key] = (
            {"train": {"threshold": selected[0], "tpr": selected[1], "control_fpr": selected[2]},
             "test": evaluate_threshold(by_family[test], selected[0])}
            if selected is not None and by_family[test] else None
        )
    family_gate = []
    for family in ("44b6", "6bba"):
        stats = result["families"][family]["primary_spatial_contrast"]
        family_gate.append(stats["n_crops"] >= 2 and stats["ci_low"] is not None and stats["ci_low"] > 0)
    crossfit_gate = all(value is not None and value["test"]["tpr"] >= 0.15
                        and value["test"]["control_fpr"] <= 0.05
                        for value in result["crossfit"].values())
    result["signal_gate_pass"] = bool(all(family_gate) and crossfit_gate)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        atomic_text(path, "")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daxi-dir", type=Path, action="append",
                        help="repeatable; duplicate crop names are de-duplicated")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-all-family", choices=("44b6", "6bba"),
                        help="audit every frozen-OOF crop in one family; DAXI cache not required")
    parser.add_argument("--max-events-per-crop", type=int, default=30)
    parser.add_argument("--control-shortlist", type=int, default=24)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/inventory/daxi_accumulation_v3.json")
    args = parser.parse_args()

    if args.audit_all_family:
        if not args.audit_only:
            raise SystemExit("--audit-all-family requires --audit-only")
        split = 0 if args.audit_all_family == "44b6" else 1
        cache_files = [Path(f"{path.stem}.npz") for path in
                       sorted((E0C_CACHE / "graphs" / str(split)).glob(f"{args.audit_all_family}_*.parquet"))]
    else:
        if not args.daxi_dir:
            raise SystemExit("--daxi-dir is required unless --audit-all-family is used")
        by_crop = {}
        for directory in args.daxi_dir:
            for path in sorted(directory.glob("*.npz")):
                by_crop[path.stem] = path
        cache_files = [by_crop[crop] for crop in sorted(by_crop)]
    if not cache_files:
        raise SystemExit("no crops found for audit/inference")
    torch.set_num_threads(max(1, args.torch_threads))
    coverage, all_rows = [], []
    net = None
    if not args.audit_only:
        net = torch.jit.load(str(ROOT / "weights/unet-daxi.pt"), map_location="cpu")
        net.eval()

    for cache_path in cache_files:
        crop = cache_path.stem
        split = crop_split(crop)
        if args.audit_only:
            events, cov, _, _, _ = prepare_crop(
                split, crop, max_events=args.max_events_per_crop, seed=args.seed
            )
            coverage.append(cov)
            print(f"AUDIT {crop}: isolated={cov['isolated_misses']} eligible5={cov['eligible5']} "
                  f"eligible7={cov['eligible7']} selected={len(events)}", flush=True)
        else:
            rows, cov = score_crop(
                split, crop, cache_path, net,
                max_events=args.max_events_per_crop,
                shortlist=args.control_shortlist,
                seed=args.seed,
            )
            coverage.append(cov)
            all_rows.extend(rows)
            print(f"SCORE {crop}: eligible5={cov['eligible5']} paired={cov.get('paired', 0)}", flush=True)

    family_coverage = {}
    for family in ("44b6", "6bba"):
        rows = [row for row in coverage if row["family"] == family]
        family_coverage[family] = {
            "crops": len(rows),
            "informative_crops": sum(row["eligible5"] > 0 for row in rows),
            "isolated_misses": sum(row["isolated_misses"] for row in rows),
            "eligible5": sum(row["eligible5"] for row in rows),
            "eligible7": sum(row["eligible7"] for row in rows),
            "selected": sum(row["selected"] for row in rows),
        }
    if args.audit_only:
        summary = {"mode": "audit", "family_coverage": family_coverage, "coverage": coverage}
    else:
        summary = summarise(all_rows, coverage, draws=args.bootstrap_draws, seed=args.seed)
        summary.update(mode="inference", family_coverage=family_coverage,
                       preregistered_primary="5-frame spatial-flow incremental paired contrast")

    atomic_text(args.out, json.dumps(summary, indent=2, sort_keys=True))
    write_csv(args.out.with_name(args.out.stem + "_coverage.csv"), coverage)
    if not args.audit_only:
        write_csv(args.out.with_name(args.out.stem + "_events.csv"), all_rows)
    print(json.dumps({"out": str(args.out), "family_coverage": family_coverage,
                      "signal_gate_pass": summary.get("signal_gate_pass")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
