r"""GATE 1 of the association harness: cache frozen-trunk node features, and PROVE they are the
features the deployed head actually used.

WHY THIS IS GATE 1
------------------
Every later gate - label contract, listwise evaluation, full-chain - rests on training a new head
on cached trunk features instead of re-running the 3D UNet for every experiment. That is only
legitimate if the cache is bit-faithful to what the deployed path computes. If it is not, a new
head trains on subtly different inputs and every downstream number silently measures the cache
rather than the model. So the cache is not trusted because it was written carefully; it is trusted
because the DEPLOYED head, re-run from the cache alone, reproduces the candidate edges and
probabilities already recorded in the pre-ILP export.

THE PARITY TARGET IS AN ARTIFACT WE ALREADY HAVE
------------------------------------------------
The P30 pre-ILP export records, per crop, every candidate edge the deployed pipeline kept:
``(source_id, target_id, edge_prob)`` after softmax over the SOURCE axis and the 0.5 threshold
(``FACT-0369``). Node ids there are positional indices into ``coords_so_far``, verified against the
detector peak order (``FACT-0373`` check D). So the gate is exact and needs no new ground truth:
rebuild features, re-run ``predict_edges``, apply the same softmax and threshold, and require the
resulting candidate set and probabilities to match what was recorded.

WHAT IS MIRRORED, AND WHY NOT IMPORTED
--------------------------------------
The deployed windowing lives inside ``predict_video``, a script function that also runs detection
and I/O, so it cannot be called for features alone. The window loop, the window-relative time used
for positional features, and the source-axis softmax are therefore mirrored here from
``predict_unet_transformer.py``, in the same way ``ilp_replay`` mirrors ``build_graph`` - and the
parity gate exists precisely because mirroring is a risk.

SCOPE
-----
This caches and checks ONE crop at a time. It is deliberately CPU-runnable so the contract can be
proven before any GPU session is spent on training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "vendor/kaggle-cell-tracking/src"))
sys.path.insert(0, str(ROOT / "vendor/kaggle-cell-tracking/scripts"))


def build_cache(
    crop: str,
    data_dir: Path,
    weights: Path,
    det_threshold: float,
    device: torch.device,
    max_frames: int | None = None,
) -> dict:
    """Run the frozen trunk and cache per-node features, coords and positional features."""
    import predict_unet_transformer as P
    from tracking_cellmot.io import open_dataset

    model, window_size, downsample = P.load_model(weights, device)
    cfg = P.PredictConfig(det_threshold=det_threshold)
    ds = open_dataset(data_dir / f"{crop}.zarr")
    images = ds["images"] if "images" in ds else ds
    image_shape = tuple(images.shape)
    n_frames = image_shape[0] if max_frames is None else min(max_frames, image_shape[0])

    ds_arr = np.asarray(downsample, dtype=np.float32)
    voxel_size = np.asarray(P.VOXEL_SIZE_UM, dtype=np.float32) if hasattr(P, "VOXEL_SIZE_UM") \
        else np.asarray([1.625, 0.40625, 0.40625], dtype=np.float32)
    pool_k = P.pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)

    W = window_size
    coord_lists: list[np.ndarray] = []
    coord_offset: dict[int, tuple[int, int]] = {}
    seen_frames: set[int] = set()
    global_node_count = 0
    feats: dict[int, np.ndarray] = {}

    starts = list(range(0, max(n_frames - 1, 1), max(W - 1, 1)))
    for start in starts:
        frame_indices = [t for t in range(start, start + W) if t < n_frames]
        if len(frame_indices) < 2:
            continue
        volume = np.stack([np.asarray(images[t]) for t in frame_indices])[None]
        tensor = torch.from_numpy(volume.astype(np.float32)).to(device)
        with torch.no_grad():
            unet_out = model.unet(tensor)
            det_logits = model.detection_head(unet_out)

        for f_idx, t in enumerate(frame_indices):
            if t in seen_frames:
                continue
            arr = P._detect_cells_pooled(det_logits[0][f_idx][0], t, cfg.det_threshold, pool_k)
            coord_offset[t] = (global_node_count, global_node_count + len(arr))
            global_node_count += len(arr)
            coord_lists.append(arr)
            seen_frames.add(t)

        coords_so_far = (
            np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
        )
        for f_idx, t in enumerate(frame_indices):
            if t not in coord_offset:
                continue
            s, e = coord_offset[t]
            if e == s or t in feats:
                continue
            c = coords_so_far[s:e]
            p_coords = torch.from_numpy(c[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            p_mask = torch.ones(1, len(c), dtype=torch.bool, device=device)
            with torch.no_grad():
                f = model._index_features(unet_out[:, f_idx], p_coords, p_mask)
            feats[t] = f[0].cpu().numpy().astype(np.float32)
        del unet_out, det_logits

    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    return {
        "crop": crop,
        "coords": coords,          # (N, 4) = t, z, y, x on the DOWNSAMPLED grid
        "coord_offset": coord_offset,
        "features": feats,         # frame -> (n_t, 32)
        "window_size": W,
        "downsample": tuple(int(d) for d in downsample),
        "image_shape": image_shape,
        "det_threshold": det_threshold,
    }


def candidates_from_cache(cache: dict, weights: Path, device: torch.device,
                          threshold: float) -> dict[tuple[int, int], float]:
    """Re-run the DEPLOYED head from cached features and return its candidate edges.

    Mirrors predict_unet_transformer: softmax over the SOURCE axis, keep pairs above `threshold`,
    and address nodes by their global index into `coords_so_far`.
    """
    import predict_unet_transformer as P

    model, window_size, downsample = P.load_model(weights, device)
    ds_arr_t = torch.tensor(np.asarray(downsample, dtype=np.float32), device=device)
    coords, offset, feats = cache["coords"], cache["coord_offset"], cache["features"]
    W = cache["window_size"]
    window_shape = (W,) + tuple(cache["image_shape"][1:])

    out: dict[tuple[int, int], float] = {}
    frames = sorted(offset)
    for t_src, t_tgt in zip(frames[:-1], frames[1:]):
        if t_tgt != t_src + 1 or t_src not in feats or t_tgt not in feats:
            continue
        s_src, e_src = offset[t_src]
        s_tgt, e_tgt = offset[t_tgt]
        if e_src == s_src or e_tgt == s_tgt:
            continue
        c_src, c_tgt = coords[s_src:e_src], coords[s_tgt:e_tgt]
        n_src, n_tgt = len(c_src), len(c_tgt)

        p_coords_src = torch.from_numpy(c_src[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
        p_coords_tgt = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
        # window-relative time, exactly as deployed
        c_src_rel, c_tgt_rel = c_src.copy(), c_tgt.copy()
        c_src_rel[:, 0], c_tgt_rel[:, 0] = 0, 1
        p_pos_src = torch.from_numpy(
            P.extract_pos_features(c_src_rel, window_shape)).unsqueeze(0).to(device)
        p_pos_tgt = torch.from_numpy(
            P.extract_pos_features(c_tgt_rel, window_shape)).unsqueeze(0).to(device)
        f_src = torch.from_numpy(feats[t_src]).unsqueeze(0).to(device)
        f_tgt = torch.from_numpy(feats[t_tgt]).unsqueeze(0).to(device)
        m_src = torch.ones(1, n_src, dtype=torch.bool, device=device)
        m_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=device)

        with torch.no_grad():
            logits = model.predict_edges(
                f_src, f_tgt, p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                p_pos_src, p_pos_tgt, m_src, m_tgt,
            )
        probs = torch.softmax(logits[0], dim=0).cpu().numpy()
        for i in range(n_src):
            for j in range(n_tgt):
                if probs[i, j] > threshold:
                    out[(s_src + i, s_tgt + j)] = float(probs[i, j])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crop", required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--preilp", type=Path, required=True)
    ap.add_argument("--det-threshold", type=float, default=0.96875)
    ap.add_argument("--edge-threshold", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--prob-tolerance", type=float, default=1e-4)
    ap.add_argument("--cache-out", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = build_cache(args.crop, args.data_dir, args.weights, args.det_threshold,
                        device, args.max_frames)
    got = candidates_from_cache(cache, args.weights, device, args.edge_threshold)

    pre = pl.read_parquet(args.preilp).filter(pl.col("dataset") == args.crop)
    edges = pre.filter(pl.col("row_type") == "edge")
    recorded = {
        (int(s), int(t)): float(p)
        for s, t, p in zip(edges["source_id"], edges["target_id"], edges["edge_prob"])
    }
    # A frame cap makes the comparison partial by construction; restrict to covered frames.
    if args.max_frames:
        covered = set(cache["coord_offset"])
        node_t = {}
        for t, (s, e) in cache["coord_offset"].items():
            for n in range(s, e):
                node_t[n] = t
        recorded = {
            k: v for k, v in recorded.items()
            if node_t.get(k[0]) in covered and node_t.get(k[1]) in covered
        }

    missing = sorted(set(recorded) - set(got))
    extra = sorted(set(got) - set(recorded))
    shared = sorted(set(recorded) & set(got))
    deltas = [abs(got[k] - recorded[k]) for k in shared]
    max_delta = max(deltas) if deltas else 0.0
    passed = not missing and not extra and max_delta <= args.prob_tolerance

    result = {
        "schema_version": 1,
        "crop": args.crop,
        "gate": "feature_parity",
        "nodes_cached": int(len(cache["coords"])),
        "frames_cached": len(cache["coord_offset"]),
        "recorded_edges": len(recorded),
        "reproduced_edges": len(got),
        "missing_from_reproduction": len(missing),
        "extra_in_reproduction": len(extra),
        "max_abs_prob_delta": max_delta,
        "prob_tolerance": args.prob_tolerance,
        "passed": bool(passed),
        "examples_missing": missing[:10],
        "examples_extra": extra[:10],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.cache_out:
        args.cache_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.cache_out,
            coords=cache["coords"],
            frames=np.array(sorted(cache["features"])),
            **{f"feat_{t}": v for t, v in cache["features"].items()},
        )

    print(
        f"\nASSOC_FEATURE_PARITY crop={args.crop} passed={passed}\n"
        f"  nodes cached      {result['nodes_cached']:,} over {result['frames_cached']} frames\n"
        f"  recorded edges    {result['recorded_edges']:,}\n"
        f"  reproduced edges  {result['reproduced_edges']:,}\n"
        f"  missing           {result['missing_from_reproduction']:,}  "
        f"extra {result['extra_in_reproduction']:,}\n"
        f"  max |dprob|       {max_delta:.3e}  (tolerance {args.prob_tolerance:.0e})"
    )
    if not passed:
        raise SystemExit(
            "FEATURE PARITY FAILED - the cache is not what the deployed head used, so no head "
            "trained on it would be measuring the deployed substrate"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
