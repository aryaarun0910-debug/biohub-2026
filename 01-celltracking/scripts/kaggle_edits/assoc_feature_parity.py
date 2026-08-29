# =====================================================================================
# GATE 1 OF THE ASSOCIATION HARNESS: FEATURE PARITY  (PKT-0029)  -- ATTEMPT 2
#
# WHY. Every later gate - label contract, listwise evaluation, full-chain - depends on training
# association heads from CACHED frozen-trunk features rather than re-running the 3D UNet for each
# experiment. That shortcut is only legitimate if the cache is faithful to what the deployed head
# actually consumed. If it is not, a new head trains on subtly different inputs and every
# downstream number silently measures the cache instead of the model.
#
# WHY ATTEMPT 1 FAILED, AND WHY THIS IS A REDESIGN RATHER THAN A REPAIR (FACT-0387, EXP-0040).
# Attempt 1 did `import predict_unet_transformer` in the NOTEBOOK process and raised
# ModuleNotFoundError, comparing zero crops. The deployed predictor is never imported into that
# process at all: it is launched as a SUBPROCESS, `/usr/bin/python3 scripts/predict_unet_transformer.py`,
# with the repo directory as its working directory, and every patch in this tree that works reaches
# it by REWRITING THAT FILE'S SOURCE TEXT. Adding `sys.path` would have made the gate run while
# still leaving two deeper defects the PKT-0033 audit had already identified BEFORE that run
# returned. All three are fixed here, and none of the three is optional:
#
#   (i)  IT RUNS WHERE THE DEPLOYED HEAD RUNS. The check is a standalone script executed as a
#        subprocess with cwd=REPO_DIR - the same interpreter, the same working directory and the
#        same import path the deployed shards use. It therefore imports the PATCHED predictor
#        source, which is what actually ran.
#   (ii) IT ACTUALLY EXERCISES A CACHE. Attempt 1 wrote features to an in-memory dict and read
#        them back in the same process, so it tested no cache at all - every dtype, layout,
#        index-remap and staleness error a real cache can carry went untested, while the gate's
#        stated purpose is to prove the head can be re-run FROM THE CACHE ALONE. Here the work is
#        split across TWO SEPARATE PROCESSES: phase `cache` writes .npz to disk and exits; phase
#        `verify` starts fresh, loads only those files, and reproduces the edges. The verifier
#        cannot see the writer's memory, so "from the cache alone" is enforced by the operating
#        system rather than by intent.
#   (iii) IT COMPARES THE BAND THE TASK ACTUALLY USES. Attempt 1 recorded a reproduced pair only
#        when its probability exceeded the deployed 0.5, so it validated exactly the band the
#        learnable task does not use: FACT-0382 measured that ALL 691 fold-0 contested errors have
#        their true parent BELOW that threshold. This version checks two bands - the deployed
#        edges from the pre-ILP parquet above 0.5, and the ECB sidecar surface down to 0.02, which
#        is where parent ranking is actually learned.
#
# PLUS THE NODE-COUNT ASSERTION the audit asked for: per frame in scope, the reproduced detection
# count must equal the recorded one. Without it a short detection silently shrinks the comparison
# denominator and the gate passes on fewer pairs than it should have compared.
#
# ONE KNOWN BLIND SPOT, STATED HERE RATHER THAN DISCOVERED LATER. This gate compares probabilities
# AFTER a softmax over the SOURCE axis, and softmax is invariant to a constant offset along the
# axis it normalises. A cache error that shifts EVERY SOURCE IN A FRAME BY THE SAME AMOUNT is
# therefore mathematically invisible to it - not merely undetected in practice. Any error that is
# not uniform across the source axis, including a single node's features being wrong, does show up.
# The limitation was found by the test suite before a GPU session was spent believing the gate was
# total, and tests/test_assoc_feature_parity.py asserts BOTH halves of it so neither can be
# forgotten. If a stronger contract is ever needed, comparing pre-softmax logits would close it -
# but nothing in the pre-ILP export or the ECB sidecars records logits, so that would require a new
# acquisition rather than a change here.
#
# WHY IT STILL RUNS ON GPU. Unchanged and still the scientific reason: CPU and GPU floating-point
# paths can differ by more than the 1e-4 probability tolerance, so a CPU smoke would confound a
# numeric-backend difference with a genuine cache error - the exact ambiguity this gate exists to
# remove. (tracking_cellmot.io.open_dataset also calls pin_memory() and refuses to run without an
# accelerator, but that is a TRANSPORT optimisation and is not the reason.)
#
# PASSIVE. Adds a diagnostic cell, writes no submission and alters no graph.
# FAIL CLOSED. Any exception, a missing input, a zero-crop comparison or a node-count mismatch
# leaves all_passed False and prints AFP_FAILED. A gate that can quietly compare nothing looks
# exactly like a gate that passed.
# =====================================================================================
import json as _afp_json
import os as _afp_os
import subprocess as _afp_sub
import sys as _afp_sys
from pathlib import Path as _AfpPath

_afp_report = {"gate": "feature_parity", "attempt": 2, "crops": [], "all_passed": False}

_AFP_WORKER = r'''
"""Gate 1 worker. Runs INSIDE the predictor's own process (cwd=REPO_DIR), in two phases.

`cache`  : build detector coords and frozen-trunk node features, write them to disk, exit.
`verify` : start clean, load ONLY those files, re-run the deployed predict_edges, compare.

The phases are separate PROCESSES on purpose. If they shared one, the cache would never be
serialised and the gate would prove nothing about it - which is precisely how attempt 1 failed
its own stated purpose while appearing well formed.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

import predict_unet_transformer as AFP   # resolves because cwd is the repo root

VOXEL = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float32)
TOL = 1e-4


def load_trunk(weights, device):
    model, W, ds = AFP.load_model(Path(weights), device)
    cfg = AFP.PredictConfig(det_threshold=float(os.environ.get("BIOHUB_DET_THRESHOLD", "0.96875")))
    return model, W, np.asarray(ds, dtype=np.float32), cfg


def phase_cache(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, W, ds, cfg = load_trunk(args.weights, device)
    pool_k = AFP.pool_kernel_from_um(cfg.pool_kernel_um, VOXEL)
    out = Path(args.cache_dir); out.mkdir(parents=True, exist_ok=True)

    for crop in args.crops:
        image = AFP.open_dataset(Path(args.test_dir) / f"{crop}.zarr").image
        n = min(args.max_frames, image.shape[0])
        coord_lists, offset, feats, seen, total = [], {}, {}, set(), 0
        for start in range(0, max(n - 1, 1), max(W - 1, 1)):
            frames = [t for t in range(start, start + W) if t < n]
            if len(frames) < 2:
                continue
            vol = np.stack([np.asarray(image[t]) for t in frames])[None]
            ten = torch.from_numpy(vol.astype(np.float32)).to(device)
            with torch.no_grad():
                unet_out = model.unet(ten)
                det = model.detection_head(unet_out)
            for fi, t in enumerate(frames):
                if t in seen:
                    continue
                arr = AFP._detect_cells_pooled(det[0][fi][0], t, cfg.det_threshold, pool_k)
                offset[t] = (total, total + len(arr))
                total += len(arr)
                coord_lists.append(arr)
                seen.add(t)
            allc = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
            for fi, t in enumerate(frames):
                if t not in offset or t in feats:
                    continue
                s, e = offset[t]
                if e == s:
                    continue
                pc = torch.from_numpy(allc[s:e, 1:].astype(np.float32)).unsqueeze(0).to(device)
                pm = torch.ones(1, e - s, dtype=torch.bool, device=device)
                with torch.no_grad():
                    feats[t] = model._index_features(unet_out[:, fi], pc, pm)[0].cpu().numpy()
            del unet_out, det

        coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
        frames_sorted = sorted(offset)
        # Everything predict_edges consumes must survive the process boundary: the node features,
        # the coordinates it scales, and the volume shape extract_pos_features needs. A cache of
        # node features alone would not let the verifier reconstruct the position features, whose
        # frame index is rewritten to window-local 0/1.
        np.savez_compressed(
            out / f"{crop}.npz",
            coords=coords,
            frames=np.asarray(frames_sorted, dtype=np.int64),
            starts=np.asarray([offset[t][0] for t in frames_sorted], dtype=np.int64),
            ends=np.asarray([offset[t][1] for t in frames_sorted], dtype=np.int64),
            feat_frames=np.asarray(sorted(feats), dtype=np.int64),
            image_shape=np.asarray(image.shape[1:], dtype=np.int64),
            window=np.int64(W),
            downsample=ds,
            **{f"feat_{t}": feats[t] for t in feats},
        )
        print(f"AFP_CACHE crop={crop} frames={len(frames_sorted)} nodes={len(coords)}", flush=True)
    return 0


def phase_verify(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _W, _ds, _cfg = load_trunk(args.weights, device)
    pre = pl.read_parquet(args.preilp)
    report = {"gate": "feature_parity", "attempt": 2, "crops": []}

    for crop in args.crops:
        cache_path = Path(args.cache_dir) / f"{crop}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"cache missing for {crop} - the cache phase did not write it")
        z = np.load(cache_path, allow_pickle=False)
        coords = z["coords"]
        frames = z["frames"].tolist()
        offset = {int(t): (int(s), int(e)) for t, s, e in zip(frames, z["starts"], z["ends"])}
        feats = {int(t): z[f"feat_{int(t)}"] for t in z["feat_frames"].tolist()}
        shape = (int(z["window"]),) + tuple(int(v) for v in z["image_shape"])
        ds_t = torch.tensor(z["downsample"], device=device)

        node_rows = pre.filter((pl.col("dataset") == crop) & (pl.col("row_type") == "node"))
        rec_per_frame = dict(
            zip(node_rows["t"].to_numpy().astype(int).tolist(),
                [0] * node_rows.height)
        )
        for t in node_rows["t"].to_numpy().astype(int).tolist():
            rec_per_frame[t] = rec_per_frame.get(t, 0) + 1
        # NODE-COUNT PARITY. Without this a short detection shrinks the denominator and the gate
        # passes having compared fewer pairs than it should have.
        count_mismatch = [
            {"t": t, "cached": e - s, "recorded": rec_per_frame.get(t, 0)}
            for t, (s, e) in sorted(offset.items())
            if rec_per_frame.get(t, 0) != e - s
        ]

        got = {}
        for ts, tt in zip(frames[:-1], frames[1:]):
            if tt != ts + 1 or ts not in feats or tt not in feats:
                continue
            ss, se = offset[ts]
            tsq, te = offset[tt]
            if se == ss or te == tsq:
                continue
            cs_, ct_ = coords[ss:se], coords[tsq:te]
            pcs = torch.from_numpy(cs_[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            pct = torch.from_numpy(ct_[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            csr, ctr = cs_.copy(), ct_.copy()
            csr[:, 0], ctr[:, 0] = 0, 1
            pps = torch.from_numpy(AFP.extract_pos_features(csr, shape)).unsqueeze(0).to(device)
            ppt = torch.from_numpy(AFP.extract_pos_features(ctr, shape)).unsqueeze(0).to(device)
            fs = torch.from_numpy(feats[ts]).unsqueeze(0).to(device)
            ft = torch.from_numpy(feats[tt]).unsqueeze(0).to(device)
            ms = torch.ones(1, len(cs_), dtype=torch.bool, device=device)
            mt = torch.ones(1, len(ct_), dtype=torch.bool, device=device)
            with torch.no_grad():
                logits = model.predict_edges(fs, ft, pcs * ds_t, pct * ds_t, pps, ppt, ms, mt)
            probs = torch.softmax(logits[0], dim=0).cpu().numpy()   # softmax over the SOURCE axis
            # The FULL matrix is retained, not only pairs above 0.5. Band B lives below it.
            for i in range(len(cs_)):
                for j in range(len(ct_)):
                    got[(ss + i, tsq + j)] = float(probs[i, j])

        node_t = {n: t for t, (s, e) in offset.items() for n in range(s, e)}
        in_scope = lambda k: node_t.get(k[0]) in offset and node_t.get(k[1]) in offset

        # --- BAND A: the deployed edges, above 0.5 -------------------------------------
        edge_rows = pre.filter((pl.col("dataset") == crop) & (pl.col("row_type") == "edge"))
        band_a = {(int(a), int(b)): float(p) for a, b, p in
                  zip(edge_rows["source_id"], edge_rows["target_id"], edge_rows["edge_prob"])
                  if in_scope((int(a), int(b)))}
        got_a = {k: v for k, v in got.items() if v > 0.5}
        a_missing = sorted(set(band_a) - set(got_a))
        a_extra = sorted(set(got_a) - set(band_a))
        a_shared = set(band_a) & set(got_a)
        a_delta = max((abs(got_a[k] - band_a[k]) for k in a_shared), default=0.0)

        # --- BAND B: the sub-threshold surface the learnable task uses ------------------
        band_b, b_delta, b_missing, b_checked = {}, 0.0, [], 0
        sidecar = Path(args.ecb_dir) / f"{crop}.npz" if args.ecb_dir else None
        if sidecar is not None and sidecar.is_file():
            with np.load(sidecar, allow_pickle=False) as sz:
                for a, b, p in zip(sz["source_id"].astype(np.int64).tolist(),
                                   sz["target_id"].astype(np.int64).tolist(),
                                   sz["edge_prob"].astype(np.float64).tolist()):
                    if p > 0.5 or not in_scope((a, b)):
                        continue          # band A already covers the deployed band
                    band_b[(a, b)] = p
            # The sidecar is capped at top-k per target, so OUR reproduction may legitimately hold
            # pairs it does not. Containment is the contract; extras are expected, not a failure.
            for k, v in band_b.items():
                if k not in got:
                    b_missing.append(k)
                else:
                    b_checked += 1
                    b_delta = max(b_delta, abs(got[k] - v))

        passed = bool(
            not a_missing and not a_extra and a_delta <= TOL
            and not b_missing and b_delta <= TOL
            and not count_mismatch
            and b_checked > 0
        )
        report["crops"].append({
            "crop": crop, "frames": len(offset), "nodes": int(len(coords)),
            "node_count_mismatches": count_mismatch,
            "band_a": {"recorded": len(band_a), "reproduced": len(got_a),
                       "missing": len(a_missing), "extra": len(a_extra),
                       "max_abs_prob_delta": a_delta},
            "band_b": {"recorded_sub_threshold": len(band_b), "checked": b_checked,
                       "missing": len(b_missing), "max_abs_prob_delta": b_delta},
            "passed": passed,
        })
        print(f"AFP crop={crop} frames={len(offset)} nodes={len(coords)} "
              f"A[rec={len(band_a)} miss={len(a_missing)} extra={len(a_extra)} d={a_delta:.3e}] "
              f"B[rec={len(band_b)} checked={b_checked} miss={len(b_missing)} d={b_delta:.3e}] "
              f"ncount_mismatch={len(count_mismatch)} passed={passed}", flush=True)

    report["all_passed"] = bool(report["crops"]) and all(c["passed"] for c in report["crops"])
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["cache", "verify"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--crops", nargs="+", required=True)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--preilp")
    ap.add_argument("--ecb-dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.phase == "cache":
        return phase_cache(args)
    if not args.preilp or not args.out:
        raise SystemExit("verify needs --preilp and --out")
    return phase_verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
'''

try:
    _afp_worker_path = _AfpPath("/kaggle/working/afp_gate1.py")
    _afp_worker_path.write_text(_AFP_WORKER, encoding="utf-8")

    _afp_preilp = None
    for _cand in (
        _AfpPath("/kaggle/input/biohub-identity-replay-f0/meta/preilp_split0.parquet"),
        _AfpPath("/kaggle/input/biohub-identity-replay-f0/preilp_split0.parquet"),
    ):
        if _cand.is_file():
            _afp_preilp = _cand
            break
    if _afp_preilp is None:
        raise FileNotFoundError("pre-ILP parity target not found in the mounted inputs")

    _afp_ecb = None
    for _cand in (
        _AfpPath("/kaggle/input/biohub-identity-replay-f0/ecb"),
        _AfpPath("/kaggle/input/biohub-identity-replay-f0"),
    ):
        if _cand.is_dir() and any(_cand.glob("*.npz")):
            _afp_ecb = _cand
            break
    if _afp_ecb is None:
        # Band B is not optional: without the sidecars the gate would validate only the band the
        # learnable task does not use, which is exactly how attempt 1 was hollow (FACT-0387).
        raise FileNotFoundError(
            "ECB sidecars not found - band B cannot be checked, and a band-A-only pass is not Gate 1"
        )

    _afp_weights = _AfpPath(WEIGHTS_RELATIVE)
    if not _afp_weights.is_absolute():
        _afp_weights = _AfpPath(REPO_DIR) / WEIGHTS_RELATIVE

    import polars as _afp_pl
    _afp_crops = sorted(
        _afp_pl.read_parquet(_afp_preilp)["dataset"].unique().to_list()
    )[: int(_afp_os.environ.get("BIOHUB_AFP_CROPS", "2"))]
    _afp_maxframes = _afp_os.environ.get("BIOHUB_AFP_MAX_FRAMES", "8")
    _afp_cache_dir = "/kaggle/working/afp_cache"
    _afp_out = "/kaggle/working/assoc_feature_parity.json"

    _afp_common = [
        "--weights", str(_afp_weights),
        "--test-dir", str(TEST_DIR),
        "--cache-dir", _afp_cache_dir,
        "--max-frames", str(_afp_maxframes),
        "--crops", *_afp_crops,
    ]
    # TWO SEPARATE PROCESSES, cwd=REPO_DIR so `import predict_unet_transformer` resolves exactly as
    # it does for the deployed shards. The verifier is a fresh interpreter and cannot see the
    # writer's memory - that is what makes "from the cache alone" enforced rather than intended.
    for _phase, _extra in (
        ("cache", []),
        ("verify", ["--preilp", str(_afp_preilp), "--ecb-dir", str(_afp_ecb), "--out", _afp_out]),
    ):
        _afp_cmd = [_afp_sys.executable, str(_afp_worker_path), "--phase", _phase] + _afp_common + _extra
        print(f"AFP_PHASE {_phase}: {' '.join(_afp_cmd[:6])} ...", flush=True)
        _afp_res = _afp_sub.run(_afp_cmd, cwd=str(REPO_DIR), text=True, capture_output=True)
        print((_afp_res.stdout or "")[-4000:], flush=True)
        if _afp_res.returncode != 0:
            print((_afp_res.stderr or "")[-4000:], flush=True)
            raise RuntimeError(f"gate-1 {_phase} phase exited {_afp_res.returncode}")

    _afp_report = _afp_json.loads(_AfpPath(_afp_out).read_text(encoding="utf-8"))
except Exception as _afp_err:
    import traceback as _afp_tb
    _afp_report["error"] = f"{type(_afp_err).__name__}: {_afp_err}"
    _afp_report["traceback"] = _afp_tb.format_exc()[-2000:]
    _afp_report["all_passed"] = False
    print("AFP_FAILED", _afp_report["error"], flush=True)
    print(_afp_report["traceback"], flush=True)

_AfpPath("/kaggle/working/assoc_feature_parity.json").write_text(
    _afp_json.dumps(_afp_report, indent=2), encoding="utf-8"
)
print("ASSOC_FEATURE_PARITY_COMPLETE all_passed=", _afp_report.get("all_passed"), flush=True)
