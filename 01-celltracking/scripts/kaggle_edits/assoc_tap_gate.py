# =====================================================================================
# GATE 1 REPLAY DRIVER  (PKT-0029, passive-instrumentation redesign)
#
# Runs AFTER the deployed prediction has finished, because the cache it consumes is written BY
# that prediction - scripts/kaggle_edits/assoc_feature_tap.py taps the real `predict_video`.
# This cell does three things and nothing else:
#
#   1. RE-PROVES PASSIVITY FROM DISK. It re-reads the predictor source, removes the AFT-marked
#      spans and requires the sha256 to equal the pre-patch sha256 the tap recorded. Insertion is
#      the only edit the tap makes, so this is a positive proof that the deployed statements the
#      graph came from were not altered - checked after the run rather than asserted before it.
#   2. LAUNCHES THE REPLAY IN A FRESH PROCESS, across the REAL subprocess boundary, with
#      PYTHONPATH = scripts + src. "src" alone is not enough: predict_unet_transformer lives in
#      scripts/, and cwd is NOT on sys.path for a script invocation. That is FACT-0399, which was
#      the live recurrence of FACT-0387 one process over.
#   3. FAILS CLOSED. Any exception, a missing cache, a zero-crop comparison or an empty band
#      leaves all_passed False and prints AFT_GATE_FAILED. A gate that can quietly compare
#      nothing looks exactly like a gate that passed.
#
# It computes no features, opens no image and reproduces no preprocessing. Everything it compares
# was produced by the deployed run itself.
# =====================================================================================
import hashlib as _aftg_hashlib
import json as _aftg_json
import os as _aftg_os
import subprocess as _aftg_sub
import sys as _aftg_sys
from pathlib import Path as _AftgPath

_aftg_report = {"gate": "feature_tap_parity", "attempt": 4, "crops": [], "all_passed": False}

_AFTG_WORKER = r'''#!/usr/bin/env python
"""Gate 1 replay worker: re-run ONLY the deployed association head, from a persisted cache.

WHAT THIS IS. The second half of the passive-instrumentation redesign of PKT-0029 Gate 1. The
first half (scripts/kaggle_edits/assoc_feature_tap.py) injects a tap INSIDE the deployed
`predict_video` and records the exact tensors it hands to `model.predict_edges`, together with
the deployed probabilities computed one line later. This script starts in a FRESH PROCESS, loads
only those files, and re-runs the head.

WHAT IT DELIBERATELY DOES NOT DO. It never opens a zarr, never reads an image, never normalises,
never downsamples, never runs the UNet trunk and never extracts a detection peak. Five of the six
defects this gate has produced were reimplementations of exactly those steps (FACT-0387,
FACT-0397, FACT-0399, FACT-0400, plus the per-frame feature-cache defect recorded at the head of
assoc_feature_tap.py). There is nothing left here to reimplement, so there is nothing left of that
class to get wrong.

WHY A SEPARATE PROCESS. Attempt 1 wrote features into a dict and read them back in the same
process, so it exercised no cache at all. Here the operating system enforces "from the cache
alone": this interpreter cannot see the writer's memory.

WHY IT STILL NEEDS THE DEPLOYED MODULE. Only to rebuild the head: `load_model` reconstructs
UNetNodeTransformer from the same config and weights, and `predict_edges` is the deployed method.
It resolves via PYTHONPATH, which the LAUNCHER must set to scripts + src - cwd is NOT on sys.path
for a script invocation, which is the FACT-0399 defect verbatim.

FAIL CLOSED. A missing cache, an empty band, a node-ordering disagreement, an unreadable file or
any exception leaves all_passed False.

KNOWN BLIND SPOT, STATED RATHER THAN DISCOVERED LATER. The comparison is on probabilities AFTER a
softmax over the SOURCE axis, and softmax is invariant to a constant offset along the axis it
normalises. A cache error that shifts EVERY SOURCE IN A FRAME PAIR BY THE SAME AMOUNT is
mathematically invisible here. Any non-uniform error, including a single node's features being
wrong, is detected. Both halves are asserted in tests/test_assoc_feature_tap.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

BAND_A = "a"
BAND_B = "b"


def _import_deployed():
    """Import the deployed predictor, naming the launcher defect if it is not importable."""
    try:
        import predict_unet_transformer as AFP  # noqa: PLC0415
    except ImportError:
        print(
            "AFT_REPLAY_IMPORT_FAILED predict_unet_transformer is not importable. "
            "sys.path[0] is this SCRIPT's directory, not the working directory, so cwd=REPO_DIR "
            "alone never makes this work. The launcher must pass "
            'env={**os.environ, "PYTHONPATH": "scripts" + os.pathsep + "src"} - "src" alone is '
            "not enough because predict_unet_transformer lives in scripts/ (FACT-0399). "
            f"sys.path[0]={sys.path[0]!r}",
            flush=True,
        )
        raise
    return AFP


def _tensor(arr, device, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)


def _slice(cache, pair, role):
    ptr = int(cache["pair_src_ptr" if role == 0 else "pair_tgt_ptr"][pair])
    n = int(cache["pair_src_n" if role == 0 else "pair_tgt_n"][pair])
    return slice(ptr, ptr + n)


def _check_node_ordering(cache, pair, role, failures):
    """The cached role block must be exactly the deployed frame slice, in the deployed order.

    `coords_so_far` numbers nodes by frame in detection order, and the pre-ILP export and the ECB
    sidecars both address nodes by that number. If the cache reorders within a frame, every id in
    every downstream artifact silently means a different cell.
    """
    sl = _slice(cache, pair, role)
    t = int(cache["pair_t_src" if role == 0 else "pair_t_tgt"][pair])
    frames = cache["frames"].tolist()
    if t not in frames:
        failures.append({"pair": int(pair), "role": int(role), "frame": t,
                         "why": "frame absent from the deployed coord_offset"})
        return
    k = frames.index(t)
    expected = np.arange(int(cache["starts"][k]), int(cache["ends"][k]), dtype=np.int64)
    got = cache["role_gid"][sl]
    if got.shape != expected.shape or not np.array_equal(got, expected):
        failures.append({
            "pair": int(pair), "role": int(role), "frame": t,
            "why": "node ids are not the deployed frame slice in the deployed order",
            "expected_n": int(expected.shape[0]), "cached_n": int(got.shape[0]),
        })


def _pos_features(AFP, cache, pair, role):
    """Positional features: the recorded ones when present, else the DEPLOYED function.

    Re-derivation is not a reimplementation - it calls `extract_pos_features` itself, on the
    window-relative coordinates and window shape the deployed code passed it, both recorded
    verbatim. When both are available they are cross-checked, which costs nothing and catches a
    torn window_shape record.
    """
    sl = _slice(cache, pair, role)
    rel = cache["role_coord_rel"][sl].astype(np.int16)
    shape = tuple(int(v) for v in cache["pair_window_shape"][pair])
    derived = AFP.extract_pos_features(rel, shape)
    if "role_pos" in cache.files:
        recorded = cache["role_pos"][sl]
        return recorded, float(np.abs(recorded - derived).max()) if recorded.size else 0.0
    return derived, 0.0


def replay_crop(AFP, model, cache, device, tol):
    """Re-run the head for every recorded frame pair and compare against the deployed record."""
    dtype = getattr(torch, str(cache["feat_dtype"]), torch.float32)
    activation = str(cache["edge_activation"])
    threshold = float(cache["edge_threshold"])
    n_pairs = int(cache["pair_f_idx"].shape[0])

    integrity: list[dict] = []
    if int(cache["coords"].shape[0]) != int(cache["node_count"]):
        integrity.append({"why": "coords length does not equal the deployed node count",
                          "coords": int(cache["coords"].shape[0]),
                          "node_count": int(cache["node_count"])})
    if not bool(cache["role_mask"].all()):
        integrity.append({"why": "a cached mask slot is False; the deployed site builds "
                                 "torch.ones and never pads",
                          "false_slots": int((~cache["role_mask"]).sum())})

    band = {}
    for name in (BAND_A, BAND_B):
        band[name] = {
            "pair": cache[f"band_{name}_pair"].astype(np.int64),
            "i": cache[f"band_{name}_i"].astype(np.int64),
            "j": cache[f"band_{name}_j"].astype(np.int64),
            "src": cache[f"band_{name}_source_id"].astype(np.int64),
            "tgt": cache[f"band_{name}_target_id"].astype(np.int64),
            "prob": cache[f"band_{name}_prob"].astype(np.float64),
        }
    tally = {
        BAND_A: {"checked": 0, "missing": 0, "extra": 0, "delta": 0.0, "recorded": 0,
                 "reproduced": 0},
        BAND_B: {"checked": 0, "missing": 0, "extra": 0, "delta": 0.0, "recorded": 0,
                 "reproduced": 0},
    }
    id_failures: list[dict] = []
    pos_delta = 0.0
    for pair in range(n_pairs):
        for role in (0, 1):
            _check_node_ordering(cache, pair, role, integrity)
        s, t = _slice(cache, pair, 0), _slice(cache, pair, 1)
        pos_s, ds_ = _pos_features(AFP, cache, pair, 0)
        pos_t, dt_ = _pos_features(AFP, cache, pair, 1)
        pos_delta = max(pos_delta, ds_, dt_)
        n_src = s.stop - s.start
        n_tgt = t.stop - t.start
        if n_src == 0 or n_tgt == 0:
            continue
        with torch.no_grad():
            logits = model.predict_edges(
                _tensor(cache["role_feat"][s], device, dtype).unsqueeze(0),
                _tensor(cache["role_feat"][t], device, dtype).unsqueeze(0),
                _tensor(cache["role_coord_scaled"][s], device).unsqueeze(0),
                _tensor(cache["role_coord_scaled"][t], device).unsqueeze(0),
                _tensor(pos_s, device).unsqueeze(0),
                _tensor(pos_t, device).unsqueeze(0),
                torch.from_numpy(cache["role_mask"][s]).to(device).unsqueeze(0),
                torch.from_numpy(cache["role_mask"][t]).to(device).unsqueeze(0),
            )
        raw = logits[0]
        if activation == "softmax":
            probs = torch.softmax(raw, dim=0).cpu().numpy()
        else:
            probs = torch.sigmoid(raw).cpu().numpy()
        gid_s = cache["role_gid"][s]
        gid_t = cache["role_gid"][t]

        # BAND A: the deployed candidate rule (> the deployed threshold), recorded at the
        # deployed site in the SAME run. Both directions are checked: nothing recorded may be
        # absent from the replay, and nothing the replay puts above the threshold may be
        # unrecorded.
        rep = np.argwhere(probs > threshold)
        rep_keys = {(int(a), int(b)) for a, b in rep}
        sel = band[BAND_A]["pair"] == pair
        rec_i, rec_j = band[BAND_A]["i"][sel], band[BAND_A]["j"][sel]
        rec_keys = {(int(a), int(b)) for a, b in zip(rec_i, rec_j)}
        tally[BAND_A]["recorded"] += len(rec_keys)
        tally[BAND_A]["reproduced"] += len(rep_keys)
        tally[BAND_A]["missing"] += len(rec_keys - rep_keys)
        tally[BAND_A]["extra"] += len(rep_keys - rec_keys)

        for name in (BAND_A, BAND_B):
            rows = band[name]["pair"] == pair
            sel_i, sel_j = band[name]["i"][rows], band[name]["j"][rows]
            sel_p = band[name]["prob"][rows]
            sel_src, sel_tgt = band[name]["src"][rows], band[name]["tgt"][rows]
            if name == BAND_B:
                tally[BAND_B]["recorded"] += int(sel_i.size)
            if sel_i.size == 0:
                continue
            inside = (sel_i < n_src) & (sel_j < n_tgt)
            if not bool(inside.all()):
                tally[name]["missing"] += int((~inside).sum())
                id_failures.append({"pair": int(pair), "band": name,
                                    "why": "a recorded local index is outside the pair's block"})
            si, sj = sel_i[inside], sel_j[inside]
            # The recorded GLOBAL ids must be what the cached node ordering says they are. This
            # ties the band artifacts to the node numbering the pre-ILP export and the ECB
            # sidecars address, so an id in either artifact cannot silently mean a different cell.
            if not (np.array_equal(gid_s[si], sel_src[inside])
                    and np.array_equal(gid_t[sj], sel_tgt[inside])):
                id_failures.append({"pair": int(pair), "band": name,
                                    "why": "recorded global ids disagree with the cached node "
                                           "ordering"})
            tally[name]["checked"] += int(si.size)
            tally[name]["delta"] = max(
                tally[name]["delta"],
                float(np.abs(probs[si, sj] - sel_p[inside]).max()) if si.size else 0.0,
            )

    integrity.extend(id_failures)
    result = {"pairs": n_pairs, "nodes": int(cache["node_count"]),
              "integrity_failures": integrity[:20],
              "integrity_failure_count": len(integrity),
              "pos_feature_max_abs_delta": pos_delta}

    # NON-VACUITY, BOTH BANDS. A crop whose band compared nothing satisfies every condition on
    # that band trivially. FACT-0394 is what that costs: the repository's own green happy-path
    # test ran a crop whose band A compared ZERO pairs and asserted a pass. The floor is PER CROP,
    # so an uncapped run is not immune - one such crop certifies a cache nobody checked.
    # BAND B is not optional either: FACT-0382 measured that ALL 691 fold-0 contested errors have
    # their true parent at or below the deployed threshold, so a band-A-only pass is not Gate 1.
    a, b = tally[BAND_A], tally[BAND_B]
    result.update({
        "band_a": {"recorded": a["recorded"], "reproduced": a["reproduced"],
                   "checked": a["checked"], "missing": a["missing"], "extra": a["extra"],
                   "max_abs_prob_delta": a["delta"]},
        "band_b": {"recorded_sub_threshold": b["recorded"], "checked": b["checked"],
                   "missing": b["missing"], "max_abs_prob_delta": b["delta"]},
        "passed": bool(
            not integrity
            and a["missing"] == 0 and a["extra"] == 0 and a["delta"] <= tol
            and b["missing"] == 0 and b["delta"] <= tol
            and pos_delta <= 1e-6
            and a["checked"] > 0
            and b["checked"] > 0
        ),
    })
    return result


def verify(args) -> int:
    AFP = _import_deployed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, window, downsample = AFP.load_model(Path(args.weights), device)
    cache_dir = Path(args.cache_dir)
    names = args.crops or sorted(p.stem for p in cache_dir.glob("*.npz"))
    report = {
        "gate": "feature_tap_parity", "attempt": 4,
        "device": device.type, "window": int(window),
        "downsample": [int(d) for d in downsample],
        "tolerance": args.tol, "crops": [], "all_passed": False,
    }
    if not names:
        # A gate that can quietly compare nothing looks exactly like a gate that passed
        # (FACT-0387 compared zero crops and had to be diagnosed from a log).
        report["error"] = f"no cache files under {cache_dir} - the tap wrote nothing"
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"AFT_REPLAY_FAILED {report['error']}", flush=True)
        return 1

    for crop in names:
        path = cache_dir / f"{crop}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"cache missing for {crop} - the tap did not write it")
        with np.load(path, allow_pickle=False) as cache:
            if int(cache["window"]) != int(window):
                raise ValueError(
                    f"{crop}: cache window {int(cache['window'])} != model window {int(window)}"
                )
            row = replay_crop(AFP, model, cache, device, args.tol)
        row["crop"] = crop
        report["crops"].append(row)
        print(
            f"AFT_REPLAY crop={crop} pairs={row['pairs']} nodes={row['nodes']} "
            f"A[rec={row['band_a']['recorded']} checked={row['band_a']['checked']} "
            f"miss={row['band_a']['missing']} extra={row['band_a']['extra']} "
            f"d={row['band_a']['max_abs_prob_delta']:.3e}] "
            f"B[rec={row['band_b']['recorded_sub_threshold']} "
            f"checked={row['band_b']['checked']} miss={row['band_b']['missing']} "
            f"d={row['band_b']['max_abs_prob_delta']:.3e}] "
            f"integrity_failures={row['integrity_failure_count']} "
            f"pos_d={row['pos_feature_max_abs_delta']:.3e} passed={row['passed']}",
            flush=True,
        )

    report["all_passed"] = bool(report["crops"]) and all(c["passed"] for c in report["crops"])
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["all_passed"] else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True,
                    help="the SAME weights the deployed run used; config.json sits beside it")
    ap.add_argument("--cache-dir", required=True, help="directory of per-crop tap .npz files")
    ap.add_argument("--out", required=True, help="where the gate report JSON is written")
    ap.add_argument("--crops", nargs="*", default=None)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args(argv)
    return verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
'''

try:
    _aftg_dir = _AftgPath(_aftg_os.environ.get("BIOHUB_AFT_DIR", "").strip()
                          or "/kaggle/working/aft_cache")

    # ---- 1. PASSIVITY, RE-PROVED FROM DISK ------------------------------------------------
    _aftg_receipt_path = _AftgPath("/kaggle/working/aft_patch_receipt.json")
    if not _aftg_receipt_path.is_file():
        raise FileNotFoundError(
            "aft_patch_receipt.json is absent - the tap patch never ran, so this run has no "
            "cache and no passivity claim"
        )
    _aftg_receipt = _aftg_json.loads(_aftg_receipt_path.read_text(encoding="utf-8"))
    _aftg_ps = _AftgPath(_aftg_receipt["predictor"])
    _aftg_now = _aftg_ps.read_text()
    _aftg_strip = globals().get("aft_strip_inserts")
    if _aftg_strip is None:
        raise RuntimeError(
            "aft_strip_inserts is not in scope - assoc_feature_tap.py did not run in this kernel"
        )
    _aftg_stripped = _aftg_strip(_aftg_now)
    _aftg_back = _aftg_hashlib.sha256(_aftg_stripped.encode("utf-8")).hexdigest()
    _aftg_report["passivity"] = {
        "pre_sha256": _aftg_receipt["pre_sha256"],
        "stripped_sha256": _aftg_back,
        "post_sha256_now": _aftg_hashlib.sha256(_aftg_now.encode("utf-8")).hexdigest(),
        "identical": _aftg_back == _aftg_receipt["pre_sha256"],
    }
    if not _aftg_report["passivity"]["identical"]:
        raise RuntimeError(
            "PASSIVITY BROKEN: removing the AFT spans does not restore the pre-patch predictor. "
            "The emitted graph cannot be claimed identical to its control."
        )
    print("AFT_PASSIVITY_VERIFIED stripped_sha256 == pre_sha256 "
          f"{_aftg_receipt['pre_sha256'][:12]}", flush=True)

    # ---- 2. THE REPLAY, IN A FRESH PROCESS ------------------------------------------------
    _aftg_worker_path = _AftgPath("/kaggle/working/assoc_tap_replay.py")
    _aftg_worker_path.write_text(_AFTG_WORKER, encoding="utf-8")

    _aftg_weights = _AftgPath(WEIGHTS_RELATIVE)
    if not _aftg_weights.is_absolute():
        _aftg_weights = _AftgPath(REPO_DIR) / WEIGHTS_RELATIVE
    _aftg_out = "/kaggle/working/assoc_feature_tap_gate.json"
    _aftg_cmd = [
        _aftg_sys.executable, str(_aftg_worker_path),
        "--weights", str(_aftg_weights),
        "--cache-dir", str(_aftg_dir),
        "--out", _aftg_out,
        "--tol", _aftg_os.environ.get("BIOHUB_AFT_TOL", "1e-4"),
    ]
    # PYTHONPATH IS THE WHOLE IMPORT FIX AND IT IS NOT OPTIONAL. The deployed shards launch with
    # env={**os.environ, "PYTHONPATH": "src"}; copying that alone would STILL raise
    # ModuleNotFoundError here, because predict_unet_transformer lives in scripts/, not src/.
    _aftg_env = {**_aftg_os.environ,
                 "PYTHONPATH": "scripts" + _aftg_os.pathsep + "src"}
    print(f"AFT_GATE_LAUNCH {' '.join(_aftg_cmd[:4])} ... cwd={REPO_DIR} "
          f"PYTHONPATH={_aftg_env['PYTHONPATH']}", flush=True)
    _aftg_res = _aftg_sub.run(_aftg_cmd, cwd=str(REPO_DIR), text=True, capture_output=True,
                              env=_aftg_env)
    print((_aftg_res.stdout or "")[-6000:], flush=True)
    if _aftg_res.returncode not in (0, 2):
        print((_aftg_res.stderr or "")[-6000:], flush=True)
        raise RuntimeError(f"gate-1 replay exited {_aftg_res.returncode}")
    if not _AftgPath(_aftg_out).is_file():
        print((_aftg_res.stderr or "")[-6000:], flush=True)
        raise FileNotFoundError("the replay produced no report")
    _aftg_loaded = _aftg_json.loads(_AftgPath(_aftg_out).read_text(encoding="utf-8"))
    _aftg_loaded["passivity"] = _aftg_report["passivity"]
    _aftg_report = _aftg_loaded

    # ---- 3. THE CACHE LEAVES THE SESSION AS ONE ARTIFACT -----------------------------------
    # The export sweep deletes every /kaggle/working entry not in _LOEO_KEEP, directories
    # included, and a bare cache directory is also too many files to fetch by URL on a full
    # fold. One archive keeps the standing rule satisfiable: the session's asset is a reusable
    # feature cache, not just a verdict about one.
    import hashlib as _aftg_h2
    import tarfile as _aftg_tar
    _aftg_arch = _AftgPath("/kaggle/working/aft_cache.tar.gz")
    with _aftg_tar.open(_aftg_arch, "w:gz") as _aftg_tf:
        _aftg_tf.add(str(_aftg_dir), arcname="aft_cache")
    _aftg_report["cache_archive"] = {
        "path": str(_aftg_arch), "bytes": _aftg_arch.stat().st_size,
        "sha256": _aftg_h2.sha256(_aftg_arch.read_bytes()).hexdigest(),
        "crops": len(sorted(_aftg_dir.glob("*.npz"))),
    }
    print(f"AFT_CACHE_ARCHIVED {_aftg_arch} "
          f"bytes={_aftg_report['cache_archive']['bytes']} "
          f"crops={_aftg_report['cache_archive']['crops']} "
          f"sha256={_aftg_report['cache_archive']['sha256'][:12]}", flush=True)
except Exception as _aftg_err:
    import traceback as _aftg_tb
    _aftg_report["error"] = f"{type(_aftg_err).__name__}: {_aftg_err}"
    _aftg_report["traceback"] = _aftg_tb.format_exc()[-2000:]
    _aftg_report["all_passed"] = False
    print("AFT_GATE_FAILED", _aftg_report["error"], flush=True)
    print(_aftg_report["traceback"], flush=True)

_AftgPath("/kaggle/working/assoc_feature_tap_gate.json").write_text(
    _aftg_json.dumps(_aftg_report, indent=2), encoding="utf-8"
)
print("ASSOC_FEATURE_TAP_GATE_COMPLETE all_passed=", _aftg_report.get("all_passed"), flush=True)
