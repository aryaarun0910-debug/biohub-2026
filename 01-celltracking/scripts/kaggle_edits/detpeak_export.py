# =====================================================================================
# DETECTION-PEAK SUPERSET EXPORT
#
# Purpose: buy the ENTIRE detection-threshold curve with ONE GPU run.
#
# `_detect_cells_pooled` selects peaks as
#     is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
# (vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py:285).
# The `logits == pooled` local-max test does NOT depend on det_threshold; only the
# sigmoid mask does, and it is monotone in the threshold. So the peak set at T=0.5 is a
# strict SUPERSET of the set at any higher T, and exporting every peak at T=0.5 with its
# logit makes the whole curve [0.5, 1.0] a pure CPU replay.
#
# CRITICAL DESIGN POINT -- the export is DECOUPLED from the pipeline threshold.
# The patch computes the local-max mask once, records everything above BIOHUB_DETPEAK_
# EXPORT_T (0.5) into the export sink, and then returns ONLY the peaks above the
# pipeline's own BIOHUB_DET_THRESHOLD (0.96875). Consequences:
#   * the graph this kernel builds is bit-identical to the paired p3_base LOEO run, so
#     the emitted submission stays directly comparable and the delta is isolated;
#   * runtime and memory are unchanged -- the transformer never sees the extra peaks;
#   * the export is purely observational.
# Setting BIOHUB_DET_THRESHOLD=0.5 instead would have changed the graph AND blown up the
# edge-prediction cost (n_src x n_tgt is quadratic in peaks per frame). Do not do that.
#
# Coordinates are written in the DOWNSAMPLED grid, exactly as `_detect_cells_pooled`
# returns them, i.e. BEFORE `coords[:, 1:] *= ds_arr` at :495. The replay multiplies by
# `downsample = [1, 4, 4]` (:157) itself. This keeps values in int16 range and makes the
# scaling convention explicit rather than baked in.
#
# -------------------------------------------------------------------------------------
# 2026-08-25 -- ROOT-CAUSE REPAIR. The first version of this patch exported NOTHING.
#
# Measured on p4 fold 0 (kernel biohub-p4-detpeak-superset-export-loeo-f0, 9,439 s):
# `detpeaks/` came back an EMPTY directory across all 71 crops, and the log contains
# neither the per-crop success line nor the `no peaks buffered` warning -- i.e. the flush
# never ran at all, so the failure was SILENT IN BOTH BRANCHES.
#
# WHY. The old patch installed the sink and flush on the NOTEBOOK process's `builtins`
# (`_builtins._BIOHUB_PEAK_SINK = ...`). But the notebook does not import the predictor --
# it launches it as a SUBPROCESS, one shard per GPU:
#     subprocess.Popen([sys.executable, "scripts/predict_unet_transformer.py", ...],
#                      cwd=REPO_DIR, env=shard_env)      # base notebook cell 5:388
# A subprocess is a fresh interpreter with its OWN builtins, so inside it both lookups
#     getattr(builtins, "_BIOHUB_PEAK_SINK", None)   -> None
#     getattr(builtins, "_BIOHUB_FLUSH_PEAKS", None) -> None
# returned None. The export block was guarded by `if _sink is not None` and the flush by
# `if _fl is not None`, so BOTH no-oped without raising -- which is exactly why the run
# looked healthy. Rewriting the source file DID cross the process boundary (the child
# reads the patched file); only the in-memory hooks did not.
#
# THE FIX. State that must reach the child travels by ENV VAR or by SOURCE, never by
# process-local memory. `env=shard_env` is derived from `os.environ`, so env vars do
# propagate. This version therefore defines the sink and flush INSIDE the patched source
# file, at module level, gated on `BIOHUB_DETPEAK_ENABLE`. Nothing is read from builtins.
#
# This is the same class of bug as the DataParallel/autocast trap in the ledger: state
# scoped to one execution context silently failing to reach another.
# =====================================================================================
import json
import os
from pathlib import Path

_PEAK_EXPORT_T = float(os.environ.get("BIOHUB_DETPEAK_EXPORT_T", "0.5"))
_PEAK_DIR = Path(os.environ.get("BIOHUB_DETPEAK_DIR", "/kaggle/working/detpeaks"))

# Propagate to the prediction subprocesses. This is the ONLY channel that crosses the
# process boundary; `shard_env` is built from `os.environ`.
os.environ["BIOHUB_DETPEAK_ENABLE"] = "1"
os.environ["BIOHUB_DETPEAK_EXPORT_T"] = repr(_PEAK_EXPORT_T)
os.environ["BIOHUB_DETPEAK_DIR"] = str(_PEAK_DIR)
_PEAK_DIR.mkdir(parents=True, exist_ok=True)

_dp_src = _ps.read_text()

# --- 1. subprocess-local sink + flush, injected into the source ----------------------
# `os`, `Path` and `np` are already imported by the target module (:11, :14, :16).
_dp_pre_old = "def _detect_cells_pooled("
_dp_pre_new = '''_BIOHUB_DETPEAK_ENABLE = os.environ.get("BIOHUB_DETPEAK_ENABLE", "0") != "0"
_BIOHUB_DETPEAK_T = float(os.environ.get("BIOHUB_DETPEAK_EXPORT_T", "0.5"))
_BIOHUB_DETPEAK_DIR = Path(os.environ.get("BIOHUB_DETPEAK_DIR", "/kaggle/working/detpeaks"))
_BIOHUB_PEAK_ROWS: list = []
_BIOHUB_PIPELINE_THRESHOLDS: set[float] = set()
if _BIOHUB_DETPEAK_ENABLE:
    _BIOHUB_DETPEAK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"detpeak: export ACTIVE in pid {os.getpid()} "
          f"(T={_BIOHUB_DETPEAK_T}, dir={_BIOHUB_DETPEAK_DIR})", flush=True)


def _biohub_peak_sink(t, idx, logit, pipeline_threshold) -> None:
    _BIOHUB_PEAK_ROWS.append((int(t), idx, logit))
    _BIOHUB_PIPELINE_THRESHOLDS.add(float(pipeline_threshold))


def _biohub_flush_peaks(crop_name: str) -> None:
    """Write one compressed .npz for `crop_name`, then reset the buffer."""
    if not _BIOHUB_DETPEAK_ENABLE:
        return
    if not _BIOHUB_PEAK_ROWS:
        print(f"  detpeak: no peaks buffered for {crop_name}", flush=True)
        return
    ts = np.concatenate([np.full(len(i), t, dtype=np.int16) for t, i, _ in _BIOHUB_PEAK_ROWS])
    zyx = np.concatenate([i for _, i, _ in _BIOHUB_PEAK_ROWS]).astype(np.int16)
    lg = np.concatenate([g for _, _, g in _BIOHUB_PEAK_ROWS]).astype(np.float32)
    if len(_BIOHUB_PIPELINE_THRESHOLDS) != 1:
        raise RuntimeError(
            f"detpeak: expected one pipeline threshold for {crop_name}, "
            f"got {sorted(_BIOHUB_PIPELINE_THRESHOLDS)}"
        )
    pipeline_threshold = next(iter(_BIOHUB_PIPELINE_THRESHOLDS))
    out = _BIOHUB_DETPEAK_DIR / f"{crop_name}.npz"
    pipeline_peak_count = np.asarray(
        np.count_nonzero(1.0 / (1.0 + np.exp(-lg.astype(np.float64))) > pipeline_threshold),
        dtype=np.int64,
    )
    payload = {
        "t": ts,
        "zyx": zyx,
        "logit": lg,
        "pipeline_threshold": np.asarray(pipeline_threshold, dtype=np.float64),
        "pipeline_peak_count": pipeline_peak_count,
    }
    if out.exists():
        # Some inherited notebooks invoke each deterministic shard twice. Never allow a
        # last-writer-wins sidecar: an identical replay is explicit; any divergent detector
        # stream or nondeterminism crashes the diagnostic.
        with np.load(out, allow_pickle=False) as previous:
            mismatch = [
                key for key, value in payload.items()
                if key not in previous.files or not np.array_equal(previous[key], value)
            ]
        if mismatch:
            raise RuntimeError(
                f"detpeak: divergent duplicate write for {crop_name}: {mismatch}"
            )
        print(f"  detpeak: {crop_name} duplicate-identical replay VERIFIED", flush=True)
    else:
        tmp = out.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, out)
    print(f"  detpeak: {crop_name} -> {len(ts):,} peaks, "
          f"{out.stat().st_size / 1e6:.1f} MB", flush=True)
    _BIOHUB_PEAK_ROWS.clear()
    _BIOHUB_PIPELINE_THRESHOLDS.clear()


def _detect_cells_pooled('''
assert _dp_src.count(_dp_pre_old) == 1, f"detpeak: preamble anchor count {_dp_src.count(_dp_pre_old)}"
_dp_src = _dp_src.replace(_dp_pre_old, _dp_pre_new, 1)

# --- 2. record the superset, return the pipeline subset ------------------------------
_dp_old = """    logits = det_logits.unsqueeze(0)  # (1, 1, Z, Y, X)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])  # (N, 3)
"""
_dp_new = """    logits = det_logits.unsqueeze(0)  # (1, 1, Z, Y, X)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    _local_max = (logits == pooled)
    _sig = torch.sigmoid(logits)

    # --- DETPEAK EXPORT (observational; does NOT affect what is returned) ---
    if _BIOHUB_DETPEAK_ENABLE:
        try:
            _exp_idx = torch.nonzero((_local_max & (_sig > _BIOHUB_DETPEAK_T))[0, 0])
            if _exp_idx.shape[0] > 0:
                _pi = _exp_idx.long()
                _lg = det_logits[0][_pi[:, 0], _pi[:, 1], _pi[:, 2]]
                _biohub_peak_sink(t, _pi.cpu().numpy().astype(np.int16),
                                  _lg.detach().float().cpu().numpy().astype(np.float32),
                                  det_threshold)
            del _exp_idx
        except Exception as _exc:  # never let export break inference
            print("detpeak export warning:", _exc, flush=True)

    is_peak = _local_max & (_sig > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])  # (N, 3)
    del _local_max, _sig
"""
assert _dp_src.count(_dp_old) == 1, f"detpeak: peak anchor count {_dp_src.count(_dp_old)}"
_dp_src = _dp_src.replace(_dp_old, _dp_new, 1)

# --- 3. flush once per crop ----------------------------------------------------------
# `predict()` saves one geff per crop (predict_unet_transformer.py:564). Hooking the save
# gives exactly one flush per crop without touching the loop structure.
_dp_sv_old = "        save_graph(graph, output_dir / f\"{name}.geff\")"
_dp_sv_new = """        save_graph(graph, output_dir / f"{name}.geff")
        _biohub_flush_peaks(name)"""
assert _dp_src.count(_dp_sv_old) == 1, f"detpeak: save anchor count {_dp_src.count(_dp_sv_old)}"
_dp_src = _dp_src.replace(_dp_sv_old, _dp_sv_new, 1)

compile(_dp_src, str(_ps), "exec")
_ps.write_text(_dp_src)
print(f"detpeak export patch applied (export_T={_PEAK_EXPORT_T}, pipeline threshold untouched)")
print(f"detpeak: subprocess env armed -> BIOHUB_DETPEAK_ENABLE=1, dir={_PEAK_DIR}")

_PEAK_MANIFEST = {
    "purpose": "detection-peak superset with logits; enables CPU-only threshold-curve replay",
    "export_threshold": _PEAK_EXPORT_T,
    "pipeline_threshold_env": "BIOHUB_DET_THRESHOLD",
    "coord_space": "DOWNSAMPLED grid; multiply z,y,x by downsample=[1,4,4] for level-0",
    "columns": {"t": "int16", "zyx": "int16 (N,3)", "logit": "float32 raw pre-sigmoid",
                "pipeline_threshold": "float64 scalar", "pipeline_peak_count": "int64 scalar"},
    "graph_unchanged": True,
    "delivery": "sink/flush are module-level in the predictor source and env-gated, because "
                "prediction runs in a SUBPROCESS and process-local builtins do not cross it",
    "collision_policy": "duplicate crop writes must be array-identical; divergence raises instead of overwriting",
    "note": "peaks are local maxima under the 3um pool kernel; the local-max test is "
            "threshold-independent, so this set is a superset of every higher threshold",
}
Path("/kaggle/working/detpeak_manifest.json").write_text(json.dumps(_PEAK_MANIFEST, indent=2))
print(json.dumps(_PEAK_MANIFEST, indent=2))
