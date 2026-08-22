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
# =====================================================================================
import json
import os
from pathlib import Path

import numpy as np

_PEAK_EXPORT_T = float(os.environ.get("BIOHUB_DETPEAK_EXPORT_T", "0.5"))
_PEAK_ROWS: list = []
_PEAK_DIR = Path("/kaggle/working/detpeaks")
_PEAK_DIR.mkdir(parents=True, exist_ok=True)

_dp_src = _ps.read_text()

# --- 1. record the superset, return the pipeline subset ------------------------------
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
    try:
        import builtins as _bi
        _sink = getattr(_bi, "_BIOHUB_PEAK_SINK", None)
        if _sink is not None:
            _exp_t = getattr(_bi, "_BIOHUB_PEAK_EXPORT_T", 0.5)
            _exp_idx = torch.nonzero((_local_max & (_sig > _exp_t))[0, 0])
            if _exp_idx.shape[0] > 0:
                _pi = _exp_idx.long()
                _lg = det_logits[0][_pi[:, 0], _pi[:, 1], _pi[:, 2]]
                _sink(t, _pi.cpu().numpy().astype(np.int16),
                      _lg.detach().float().cpu().numpy().astype(np.float32))
            del _exp_idx
    except Exception as _exc:  # never let export break inference
        print("detpeak export warning:", _exc)

    is_peak = _local_max & (_sig > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])  # (N, 3)
    del _local_max, _sig
"""
assert _dp_src.count(_dp_old) == 1, f"detpeak: peak anchor count {_dp_src.count(_dp_old)}"
_dp_src = _dp_src.replace(_dp_old, _dp_new, 1)

# --- 2. flush once per crop ----------------------------------------------------------
# `predict()` saves one geff per crop (predict_unet_transformer.py:566). Hooking the save
# gives exactly one flush per crop without touching the loop structure.
_dp_sv_old = "        save_graph(graph, output_dir / f\"{name}.geff\")"
_dp_sv_new = """        save_graph(graph, output_dir / f"{name}.geff")
        import builtins as _bi2
        _fl = getattr(_bi2, "_BIOHUB_FLUSH_PEAKS", None)
        if _fl is not None:
            _fl(name)"""
assert _dp_src.count(_dp_sv_old) == 1, f"detpeak: save anchor count {_dp_src.count(_dp_sv_old)}"
_dp_src = _dp_src.replace(_dp_sv_old, _dp_sv_new, 1)

compile(_dp_src, str(_ps), "exec")
_ps.write_text(_dp_src)
print(f"detpeak export patch applied (export_T={_PEAK_EXPORT_T}, pipeline threshold untouched)")


# --- 3. install sink + flush ---------------------------------------------------------
import builtins as _builtins

_builtins._BIOHUB_PEAK_EXPORT_T = _PEAK_EXPORT_T


def _biohub_peak_sink(t, idx, logit):
    _PEAK_ROWS.append((int(t), idx, logit))


def _biohub_flush_peaks(crop_name: str) -> None:
    """Write one compressed .npz for `crop_name`, then reset the buffer."""
    if not _PEAK_ROWS:
        print(f"  detpeak: no peaks buffered for {crop_name}")
        return
    ts = np.concatenate([np.full(len(i), t, dtype=np.int16) for t, i, _ in _PEAK_ROWS])
    zyx = np.concatenate([i for _, i, _ in _PEAK_ROWS]).astype(np.int16)
    lg = np.concatenate([g for _, _, g in _PEAK_ROWS]).astype(np.float32)
    out = _PEAK_DIR / f"{crop_name}.npz"
    np.savez_compressed(out, t=ts, zyx=zyx, logit=lg)
    print(f"  detpeak: {crop_name} -> {len(ts):,} peaks, {out.stat().st_size / 1e6:.1f} MB")
    _PEAK_ROWS.clear()


_builtins._BIOHUB_PEAK_SINK = _biohub_peak_sink
_builtins._BIOHUB_FLUSH_PEAKS = _biohub_flush_peaks

_PEAK_MANIFEST = {
    "purpose": "detection-peak superset with logits; enables CPU-only threshold-curve replay",
    "export_threshold": _PEAK_EXPORT_T,
    "pipeline_threshold_env": "BIOHUB_DET_THRESHOLD",
    "coord_space": "DOWNSAMPLED grid; multiply z,y,x by downsample=[1,4,4] for level-0",
    "columns": {"t": "int16", "zyx": "int16 (N,3)", "logit": "float32 raw pre-sigmoid"},
    "graph_unchanged": True,
    "note": "peaks are local maxima under the 3um pool kernel; the local-max test is "
            "threshold-independent, so this set is a superset of every higher threshold",
}
Path("/kaggle/working/detpeak_manifest.json").write_text(json.dumps(_PEAK_MANIFEST, indent=2))
print(json.dumps(_PEAK_MANIFEST, indent=2))
