# =====================================================================================
# D1 + D1-F  —  RAW DETECTOR-RESPONSE AUDIT AND FROZEN-FEATURE EXPORT
#
# Injected into predict_unet_transformer.py's per-frame peak-extraction loop, where the
# FINAL post-TTA `det_logits[f_idx]` and the 32-channel `unet_out[:, f_idx]` are both in
# scope. Exports compact statistics only -- never a heatmap volume.
#
# WHY BOTH IN ONE PASS. The encoder forward is the expensive part; the A/B/D audit and the
# frozen-feature probe need the same activations, so they share it.
#
# THE CLASSIFICATION IS EXACT, NOT HEURISTIC. Acceptance in _detect_cells_pooled is
#     is_peak = (logits == pooled) & (sigmoid(logits) > det_threshold)
# two independent conditions, so at any GT voxel:
#     A  logits == pooled  and  sigmoid <= thr   -> genuine local max, THRESHOLD rejected it
#     B  sigmoid > thr     and  logits != pooled -> above threshold, POOL suppressed it
#     D  neither                                 -> no usable response
# No prominence modelling is required or used.
#
# D1-F EXISTS BECAUSE CLASS D IS AMBIGUOUS. `detect_head` is Conv3d(32,1,kernel_size=1),
# verified at 33 parameters, so a low logit is equally consistent with (a) an uninformative
# 32-D feature, (b) a linear head too weak to exploit an informative one, or (c) a bad
# operating point. Observing D therefore does NOT authorise encoder retraining. The exported
# features let a CPU linear probe decide which of the three it is.
#
# GT IS USED ONLY AS AN AUDIT LABEL. It never enters inference, peak extraction, or any
# proposal. The pipeline below this block is byte-unchanged.
# =====================================================================================
import json as _d1_json
from pathlib import Path as _D1Path

import numpy as _d1_np
import torch as _d1_torch
import torch.nn.functional as _d1_F

_D1_OUT = _D1Path("/kaggle/working/d1_audit")
_D1_OUT.mkdir(parents=True, exist_ok=True)
_D1_ROWS: list = []
_D1_FEATS: list = []
_D1_GT_CACHE: dict = {}
_D1_RNG = _d1_np.random.default_rng(20260805)

# uniform output-grid samples per frame, and under-threshold local maxima kept per frame
_D1_N_UNIFORM = int(os.environ.get("BIOHUB_D1_N_UNIFORM", "64"))
_D1_N_SUBTHR = int(os.environ.get("BIOHUB_D1_N_SUBTHR", "32"))
_D1_RADII_UM = (3.0, 5.0, 7.0, 10.0, 15.0)


def _d1_load_gt(dataset: str, gt_dir):
    """GT centres per frame, in ORIGINAL voxel coordinates. Audit labels only."""
    if dataset in _D1_GT_CACHE:
        return _D1_GT_CACHE[dataset]
    by_t: dict = {}
    try:
        import geff as _geff  # noqa: F401
        import tracksdata as _td
        _p = _D1Path(str(gt_dir)) / f"{dataset}.geff"
        _g = _td.graph.InMemoryGraph.from_geff(str(_p)) if hasattr(
            _td.graph.InMemoryGraph, "from_geff") else _td.io.load_geff(str(_p))
        if isinstance(_g, tuple):
            _g = _g[0]
        _na = _g.node_attrs(attr_keys=["t", "z", "y", "x"])
        for _t, _z, _y, _x in zip(_na["t"].to_list(), _na["z"].to_list(),
                                  _na["y"].to_list(), _na["x"].to_list()):
            by_t.setdefault(int(_t), []).append((float(_z), float(_y), float(_x)))
    except Exception as _exc:  # never break inference for an audit
        print(f"D1: GT load failed for {dataset}: {type(_exc).__name__}: {_exc}", flush=True)
    _D1_GT_CACHE[dataset] = by_t
    return by_t


def _d1_audit_frame(dataset, gt_dir, t, logits_1zyx, feats_czyx, det_threshold,
                    pool_kernel, voxel_size, downsample):
    """One frame. `logits_1zyx` is (1,Z,Y,X) final logits; `feats_czyx` is (C,Z,Y,X)."""
    _lg = logits_1zyx.detach().float()
    _pooled = _d1_F.max_pool3d(_lg.unsqueeze(0), pool_kernel, stride=1,
                               padding=tuple(k // 2 for k in pool_kernel))[0]
    _prob = _d1_torch.sigmoid(_lg)
    _is_max = (_lg == _pooled)
    _over = (_prob > det_threshold)
    _accepted = _is_max & _over

    _Z, _Y, _X = _lg.shape[1:]
    # `voxel_size` in predict_video is ALREADY scale*downsample, i.e. the OUTPUT-grid step.
    # Multiplying by downsample again would double-count and inflate every radius 4x in y/x.
    # Measured: (1.625, 0.40625, 0.40625) * (1,4,4) = (1.625, 1.625, 1.625) -- isotropic.
    _step = tuple(float(v) for v in voxel_size)

    _lg_c = _lg[0].cpu().numpy()
    _pl_c = _pooled[0].cpu().numpy()
    _pr_c = _prob[0].cpu().numpy()
    _ac_c = _accepted[0].cpu().numpy()
    _fe_c = feats_czyx.detach().float().cpu().numpy()

    def _emit(kind, zz, yy, xx, extra=None):
        zz = int(min(max(zz, 0), _Z - 1)); yy = int(min(max(yy, 0), _Y - 1))
        xx = int(min(max(xx, 0), _X - 1))
        _l = float(_lg_c[zz, yy, xx]); _p = float(_pr_c[zz, yy, xx])
        _is_lm = bool(_lg_c[zz, yy, xx] == _pl_c[zz, yy, xx])
        _ov = bool(_p > det_threshold)
        _cls = "accepted" if (_is_lm and _ov) else ("A" if _is_lm else ("B" if _ov else "D"))
        _row = {"dataset": dataset, "t": int(t), "kind": kind,
                "z": zz, "y": yy, "x": xx,
                "logit": _l, "prob": _p, "pooled": float(_pl_c[zz, yy, xx]),
                "is_local_max": _is_lm, "over_threshold": _ov, "d1_class": _cls}
        if extra:
            _row.update(extra)
        # neighbourhood maxima at physical radii
        for _r in _D1_RADII_UM:
            _rz = max(1, int(round(_r / _step[0]))); _ry = max(1, int(round(_r / _step[1])))
            _rx = max(1, int(round(_r / _step[2])))
            _sub = _lg_c[max(0, zz - _rz):zz + _rz + 1,
                         max(0, yy - _ry):yy + _ry + 1,
                         max(0, xx - _rx):xx + _rx + 1]
            _acc = _ac_c[max(0, zz - _rz):zz + _rz + 1,
                         max(0, yy - _ry):yy + _ry + 1,
                         max(0, xx - _rx):xx + _rx + 1]
            _row[f"max_logit_{int(_r)}um"] = float(_sub.max())
            _row[f"n_accepted_{int(_r)}um"] = int(_acc.sum())
        _D1_ROWS.append(_row)
        _D1_FEATS.append(_fe_c[:, zz, yy, xx].astype(_d1_np.float32))

    # 1) every labelled centre, mapped to the OUTPUT grid
    for (_gz, _gy, _gx) in _d1_load_gt(dataset, gt_dir).get(int(t), []):
        _emit("gt_centre",
              int(round(_gz / downsample[0])),
              int(round(_gy / downsample[1])),
              int(round(_gx / downsample[2])),
              {"gt_z": _gz, "gt_y": _gy, "gt_x": _gx})

    # 2) uniform output-grid positions (the count-prior / base-rate denominator)
    for _ in range(_D1_N_UNIFORM):
        _emit("uniform", int(_D1_RNG.integers(_Z)), int(_D1_RNG.integers(_Y)),
              int(_D1_RNG.integers(_X)))

    # 3) under-threshold local maxima -- the population arm T would admit
    _sub_idx = _d1_np.argwhere((_lg_c == _pl_c) & (_pr_c <= det_threshold))
    if len(_sub_idx):
        _pick = _D1_RNG.choice(len(_sub_idx), size=min(_D1_N_SUBTHR, len(_sub_idx)),
                               replace=False)
        for _i in _pick:
            _z, _y, _x = _sub_idx[_i]
            _emit("subthr_localmax", _z, _y, _x)


def _d1_flush():
    if not _D1_ROWS:
        print("D1: nothing to flush", flush=True)
        return
    import polars as _pl
    _df = _pl.DataFrame(_D1_ROWS)
    _df.write_parquet(_D1_OUT / "d1_rows.parquet")
    _d1_np.save(_D1_OUT / "d1_feats.npy", _d1_np.stack(_D1_FEATS))
    _meta = {"n_rows": len(_D1_ROWS), "feat_dim": int(_D1_FEATS[0].shape[0]),
             "n_uniform_per_frame": _D1_N_UNIFORM, "n_subthr_per_frame": _D1_N_SUBTHR,
             "radii_um": list(_D1_RADII_UM)}
    (_D1_OUT / "d1_meta.json").write_text(_d1_json.dumps(_meta, indent=2))
    print(f"D1: wrote {len(_D1_ROWS)} rows, feats {_d1_np.stack(_D1_FEATS).shape}", flush=True)
