# =====================================================================================
# D1 + D1-F  —  RAW DETECTOR-RESPONSE AUDIT AND FROZEN-FEATURE EXPORT
#
# Injected into predict_unet_transformer.py's per-frame peak-extraction loop, where the FINAL
# post-TTA `det_logits[f_idx]` and the 32-channel `unet_out[:, f_idx]` are both in scope. One
# encoder pass therefore serves the A/B/D audit, the frozen-feature probe and (downstream,
# unchanged) the P3 pregraph. Compact statistics only -- never a heatmap volume.
#
# THE PARTITION IS EXACT. Acceptance in _detect_cells_pooled is
#     is_peak = (logits == pooled) & (sigmoid(logits) > det_threshold)
# two independent conditions, so at any GT voxel:
#     A  logits == pooled  and  sigmoid <= thr   -> genuine local max, THRESHOLD rejected it
#     B  sigmoid > thr     and  logits != pooled -> above threshold, POOL suppressed it
#     D  neither                                 -> no usable response AT THAT VOXEL
#
# D IS STRATIFIED, BECAUSE RAW D MEMBERSHIP PROVES NOTHING. A GT voxel can be D purely from
# coordinate rounding, localisation error or heatmap displacement while a perfectly good
# maximum sits a couple of microns away. So for every GT centre we also locate the strongest
# LOCAL MAXIMUM in a 15 um box and record its distance, logit and its own 32-D feature:
#     D-near   useful local max within  5 um
#     D-mid    useful local max within 10 um
#     D-far    nothing useful within   15 um
# D-near/mid is a localisation or calibration story. Only D-far is even a candidate for a
# representation story -- and D1-F, not D membership, remains the verdict.
#
# `detect_head` is Conv3d(32,1,kernel_size=1), verified at exactly 33 parameters, so a low
# logit is equally consistent with an uninformative feature, a linear map too weak to exploit
# an informative one, or a bad operating point. Exporting features at BOTH the exact GT voxel
# and the nearby maximum lets a CPU probe separate those three.
#
# GT IS AN AUDIT LABEL ONLY. It never enters inference, peak extraction or any proposal, and
# the pipeline below this block is byte-unchanged.
# =====================================================================================
import json as _d1_json
import os as _d1_os
import traceback as _d1_tb
from pathlib import Path as _D1Path

import numpy as _d1_np
import torch as _d1_torch
import torch.nn.functional as _d1_F

_D1_OUT = _D1Path("/kaggle/working/d1_audit")
_D1_OUT.mkdir(parents=True, exist_ok=True)
_D1_BUF: dict = {}          # dataset -> {"rows": [...], "feat_gt": [...], "feat_near": [...]}
_D1_MANIFEST: dict = {}     # dataset -> manifest entry
_D1_GT_CACHE: dict = {}
_D1_RNG = _d1_np.random.default_rng(20260806)

_D1_N_UNIFORM = int(_d1_os.environ.get("BIOHUB_D1_N_UNIFORM", "64"))
_D1_N_SUBTHR = int(_d1_os.environ.get("BIOHUB_D1_N_SUBTHR", "32"))
_D1_RADII_UM = (3.0, 5.0, 7.0, 10.0, 15.0)
_D1_SEARCH_UM = 15.0
_D1_NEAR_UM, _D1_MID_UM = 5.0, 10.0


def _d1_resolve_gt(dataset, gt_dir):
    """Locate the GT geff.

    The LOEO retarget deliberately symlinks ONLY the .zarr images into its working dir --
    "the kernel must be incapable of reading a label" -- so `ds_path.parent` has no .geff and
    the first smoke returned gt=0 rows. That invariant protects the SCORING arm. This kernel
    emits no submission and uses GT strictly as an audit LABEL, never for inference or
    proposal generation, so it resolves the geff from the mounted competition data instead.
    Bounded depth ladder, never a recursive walk: `**` over /kaggle/input would descend the
    79 GB zarr tree.
    """
    import glob as _g
    _direct = _D1Path(str(gt_dir)) / f"{dataset}.geff"
    if _direct.exists():
        return str(_direct)
    for _pat in (f"/kaggle/input/*/train/{dataset}.geff",
                 f"/kaggle/input/*/*/train/{dataset}.geff",
                 f"/kaggle/input/*/*/*/train/{dataset}.geff",
                 f"/kaggle/input/*/{dataset}.geff",
                 f"/kaggle/input/*/*/{dataset}.geff"):
        _hits = _g.glob(_pat)
        if _hits:
            return sorted(_hits)[0]
    return None


def _d1_load_gt(dataset, gt_dir):
    """GT centres per frame in ORIGINAL voxel coords. Time comes from the `t` ATTRIBUTE."""
    if dataset in _D1_GT_CACHE:
        return _D1_GT_CACHE[dataset]
    by_t, err = {}, None
    try:
        import tracksdata as _td
        _p = _d1_resolve_gt(dataset, gt_dir)
        if _p is None:
            raise FileNotFoundError(f"no GT geff for {dataset} under /kaggle/input")
        # Use the SAME loader our exact scorer uses (src/biotrack/metric.py::load_graph).
        # `tracksdata.io.load_geff` does not exist in this build -- v3 returned gt=0 on
        # AttributeError for every crop.
        _g = _td.graph.IndexedRXGraph.from_geff(_D1Path(str(_p)))
        if isinstance(_g, tuple):
            _g = _g[0]
        _na = _g.node_attrs(attr_keys=["t", "z", "y", "x"])
        for _t, _z, _y, _x in zip(_na["t"].to_list(), _na["z"].to_list(),
                                  _na["y"].to_list(), _na["x"].to_list()):
            by_t.setdefault(int(_t), []).append((float(_z), float(_y), float(_x)))
    except Exception as _exc:
        err = f"{type(_exc).__name__}: {_exc}"
        print(f"D1: GT load FAILED for {dataset}: {err}", flush=True)
    _D1_GT_CACHE[dataset] = by_t
    _D1_MANIFEST.setdefault(dataset, {})["gt_load_error"] = err
    return by_t


def _d1_audit_frame(dataset, gt_dir, t, logits_1zyx, feats_czyx, det_threshold,
                    pool_kernel, voxel_size, downsample):
    """One frame. `logits_1zyx` is (1,Z,Y,X) final logits; `feats_czyx` is (C,Z,Y,X)."""
    _b = _D1_BUF.setdefault(dataset, {"rows": [], "feat_gt": [], "feat_near": []})
    _lg = logits_1zyx.detach().float()
    _pooled = _d1_F.max_pool3d(_lg.unsqueeze(0), pool_kernel, stride=1,
                               padding=tuple(k // 2 for k in pool_kernel))[0]
    _prob = _d1_torch.sigmoid(_lg)

    _Z, _Y, _X = _lg.shape[1:]
    # `voxel_size` in predict_video is ALREADY scale*downsample, i.e. the OUTPUT-grid step.
    # Multiplying by downsample again would double-count and inflate every radius 4x in y/x.
    # Measured: (1.625, 0.40625, 0.40625) * (1,4,4) = (1.625, 1.625, 1.625) -- isotropic.
    _step = tuple(float(v) for v in voxel_size)

    _lg_c = _lg[0].cpu().numpy()
    _pl_c = _pooled[0].cpu().numpy()
    _pr_c = _prob[0].cpu().numpy()
    _islm = (_lg_c == _pl_c)
    _acc = _islm & (_pr_c > det_threshold)
    _fe = feats_czyx.detach().float().cpu().numpy()

    def _feat(zz, yy, xx):
        return _fe[:, zz, yy, xx].astype(_d1_np.float32)

    def _box(zz, yy, xx, r_um):
        rz = max(1, int(round(r_um / _step[0]))); ry = max(1, int(round(r_um / _step[1])))
        rx = max(1, int(round(r_um / _step[2])))
        return (slice(max(0, zz - rz), min(_Z, zz + rz + 1)),
                slice(max(0, yy - ry), min(_Y, yy + ry + 1)),
                slice(max(0, xx - rx), min(_X, xx + rx + 1)))

    def _emit(kind, zz, yy, xx, extra=None):
        zz = int(min(max(zz, 0), _Z - 1)); yy = int(min(max(yy, 0), _Y - 1))
        xx = int(min(max(xx, 0), _X - 1))
        _is_lm = bool(_islm[zz, yy, xx]); _ov = bool(_pr_c[zz, yy, xx] > det_threshold)
        _cls = "accepted" if (_is_lm and _ov) else ("A" if _is_lm else ("B" if _ov else "D"))
        _row = {"dataset": dataset, "t": int(t), "kind": kind, "z": zz, "y": yy, "x": xx,
                "logit": float(_lg_c[zz, yy, xx]), "prob": float(_pr_c[zz, yy, xx]),
                "pooled": float(_pl_c[zz, yy, xx]), "is_local_max": _is_lm,
                "over_threshold": _ov, "d1_class": _cls}
        for _r in _D1_RADII_UM:
            _s = _box(zz, yy, xx, _r)
            _row[f"max_logit_{int(_r)}um"] = float(_lg_c[_s].max())
            _row[f"n_accepted_{int(_r)}um"] = int(_acc[_s].sum())

        # --- strongest LOCAL MAXIMUM in the 15 um search box -> D stratification ---
        _s = _box(zz, yy, xx, _D1_SEARCH_UM)
        _sub_lg, _sub_lm = _lg_c[_s], _islm[_s]
        _nz, _ny, _nx, _ndist, _nlogit, _nfrom_lm = zz, yy, xx, float("inf"), float("nan"), False
        if _sub_lm.any():
            _masked = _d1_np.where(_sub_lm, _sub_lg, -_d1_np.inf)
            _fl = int(_d1_np.argmax(_masked)); _nfrom_lm = True
        else:
            _fl = int(_d1_np.argmax(_sub_lg))
        _dz, _dy, _dx = _d1_np.unravel_index(_fl, _sub_lg.shape)
        _nz = int(_s[0].start + _dz); _ny = int(_s[1].start + _dy); _nx = int(_s[2].start + _dx)
        _nlogit = float(_lg_c[_nz, _ny, _nx])
        _ndist = float(_d1_np.sqrt(((_nz - zz) * _step[0]) ** 2 + ((_ny - yy) * _step[1]) ** 2
                                   + ((_nx - xx) * _step[2]) ** 2))
        _row.update({"near_z": _nz, "near_y": _ny, "near_x": _nx,
                     "near_dist_um": _ndist, "near_logit": _nlogit,
                     "near_prob": float(_pr_c[_nz, _ny, _nx]),
                     "near_is_local_max": bool(_nfrom_lm),
                     "near_accepted": bool(_acc[_nz, _ny, _nx])})
        if _cls == "D":
            _row["d_stratum"] = ("D-near" if _ndist <= _D1_NEAR_UM else
                                 ("D-mid" if _ndist <= _D1_MID_UM else "D-far"))
        else:
            _row["d_stratum"] = ""
        if extra:
            _row.update(extra)
        _b["rows"].append(_row)
        _b["feat_gt"].append(_feat(zz, yy, xx))
        _b["feat_near"].append(_feat(_nz, _ny, _nx))

    for (_gz, _gy, _gx) in _d1_load_gt(dataset, gt_dir).get(int(t), []):
        _emit("gt_centre",
              int(round(_gz / downsample[0])), int(round(_gy / downsample[1])),
              int(round(_gx / downsample[2])),
              {"gt_z": _gz, "gt_y": _gy, "gt_x": _gx})

    for _ in range(_D1_N_UNIFORM):
        _emit("uniform", int(_D1_RNG.integers(_Z)), int(_D1_RNG.integers(_Y)),
              int(_D1_RNG.integers(_X)))

    _si = _d1_np.argwhere(_islm & (_pr_c <= det_threshold))
    if len(_si):
        for _i in _D1_RNG.choice(len(_si), size=min(_D1_N_SUBTHR, len(_si)), replace=False):
            _emit("subthr_localmax", *(int(v) for v in _si[_i]))


def _d1_atomic_write(path, write_fn):
    _tmp = _D1Path(str(path) + ".tmp")
    write_fn(_tmp)
    _tmp.replace(path)


def _d1_flush(fold=None, ckpt_hash=None, expected_crops=None):
    """Per-crop atomic artifacts + manifest. Global sentinel only if every crop closed."""
    import polars as _pl
    # Seed an entry for every EXPECTED stem so a crop that was never reached is visible as
    # "not_reached" rather than silently absent. v3 wrote two crops' artifacts but listed only
    # one in the manifest, which made a missing crop look like it had never existed.
    for _want in _d1_json.loads(_d1_os.environ.get("BIOHUB_LOEO_STEMS", "[]")):
        _e = _D1_MANIFEST.setdefault(_want, {})
        _e.setdefault("dataset", _want)
        _e.setdefault("status", "not_reached")
    _ok = []
    for _ds, _b in _D1_BUF.items():
        _entry = _D1_MANIFEST.setdefault(_ds, {})
        _entry.update({"dataset": _ds, "fold": fold, "checkpoint_sha256": ckpt_hash,
                       "status": "incomplete", "exception": None})
        try:
            _rows = _b["rows"]
            _fg = _d1_np.stack(_b["feat_gt"]) if _b["feat_gt"] else _d1_np.zeros((0, 32), "f4")
            _fn = _d1_np.stack(_b["feat_near"]) if _b["feat_near"] else _d1_np.zeros((0, 32), "f4")
            _df = _pl.DataFrame(_rows)
            _d1_atomic_write(_D1_OUT / f"{_ds}__rows.parquet", lambda p: _df.write_parquet(p))
            # np.save APPENDS ".npy" when the path does not already end in it, so writing to
            # "<name>.npy.tmp" silently produced "<name>.npy.tmp.npy" and the rename source
            # never existed. Hand it an open file object, which suppresses that behaviour.
            def _save_npy(_arr):
                def _w(_p):
                    with open(_p, "wb") as _fh:
                        _d1_np.save(_fh, _arr)
                return _w

            _d1_atomic_write(_D1_OUT / f"{_ds}__feat_gt.npy", _save_npy(_fg))
            _d1_atomic_write(_D1_OUT / f"{_ds}__feat_near.npy", _save_npy(_fn))
            _gt = _df.filter(_pl.col("kind") == "gt_centre") if len(_df) else _df
            _cls = dict(zip(*_gt["d1_class"].value_counts().to_dict(as_series=False).values())) \
                if len(_gt) else {}
            _str = dict(zip(*_gt.filter(_pl.col("d1_class") == "D")["d_stratum"]
                            .value_counts().to_dict(as_series=False).values())) \
                if len(_gt) else {}
            _nd = _gt.filter(_pl.col("d1_class") == "D")["near_dist_um"].to_numpy() \
                if len(_gt) else _d1_np.zeros(0)
            _entry.update({
                "n_rows": int(len(_df)), "gt_rows": int(len(_gt)),
                "abd_counts": {str(k): int(v) for k, v in _cls.items()},
                "d_strata": {str(k): int(v) for k, v in _str.items()},
                "feat_rows": int(_fg.shape[0]),
                "feat_dim": int(_fg.shape[1]) if _fg.size else 0,
                "feat_finite": bool(_d1_np.isfinite(_fg).all() and _d1_np.isfinite(_fn).all()),
                "near_dist_um": {
                    "p50": float(_d1_np.percentile(_nd, 50)) if _nd.size else None,
                    "p90": float(_d1_np.percentile(_nd, 90)) if _nd.size else None,
                    "max": float(_nd.max()) if _nd.size else None},
                "status": "complete"})
            _ok.append(_ds)
        except Exception as _exc:
            _entry["exception"] = f"{type(_exc).__name__}: {_exc}"
            _entry["traceback"] = _d1_tb.format_exc()[-1500:]
            print(f"D1: FLUSH FAILED for {_ds}: {_entry['exception']}", flush=True)

    _man = {"crops": _D1_MANIFEST, "n_complete": len(_ok),
            "expected_crops": expected_crops,
            "COMPLETE": bool(expected_crops is not None and len(_ok) == expected_crops
                             and all(v.get("status") == "complete"
                                     for v in _D1_MANIFEST.values()))}
    _d1_atomic_write(_D1_OUT / "d1_manifest.json",
                     lambda p: _D1Path(p).write_text(_d1_json.dumps(_man, indent=2, default=str)))
    print(f"D1: flushed {len(_ok)}/{expected_crops} crops, COMPLETE={_man['COMPLETE']}",
          flush=True)
