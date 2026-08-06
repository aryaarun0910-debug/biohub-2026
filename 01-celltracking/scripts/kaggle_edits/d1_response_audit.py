# =====================================================================================
# D1 + D1-F  —  RESPONSE AUDIT AND FROZEN-FEATURE EXPORT  (v5)
#
# Injected into predict_unet_transformer.py's per-frame peak-extraction loop, where the FINAL
# post-TTA det_logits[f_idx][0] and the 32-channel unet_out[:, f_idx] are both in scope.
# Compact statistics only -- never a heatmap volume.
#
# WHAT CHANGED FROM v4, AND WHY.
#
# 1. SPHERE, NOT CUBE. v4 searched an axis-aligned 15 um BOX, which admits 15*sqrt(3) = 25.98 um
#    at the corner; it duly reported near-distance p50 15.4 um and max 24.4 um for a "15 um"
#    search. Every radial comparison here is physical Euclidean and masked to a sphere first.
#
# 2. SUFFICIENT STATISTICS, NOT K-NEAREST. The causal class depends on the STRONGEST response
#    in each radius, not the closest. We export the strongest maximum within <=7 um, the
#    strongest within (7,15], the nearest within <=15, and counts -- so no arbitrary K can
#    influence the primary M/C/T/L/D classification. An optional K-list is ranked by LOGIT
#    after the spherical mask, carrying both rank and distance.
#
# 3. THE EXACT-VOXEL CLASS IS SECONDARY. Detection success is decided by bipartite matching of
#    accepted peaks to GT within the scorer's 7 um radius, so an accepted peak one voxel away
#    matches correctly even when the GT voxel itself is pool-suppressed. v4 proved the point:
#    44b6_0113de3b matched 52/52 GT while the exact-voxel rule called 26 of them "suppressed".
#    Those columns are retained, renamed `voxel_*`, and drive NOTHING.
#
# 4. IMMUTABLE PER-CROP MANIFESTS. v4 wrote one shared manifest per flush and overwrote it, so
#    with several flushes only the last crop survived and the other looked as if it had never
#    run. Each crop now writes its own start/complete/error record; a parent aggregator builds
#    the global manifest afterwards and fails hard.
#
# GT IS AN AUDIT LABEL ONLY. Never used for inference, peak extraction or proposals.
# =====================================================================================
import json as _d1_json
import os as _d1_os
import traceback as _d1_tb
from pathlib import Path as _D1Path

import numpy as _d1_np
import torch as _d1_torch
import torch.nn.functional as _d1_F

_D1_OUT = _D1Path("/kaggle/working/d1_audit")
_D1_MAN = _D1_OUT / "manifests"
_D1_OUT.mkdir(parents=True, exist_ok=True)
_D1_MAN.mkdir(parents=True, exist_ok=True)

_D1_BUF: dict = {}
_D1_GT_CACHE: dict = {}
_D1_STARTED: set = set()
_D1_RNG = _d1_np.random.default_rng(20260806)

_D1_N_UNIFORM = int(_d1_os.environ.get("BIOHUB_D1_N_UNIFORM", "64"))
_D1_N_SUBTHR = int(_d1_os.environ.get("BIOHUB_D1_N_SUBTHR", "32"))

# Columns that MUST survive to the parquet whenever the crop has any GT row. Asserted in
# _d1_flush; see the trap-21 note there for why an assertion rather than trust.
_D1_REQUIRED_COLS = (
    "dataset", "t", "kind", "z", "y", "x", "logit", "prob", "pooled",
    "voxel_is_local_max", "voxel_over_threshold", "voxel_accepted",
    "n_lm_7um", "n_lm_15um", "n_acc_7um", "n_acc_15um",
    "best7_dist_um", "near15_dist_um", "best15_dist_um",
    "gt_z", "gt_y", "gt_x",
)
_D1_KLIST = int(_d1_os.environ.get("BIOHUB_D1_KLIST", "8"))
_D1_MATCH_UM = 7.0      # the scorer's max_distance
_D1_SEARCH_UM = 15.0


def _d1_atomic(path, write_fn):
    _tmp = _D1Path(str(path) + ".partial")
    write_fn(_tmp)
    _tmp.replace(path)


def _d1_write_json(path, obj):
    _d1_atomic(path, lambda p: _D1Path(p).write_text(
        _d1_json.dumps(obj, indent=2, default=str), encoding="utf-8"))


def _d1_start(dataset):
    """One immutable START record per crop, written before its first frame."""
    if dataset in _D1_STARTED:
        return
    _D1_STARTED.add(dataset)
    _d1_write_json(_D1_MAN / f"{dataset}.start.json", {
        "dataset": dataset, "pid": _d1_os.getpid(),
        "fold": _d1_os.environ.get("BIOHUB_D1_FOLD"),
        "checkpoint_sha256": _d1_os.environ.get("BIOHUB_D1_CKPT_SHA"),
    })


def _d1_resolve_gt(dataset, gt_dir):
    """The LOEO retarget symlinks ONLY .zarr -- the scoring arm must be label-blind. This
    kernel emits no submission and uses GT strictly as an audit label, so it resolves the geff
    from the mounted competition data. Bounded ladder; ** would descend the 79 GB zarr tree."""
    import glob as _g
    _direct = _D1Path(str(gt_dir)) / f"{dataset}.geff"
    if _direct.exists():
        return str(_direct)
    for _pat in (f"/kaggle/input/*/train/{dataset}.geff",
                 f"/kaggle/input/*/*/train/{dataset}.geff",
                 f"/kaggle/input/*/*/*/train/{dataset}.geff",
                 f"/kaggle/input/*/{dataset}.geff",
                 f"/kaggle/input/*/*/{dataset}.geff"):
        _h = _g.glob(_pat)
        if _h:
            return sorted(_h)[0]
    return None


def _d1_load_gt(dataset, gt_dir):
    """GT centres per frame, ORIGINAL voxel coords. Time from the `t` ATTRIBUTE, never ids."""
    if dataset in _D1_GT_CACHE:
        return _D1_GT_CACHE[dataset]
    by_t, err = {}, None
    try:
        import tracksdata as _td
        _p = _d1_resolve_gt(dataset, gt_dir)
        if _p is None:
            raise FileNotFoundError(f"no GT geff for {dataset}")
        # SAME call as our exact scorer (src/biotrack/metric.py::load_graph).
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
    _D1_BUF.setdefault(dataset, {}).setdefault("gt_load_error", err)
    return by_t


def _d1_audit_frame(dataset, gt_dir, t, logits_1zyx, feats_czyx, det_threshold,
                    pool_kernel, voxel_size, downsample):
    _d1_start(dataset)
    _b = _D1_BUF.setdefault(dataset, {})
    _b.setdefault("rows", []); _b.setdefault("feat_gt", []); _b.setdefault("feat_max", [])

    _lg = logits_1zyx.detach().float()
    _pooled = _d1_F.max_pool3d(_lg.unsqueeze(0), pool_kernel, stride=1,
                               padding=tuple(k // 2 for k in pool_kernel))[0]
    _prob = _d1_torch.sigmoid(_lg)
    _Z, _Y, _X = _lg.shape[1:]
    # voxel_size in predict_video is ALREADY scale*downsample: the OUTPUT-grid step.
    _step = tuple(float(v) for v in voxel_size)

    _lgc = _lg[0].cpu().numpy()
    _plc = _pooled[0].cpu().numpy()
    _prc = _prob[0].cpu().numpy()
    _islm = (_lgc == _plc)
    _acc = _islm & (_prc > det_threshold)
    _fe = feats_czyx.detach().float().cpu().numpy()

    _rz = max(1, int(_d1_np.ceil(_D1_SEARCH_UM / _step[0])))
    _ry = max(1, int(_d1_np.ceil(_D1_SEARCH_UM / _step[1])))
    _rx = max(1, int(_d1_np.ceil(_D1_SEARCH_UM / _step[2])))

    def _feat(z, y, x):
        return _fe[:, z, y, x].astype(_d1_np.float32)

    def _sphere_maxima(z, y, x):
        """Local maxima within a 15 um SPHERE, as (dist_um, logit, prob, accepted, z,y,x)."""
        z0, z1 = max(0, z - _rz), min(_Z, z + _rz + 1)
        y0, y1 = max(0, y - _ry), min(_Y, y + _ry + 1)
        x0, x1 = max(0, x - _rx), min(_X, x + _rx + 1)
        sub = _islm[z0:z1, y0:y1, x0:x1]
        idx = _d1_np.argwhere(sub)
        if not len(idx):
            return []
        gz = idx[:, 0] + z0; gy = idx[:, 1] + y0; gx = idx[:, 2] + x0
        d = _d1_np.sqrt(((gz - z) * _step[0]) ** 2 + ((gy - y) * _step[1]) ** 2
                        + ((gx - x) * _step[2]) ** 2)
        keep = d <= _D1_SEARCH_UM          # SPHERE mask, inclusive
        if not keep.any():
            return []
        gz, gy, gx, d = gz[keep], gy[keep], gx[keep], d[keep]
        return [(float(d[i]), float(_lgc[gz[i], gy[i], gx[i]]),
                 float(_prc[gz[i], gy[i], gx[i]]), bool(_acc[gz[i], gy[i], gx[i]]),
                 int(gz[i]), int(gy[i]), int(gx[i])) for i in range(len(d))]

    def _emit(kind, z, y, x, extra=None):
        z = int(min(max(z, 0), _Z - 1)); y = int(min(max(y, 0), _Y - 1))
        x = int(min(max(x, 0), _X - 1))
        row = {"dataset": dataset, "t": int(t), "kind": kind, "z": z, "y": y, "x": x,
               "logit": float(_lgc[z, y, x]), "prob": float(_prc[z, y, x]),
               "pooled": float(_plc[z, y, x]),
               # SECONDARY voxel diagnostics only -- these drive nothing.
               "voxel_is_local_max": bool(_islm[z, y, x]),
               "voxel_over_threshold": bool(_prc[z, y, x] > det_threshold),
               "voxel_accepted": bool(_acc[z, y, x])}
        fmax = None
        if kind == "gt_centre":
            ms = _sphere_maxima(z, y, x)
            in7 = [m for m in ms if m[0] <= _D1_MATCH_UM]
            mid = [m for m in ms if _D1_MATCH_UM < m[0] <= _D1_SEARCH_UM]
            row["n_lm_7um"] = len(in7); row["n_lm_15um"] = len(ms)
            row["n_acc_7um"] = sum(1 for m in in7 if m[3])
            row["n_acc_15um"] = sum(1 for m in ms if m[3])
            for tagname, sel in (("best7", max(in7, key=lambda m: m[1]) if in7 else None),
                                 ("best7_15", max(mid, key=lambda m: m[1]) if mid else None),
                                 ("near15", min(ms, key=lambda m: m[0]) if ms else None)):
                row[f"{tagname}_dist_um"] = sel[0] if sel else None
                row[f"{tagname}_logit"] = sel[1] if sel else None
                row[f"{tagname}_prob"] = sel[2] if sel else None
                row[f"{tagname}_accepted"] = sel[3] if sel else None
            best15 = max(ms, key=lambda m: m[1]) if ms else None
            row["best15_dist_um"] = best15[0] if best15 else None
            row["best15_logit"] = best15[1] if best15 else None
            row["best15_accepted"] = best15[3] if best15 else None
            # K-list ranked by LOGIT after the spherical mask, carrying rank AND distance.
            for r, m in enumerate(sorted(ms, key=lambda m: -m[1])[:_D1_KLIST]):
                row[f"k{r}_dist_um"] = m[0]; row[f"k{r}_logit"] = m[1]
                row[f"k{r}_prob"] = m[2]; row[f"k{r}_accepted"] = m[3]; row[f"k{r}_rank"] = r
            fmax = _feat(best15[4], best15[5], best15[6]) if best15 else None
        if extra:
            row.update(extra)
        _b["rows"].append(row)
        _b["feat_gt"].append(_feat(z, y, x))
        _b["feat_max"].append(fmax if fmax is not None else _feat(z, y, x))

    for (_gz, _gy, _gx) in _d1_load_gt(dataset, gt_dir).get(int(t), []):
        _emit("gt_centre", int(round(_gz / downsample[0])), int(round(_gy / downsample[1])),
              int(round(_gx / downsample[2])), {"gt_z": _gz, "gt_y": _gy, "gt_x": _gx})
    for _ in range(_D1_N_UNIFORM):
        _emit("uniform", int(_D1_RNG.integers(_Z)), int(_D1_RNG.integers(_Y)),
              int(_D1_RNG.integers(_X)))
    _si = _d1_np.argwhere(_islm & (_prc <= det_threshold))
    if len(_si):
        for _i in _D1_RNG.choice(len(_si), size=min(_D1_N_SUBTHR, len(_si)), replace=False):
            _emit("subthr_localmax", *(int(v) for v in _si[_i]))


def _d1_flush(fold=None, ckpt_hash=None, expected_crops=None):
    """One IMMUTABLE terminal record per crop. No shared mutable manifest."""
    import polars as _pl
    for _ds, _b in _D1_BUF.items():
        _term = _D1_MAN / f"{_ds}.complete.json"
        _errp = _D1_MAN / f"{_ds}.error.json"
        if _term.exists() or _errp.exists():
            continue                                    # terminal record is write-once
        try:
            _rows = _b.get("rows", [])
            _fg = (_d1_np.stack(_b["feat_gt"]) if _b.get("feat_gt")
                   else _d1_np.zeros((0, 32), "f4"))
            _fm = (_d1_np.stack(_b["feat_max"]) if _b.get("feat_max")
                   else _d1_np.zeros((0, 32), "f4"))
            # TRAP 21. Rows are heterogeneous dicts: gt-only keys are absent from the
            # 96 non-GT rows emitted per frame (64 uniform + 32 subthr). polars infers the
            # schema from the first 100 rows by default, so a crop whose first GT lands at
            # frame >= 2 (96*2 = 192 > 100) SILENTLY LOSES every gt-only column while the
            # terminal record still reports the right gt_rows and status=complete.
            # Measured: 28 of 199 corpus crops (14.1%) -- 44b6 19.7%, 6bba 10.9%. All three
            # smoke crops start at frame 0, so the smoke could never have caught it.
            _df = _pl.DataFrame(_rows, infer_schema_length=None)
            _missing = [c for c in _D1_REQUIRED_COLS if c not in _df.columns]
            if _missing and any(r.get("kind") == "gt_centre" for r in _rows):
                raise RuntimeError(
                    f"D1 column contract violated for {_ds}: missing {_missing}. "
                    "Schema inference dropped gt-only columns (trap 21)."
                )
            _d1_atomic(_D1_OUT / f"{_ds}__rows.parquet", lambda p: _df.write_parquet(p))

            def _save(arr):
                def _w(p):
                    with open(p, "wb") as fh:
                        _d1_np.save(fh, arr)
                return _w
            _d1_atomic(_D1_OUT / f"{_ds}__feat_gt.npy", _save(_fg))
            _d1_atomic(_D1_OUT / f"{_ds}__feat_max.npy", _save(_fm))
            _gt = _df.filter(_pl.col("kind") == "gt_centre") if len(_df) else _df
            _d1_write_json(_term, {
                "dataset": _ds, "status": "complete", "pid": _d1_os.getpid(),
                "fold": fold, "checkpoint_sha256": ckpt_hash,
                "gt_load_error": _b.get("gt_load_error"),
                "n_rows": int(len(_df)), "gt_rows": int(len(_gt)),
                "feat_rows": int(_fg.shape[0]),
                "feat_dim": int(_fg.shape[1]) if _fg.size else 0,
                "feat_finite": bool(_d1_np.isfinite(_fg).all() and _d1_np.isfinite(_fm).all()),
                "exception": None,
            })
        except Exception as _exc:
            _d1_write_json(_errp, {"dataset": _ds, "status": "error", "pid": _d1_os.getpid(),
                                   "fold": fold, "checkpoint_sha256": ckpt_hash,
                                   "exception": f"{type(_exc).__name__}: {_exc}",
                                   "traceback": _d1_tb.format_exc()[-1500:]})
            print(f"D1: FLUSH FAILED for {_ds}: {_exc}", flush=True)
    print(f"D1: wrote terminal records for {sorted(_D1_BUF)}", flush=True)
