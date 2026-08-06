# =====================================================================================
# D1 + D1-F  —  RESPONSE AUDIT AND FROZEN-FEATURE EXPORT  (v6)
#
# Injected into predict_unet_transformer.py's per-frame peak-extraction loop, where the FINAL
# post-TTA det_logits[f_idx][0], the TTA-MEAN feature accumulator _d1_unet_tta[0, f_idx] and
# the IDENTITY-VIEW unet_out[0, f_idx] are all in scope.
# Compact statistics only -- never a heatmap volume.
#
# THE RULE THAT DEFINES v6. `unet_out` is READ-ONLY. It is read at L442/L445 by
# `_index_features` -> `predict_edges`, AFTER the TTA block; it is the ASSOCIATION
# representation and carries the 0.915 substrate. The TTA-consistent detector feature is
# accumulated into a SEPARATE tensor (`_d1_unet_tta`) that only this audit ever reads.
# Accumulating into `unet_out` -- the fix every research lane independently proposed -- would
# silently change every edge feature in the pipeline.
#
# WHAT CHANGED FROM v5, AND WHY.
#
# 1. TTA-CONSISTENT FEATURES (blocker B3). v5 paired post-TTA det_logits with the
#    IDENTITY-VIEW feature and called the pair a detector representation. It is not: the
#    deployed logit is the mean of 8 aligned views, so checkpoint_detect_head(feat_v5)
#    reproduces the identity-view logit, not the deployed one. v6 exports FOUR arrays and
#    never conflates them:
#       {crop}__feat_tta_mean_gt.npy   32-D at the sampled voxel,  8-view TTA mean
#       {crop}__feat_tta_mean_max.npy  32-D at the strongest nearby local max, TTA mean
#       {crop}__feat_idview_gt.npy     identity view -- the ASSOCIATION representation
#       {crop}__feat_idview_max.npy    identity view at that same maximum
#    The identity-view arrays are retained because `predict_edges` genuinely reads the
#    identity view. They must NEVER be described as the post-TTA detector representation.
#
# 2. THE PARITY ASSERT. `detect_head` is Conv3d(32, 1, kernel_size=1): 33 parameters, a
#    pointwise affine map, so it commutes with both the spatial permutations and the mean.
#    Therefore detect_head(mean_v aligned_feature_v) MUST equal mean_v aligned_logit_v. The
#    audit applies the fold-routed checkpoint head to the TTA-mean feature and HARD ABORTS on
#    disagreement. A near-miss is the dangerous case, not the safe one, so a residual that is
#    systematically signed or correlated with the logit aborts even when its magnitude is
#    below the tolerance. There is no "proceed with caveat" branch.
#
# 3. NaN SENTINEL, NOT A SILENT FALLBACK. v5 wrote `feat_max = feat_gt` whenever no local
#    maximum existed, so non-GT rows silently received "the feature at a random voxel" while
#    GT rows received "the feature at a local max" -- a ~100% label-correlated artefact the
#    moment the two are contrasted. v6 writes an all-NaN row and a `feat_max_valid` boolean.
#    Contract: all-NaN iff invalid, finite iff valid, never +-inf. Asserted at flush.
#
# 4. STABLE row_id. The only alignment key between rows, feature arrays and manifests.
#    0-based, monotonically increasing in emission order; feature-array row i corresponds to
#    row_id == i. Downstream code must NEVER sort rows without applying the identical
#    permutation to the feature arrays.
#
# 5. RANKING DENOMINATORS. n_local_max_in_frame / n_subthr_localmax_in_frame /
#    n_accepted_in_frame are recorded on every row, so a ranking metric no longer has to guess
#    its denominator (the required AUC bar swings ~0.86 -> ~0.99 across its plausible range).
#
# 6. dist_to_nearest_gt_um on every row -- the masked-loss arm cannot be built without it.
#
# INHERITED FROM v5 AND UNCHANGED.
#
#   SPHERE, NOT CUBE. Every radial comparison is physical Euclidean, masked to a sphere first.
#   SUFFICIENT STATISTICS, NOT K-NEAREST -- the causal class depends on the STRONGEST response
#   in each radius, not the closest.
#   THE EXACT-VOXEL CLASS IS SECONDARY (`voxel_*` columns drive NOTHING): detection success is
#   decided by bipartite matching of accepted peaks to GT within the scorer's 7 um radius.
#   IMMUTABLE PER-CROP MANIFESTS: one write-once start/complete/error record per crop.
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

# The deployed view set, in accumulation order. Derived from the GENERATED notebook's TTA
# patch (identity + 3 flips + rot90 k=1,3 + transpose + anti-transpose), NOT from the vendored
# base file -- the base file is not the run.
_D1_TTA_VIEW_SET = (
    "identity", "flip_x", "flip_y", "flip_xy",
    "rot90_k1", "rot90_k3", "transpose_yx", "rot90_k1_then_transpose_yx",
)
# A view count below this is a DEGRADED export, not a fallback. It raises.
_D1_REQUIRE_VIEWS = int(_d1_os.environ.get("BIOHUB_D1_REQUIRE_TTA_VIEWS", "8"))

# Parity gate. float32 rounding on a 32-term dot product is ~1e-6; 1e-4 is two orders of
# headroom. The two structure gates fire on residuals that are small but NOT noise: pure
# rounding noise has |mean|/rms ~ 1/sqrt(N) ~ 0.002 and |pearson| ~ 0.002 at N = 262,144.
_D1_PARITY_MAX_ABS = float(_d1_os.environ.get("BIOHUB_D1_PARITY_MAX_ABS", "1e-4"))
_D1_PARITY_MAX_SIGN_RATIO = 0.5
_D1_PARITY_MAX_CORR = 0.05

# Columns that MUST survive to the parquet whenever the crop has any GT row. Asserted in
# _d1_flush; see the trap-21 note there for why an assertion rather than trust.
_D1_REQUIRED_COLS = (
    "row_id",
    "dataset", "t", "kind", "z", "y", "x", "logit", "prob", "pooled",
    "voxel_is_local_max", "voxel_over_threshold", "voxel_accepted",
    "n_lm_7um", "n_lm_15um", "n_acc_7um", "n_acc_15um",
    "best7_dist_um", "near15_dist_um", "best15_dist_um",
    "gt_z", "gt_y", "gt_x",
    "feat_max_valid", "dist_to_nearest_gt_um",
    "n_local_max_in_frame", "n_subthr_localmax_in_frame", "n_accepted_in_frame",
)
# The four exported feature arrays, and which rows they are defined on.
_D1_FEAT_ARRAYS = (
    "feat_tta_mean_gt", "feat_tta_mean_max", "feat_idview_gt", "feat_idview_max",
)
_D1_FEAT_MAX_ARRAYS = ("feat_tta_mean_max", "feat_idview_max")
_D1_ROW_ID_CONTRACT = (
    "row_id is 0-based and monotonically increasing in emission order; feature-array row i "
    "corresponds to row_id == i. Never sort rows without applying the identical permutation "
    "to the feature arrays."
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
        "split": _d1_os.environ.get("BIOHUB_D1_SPLIT"),
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


def _d1_parity_frame(det_head, feats_tta_czyx, logits_1zyx):
    """detect_head(mean_v aligned_feature_v) MUST equal mean_v aligned_logit_v.

    `detect_head` is Conv3d(C, 1, kernel_size=1) -- a pointwise affine map -- so it commutes
    with the D4 spatial permutations and with the mean. Any disagreement therefore means the
    two tensors are not the pair we believe they are. Returns a record; the caller aborts.
    """
    if det_head is None:
        raise RuntimeError(
            "D1 v6: no detect_head supplied to the audit; the TTA-feature/logit parity "
            "assert is mandatory and must not be skipped"
        )
    _w = next(det_head.parameters())
    _x = feats_tta_czyx.detach().unsqueeze(0)
    _cast = bool(_x.dtype != _w.dtype)
    if _cast:
        _x = _x.to(_w.dtype)
    with _d1_torch.no_grad():
        _recon = det_head(_x)[0]
    if tuple(_recon.shape) != tuple(logits_1zyx.shape):
        raise RuntimeError(
            f"D1 v6 parity: shape mismatch recon {tuple(_recon.shape)} vs "
            f"det_logits {tuple(logits_1zyx.shape)} -- wrong tensor pairing"
        )
    # Statistics in numpy on the host. torch.quantile refuses inputs above ~2**24 elements,
    # which a larger output grid would hit; a silent RuntimeError inside the parity check is
    # the one failure mode that must never happen, because it is the gate for everything else.
    _lg = logits_1zyx.detach().float().reshape(-1).cpu().numpy().astype(_d1_np.float64)
    _rc_v = _recon.detach().float().reshape(-1).cpu().numpy().astype(_d1_np.float64)
    _r = _rc_v - _lg
    _absr = _d1_np.abs(_r)
    _mean = float(_r.mean())
    _rms = float(_d1_np.sqrt((_r ** 2).mean()))
    _lcent = _lg - _lg.mean()
    _rcent = _r - _r.mean()
    _den = float(_d1_np.sqrt((_lcent ** 2).sum())) * float(_d1_np.sqrt((_rcent ** 2).sum()))
    _corr = (float((_lcent * _rcent).sum()) / _den) if _den > 0.0 else 0.0
    return {
        "n_voxels": int(_r.size),
        "max_abs_err": float(_absr.max()),
        "p999_abs_err": float(_d1_np.percentile(_absr, 99.9)),
        "median_abs_err": float(_d1_np.median(_absr)),
        "mean_signed_err": _mean,
        "rms_err": _rms,
        "sign_ratio": (abs(_mean) / _rms) if _rms > 0.0 else 0.0,
        "frac_positive": float((_r > 0).mean()),
        "pearson_r_vs_logit": _corr,
        "logit_abs_max": float(_d1_np.abs(_lg).max()),
        "dtype_feat_tta_mean": str(feats_tta_czyx.dtype),
        "dtype_det_logits": str(logits_1zyx.dtype),
        "dtype_head_weight": str(_w.dtype),
        "dtype_recon": str(_recon.dtype),
        "dtype_cast_applied": _cast,
        "secondary_detection_weight": _d1_os.environ.get(
            "BIOHUB_SECONDARY_DETECTION_WEIGHT", "0"),
    }


def _d1_parity_verdict(rec):
    """Return the list of abort reasons. Empty list == float noise, proceed."""
    _sus = []
    if not (rec["max_abs_err"] <= _D1_PARITY_MAX_ABS):
        _sus.append(
            f"max_abs_err {rec['max_abs_err']:.6e} > {_D1_PARITY_MAX_ABS:.6e}. Suspects, in "
            "order: (a) det_logits was blended after the TTA mean -- "
            f"BIOHUB_SECONDARY_DETECTION_WEIGHT={rec['secondary_detection_weight']!r}, which "
            "must be '0' for parity to be achievable; (b) the feature accumulator missed a "
            "view or used a different divisor than det_logits; (c) a stale or foreign "
            "detect_head (wrong fold checkpoint); (d) an inverse transform that is not the "
            "inverse of its forward view."
        )
    if rec["sign_ratio"] > _D1_PARITY_MAX_SIGN_RATIO:
        _sus.append(
            f"residual is SYSTEMATICALLY SIGNED: |mean|/rms {rec['sign_ratio']:.4f} > "
            f"{_D1_PARITY_MAX_SIGN_RATIO} (float noise gives ~1/sqrt(N) = "
            f"{(1.0 / max(rec['n_voxels'], 1)) ** 0.5:.2e}). A signed offset is a real "
            "mismatch -- a partial accumulation or a dtype cast -- not rounding."
        )
    if abs(rec["pearson_r_vs_logit"]) > _D1_PARITY_MAX_CORR:
        _sus.append(
            f"residual is CORRELATED WITH THE LOGIT: pearson r "
            f"{rec['pearson_r_vs_logit']:+.4f}, |r| > {_D1_PARITY_MAX_CORR}. Rounding noise is "
            "uncorrelated; a correlated residual means the reconstruction is a scaled or "
            "partial version of the deployed logit."
        )
    return _sus


def _d1_audit_frame(dataset, gt_dir, t, logits_1zyx, feats_tta_czyx, feats_idview_czyx,
                    det_threshold, pool_kernel, voxel_size, downsample,
                    det_head=None, tta_view_set=None, n_views=None,
                    n_frames_total=None, window_size=None):
    """Audit one frame.

    `feats_tta_czyx` is the 8-view TTA MEAN (the post-TTA detector representation).
    `feats_idview_czyx` is the identity view -- what `predict_edges` actually reads, i.e. the
    ASSOCIATION representation. They are different tensors and are never interchanged.
    """
    _d1_start(dataset)
    _b = _D1_BUF.setdefault(dataset, {})
    _b.setdefault("rows", [])
    for _k in _D1_FEAT_ARRAYS:
        _b.setdefault(_k, [])

    # ---- view-set contract: a degraded view set is an abort, never a silent fallback -----
    _views = tuple(tta_view_set) if tta_view_set is not None else ()
    _nv = int(n_views) if n_views is not None else -1
    if len(_views) != _nv:
        raise RuntimeError(
            f"D1 v6: tta_view_set has {len(_views)} entries but n_views={_nv}; the deployed "
            "divisor and the recorded view list disagree"
        )
    if _nv != _D1_REQUIRE_VIEWS:
        raise RuntimeError(
            f"D1 v6: {_nv} TTA views accumulated, {_D1_REQUIRE_VIEWS} required. Refusing to "
            "export identity-view features under the post-TTA name -- that is blocker B3. "
            "Set BIOHUB_D1_REQUIRE_TTA_VIEWS deliberately if a different view set is intended."
        )
    # Identity, not just arity. A view set with the right COUNT but the wrong members (an
    # inverse applied in the wrong order, a rot90 k mixed up) still divides by 8 and still
    # produces a plausible-looking mean; only the parity assert would catch it, and only if
    # det_logits happened to be built the same wrong way. Pin the members too.
    if _views != _D1_TTA_VIEW_SET:
        raise RuntimeError(
            f"D1 v6: TTA view set {list(_views)} != the deployed set "
            f"{list(_D1_TTA_VIEW_SET)}; the accumulator is not averaging the views the "
            "generated notebook's patch averages"
        )
    _b["tta_view_set"] = list(_views)
    _b["n_views"] = _nv

    _lg = logits_1zyx.detach().float()
    _pooled = _d1_F.max_pool3d(_lg.unsqueeze(0), pool_kernel, stride=1,
                               padding=tuple(k // 2 for k in pool_kernel))[0]
    _prob = _d1_torch.sigmoid(_lg)
    _Z, _Y, _X = _lg.shape[1:]
    # voxel_size in predict_video is ALREADY scale*downsample: the OUTPUT-grid step.
    _step = tuple(float(v) for v in voxel_size)

    # ---- PARITY: the whole point of v6. Hard abort, no "proceed with caveat" branch. -----
    _par = _d1_parity_frame(det_head, feats_tta_czyx, logits_1zyx)
    _par["dataset"] = dataset
    _par["t"] = int(t)
    _reasons = _d1_parity_verdict(_par)
    _agg = _b.setdefault("parity", {"n_frames": 0, "worst": None, "max_abs_err": 0.0,
                                    "max_p999_abs_err": 0.0, "max_sign_ratio": 0.0,
                                    "max_abs_corr": 0.0, "dtypes": None})
    _agg["n_frames"] += 1
    _agg["dtypes"] = {k: v for k, v in _par.items() if k.startswith("dtype")}
    if _par["max_abs_err"] >= _agg["max_abs_err"]:
        _agg["max_abs_err"] = _par["max_abs_err"]
        _agg["worst"] = _par
    _agg["max_p999_abs_err"] = max(_agg["max_p999_abs_err"], _par["p999_abs_err"])
    _agg["max_sign_ratio"] = max(_agg["max_sign_ratio"], _par["sign_ratio"])
    _agg["max_abs_corr"] = max(_agg["max_abs_corr"], abs(_par["pearson_r_vs_logit"]))
    if _reasons:
        raise RuntimeError(
            f"D1 v6 PARITY ABORT for {dataset} t={t}: " + " | ".join(_reasons)
            + f" | record={_d1_json.dumps(_par, default=str)}"
        )

    _lgc = _lg[0].cpu().numpy()
    _plc = _pooled[0].cpu().numpy()
    _prc = _prob[0].cpu().numpy()
    _islm = (_lgc == _plc)
    _acc = _islm & (_prc > det_threshold)
    _fe_tta = feats_tta_czyx.detach().float().cpu().numpy()
    _fe_idv = feats_idview_czyx.detach().float().cpu().numpy()
    if _fe_tta.shape != _fe_idv.shape:
        raise RuntimeError(
            f"D1 v6: TTA-mean feature {_fe_tta.shape} and identity-view feature "
            f"{_fe_idv.shape} disagree in shape"
        )
    _C = int(_fe_tta.shape[0])
    _nan_feat = _d1_np.full((_C,), _d1_np.nan, dtype=_d1_np.float32)

    # ---- ranking denominators, recorded on every row of this frame ----------------------
    _n_lm_frame = int(_islm.sum())
    _n_acc_frame = int(_acc.sum())
    _sub_mask = _islm & (_prc <= det_threshold)
    _n_sub_frame = int(_sub_mask.sum())

    # ---- memory: the accumulator is one extra (1, W, C, Z, Y, X) float32 tensor ---------
    _slice_bytes = int(feats_tta_czyx.element_size() * feats_tta_czyx.nelement())
    _mem = _b.setdefault("memory", {})
    _mem["feat_slice_bytes"] = _slice_bytes
    _mem["window_size"] = int(window_size) if window_size is not None else None
    _mem["tta_accumulator_bytes"] = (
        _slice_bytes * int(window_size) if window_size is not None else None)
    _mem["feat_dtype"] = str(feats_tta_czyx.dtype)
    try:
        if _d1_torch.cuda.is_available():
            _mem["cuda_max_memory_allocated_bytes"] = int(
                _d1_torch.cuda.max_memory_allocated())
            _mem["cuda_max_memory_reserved_bytes"] = int(
                _d1_torch.cuda.max_memory_reserved())
    except Exception as _exc:                                   # never fail the run on telemetry
        _mem["cuda_query_error"] = f"{type(_exc).__name__}: {_exc}"

    _rz = max(1, int(_d1_np.ceil(_D1_SEARCH_UM / _step[0])))
    _ry = max(1, int(_d1_np.ceil(_D1_SEARCH_UM / _step[1])))
    _rx = max(1, int(_d1_np.ceil(_D1_SEARCH_UM / _step[2])))

    _gt_orig = _d1_load_gt(dataset, gt_dir).get(int(t), [])
    # GT centres on the OUTPUT grid, as floats -- distances stay physical via _step.
    _gt_grid = _d1_np.array(
        [[gz / downsample[0], gy / downsample[1], gx / downsample[2]]
         for (gz, gy, gx) in _gt_orig], dtype=_d1_np.float64
    ).reshape(-1, 3)

    def _feat(arr, z, y, x):
        return arr[:, z, y, x].astype(_d1_np.float32)

    def _dist_to_nearest_gt_um(z, y, x):
        """Physical Euclidean distance to the nearest GT centre IN THIS FRAME. NaN if the
        frame has no GT -- never 0, never a sentinel that could be read as 'adjacent'."""
        if not len(_gt_grid):
            return float("nan")
        _d = _d1_np.sqrt(
            ((_gt_grid[:, 0] - z) * _step[0]) ** 2
            + ((_gt_grid[:, 1] - y) * _step[1]) ** 2
            + ((_gt_grid[:, 2] - x) * _step[2]) ** 2
        )
        return float(_d.min())

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
        # row_id is assigned from the CURRENT length of the row buffer, so it is 0-based and
        # monotonic in emission order across the whole crop, and the feature lists -- appended
        # exactly once per row below -- are indexed by the same integer.
        row = {"row_id": len(_b["rows"]),
               "dataset": dataset, "t": int(t), "kind": kind, "z": z, "y": y, "x": x,
               "logit": float(_lgc[z, y, x]), "prob": float(_prc[z, y, x]),
               "pooled": float(_plc[z, y, x]),
               # SECONDARY voxel diagnostics only -- these drive nothing.
               "voxel_is_local_max": bool(_islm[z, y, x]),
               "voxel_over_threshold": bool(_prc[z, y, x] > det_threshold),
               "voxel_accepted": bool(_acc[z, y, x]),
               # ranking denominators for this frame
               "n_local_max_in_frame": _n_lm_frame,
               "n_subthr_localmax_in_frame": _n_sub_frame,
               "n_accepted_in_frame": _n_acc_frame,
               "dist_to_nearest_gt_um": _dist_to_nearest_gt_um(z, y, x)}
        best15 = None
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
        # NaN SENTINEL. A row with no local maximum in its 15 um sphere -- which is EVERY
        # non-GT row, because the search is only run for gt_centre -- gets an all-NaN feature
        # and feat_max_valid=False. v5 substituted feat_gt here, which made "feature at a
        # local max" perfectly predict "is a GT row".
        row["feat_max_valid"] = bool(best15 is not None)
        if extra:
            row.update(extra)
        _b["rows"].append(row)
        _b["feat_tta_mean_gt"].append(_feat(_fe_tta, z, y, x))
        _b["feat_idview_gt"].append(_feat(_fe_idv, z, y, x))
        if best15 is None:
            _b["feat_tta_mean_max"].append(_nan_feat.copy())
            _b["feat_idview_max"].append(_nan_feat.copy())
        else:
            _b["feat_tta_mean_max"].append(_feat(_fe_tta, best15[4], best15[5], best15[6]))
            _b["feat_idview_max"].append(_feat(_fe_idv, best15[4], best15[5], best15[6]))

    for (_gz, _gy, _gx) in _gt_orig:
        _emit("gt_centre", int(round(_gz / downsample[0])), int(round(_gy / downsample[1])),
              int(round(_gx / downsample[2])), {"gt_z": _gz, "gt_y": _gy, "gt_x": _gx})
    for _ in range(_D1_N_UNIFORM):
        _emit("uniform", int(_D1_RNG.integers(_Z)), int(_D1_RNG.integers(_Y)),
              int(_D1_RNG.integers(_X)))
    _si = _d1_np.argwhere(_sub_mask)
    if len(_si):
        for _i in _D1_RNG.choice(len(_si), size=min(_D1_N_SUBTHR, len(_si)), replace=False):
            _emit("subthr_localmax", *(int(v) for v in _si[_i]))

    # ---- per-crop shape / population bookkeeping for the manifest -----------------------
    _b["grid_zyx"] = [int(_Z), int(_Y), int(_X)]
    _b["feat_dim"] = _C
    _b["n_frames"] = int(_b.get("n_frames", 0)) + 1
    _b["n_frames_total"] = int(n_frames_total) if n_frames_total is not None else None
    # Each frame is audited exactly once (the caller guards on `seen_frames`), so summing the
    # accepted peaks over audited frames IS the estimated node population for this crop.
    _b["estimated_number_of_nodes"] = int(
        _b.get("estimated_number_of_nodes", 0)) + _n_acc_frame
    _b["n_local_max_total"] = int(_b.get("n_local_max_total", 0)) + _n_lm_frame
    _b["n_subthr_localmax_total"] = int(_b.get("n_subthr_localmax_total", 0)) + _n_sub_frame


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
            _dim = int(_b.get("feat_dim", 32))
            _arrs = {}
            for _name in _D1_FEAT_ARRAYS:
                _lst = _b.get(_name) or []
                _arrs[_name] = (_d1_np.stack(_lst) if _lst
                                else _d1_np.zeros((0, _dim), "f4"))
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
            # ---- row_id contract: the ONLY alignment key ------------------------------
            _ids = _df["row_id"].to_list() if len(_df) else []
            if _ids != list(range(len(_rows))):
                raise RuntimeError(
                    f"D1 row_id contract violated for {_ds}: row_id is not 0..n-1 in "
                    "emission order, so no positional join to the feature arrays is safe"
                )
            for _name, _a in _arrs.items():
                if _a.shape[0] != len(_rows):
                    raise RuntimeError(
                        f"D1 alignment violated for {_ds}: {_name} has {_a.shape[0]} rows, "
                        f"rows table has {len(_rows)}"
                    )
            # ---- NaN sentinel contract: all-NaN iff invalid, finite iff valid, no +-inf --
            _valid = _d1_np.array(
                [bool(r.get("feat_max_valid")) for r in _rows], dtype=bool)
            for _name in _D1_FEAT_MAX_ARRAYS:
                _a = _arrs[_name]
                if _a.size:
                    if _d1_np.isinf(_a).any():
                        raise RuntimeError(f"D1 {_name} for {_ds} contains +-inf")
                    if not bool(_d1_np.isfinite(_a[_valid]).all()):
                        raise RuntimeError(
                            f"D1 {_name} for {_ds}: a feat_max_valid row is not finite")
                    if not bool(_d1_np.isnan(_a[~_valid]).all()):
                        raise RuntimeError(
                            f"D1 {_name} for {_ds}: an invalid row is not all-NaN")
            for _name in ("feat_tta_mean_gt", "feat_idview_gt"):
                _a = _arrs[_name]
                if _a.size and not bool(_d1_np.isfinite(_a).all()):
                    raise RuntimeError(
                        f"D1 {_name} for {_ds} is not finite; the sampled-voxel feature is "
                        "defined on every row and must never be NaN"
                    )
            # ---- the TTA-mean and identity-view features must actually DIFFER ----------
            # If they are equal the accumulator collapsed to the identity view, which is
            # exactly blocker B3 wearing the v6 name.
            _identical = bool(
                _arrs["feat_tta_mean_gt"].size
                and _d1_np.array_equal(_arrs["feat_tta_mean_gt"], _arrs["feat_idview_gt"])
            )
            if _identical:
                raise RuntimeError(
                    f"D1 v6 for {_ds}: the TTA-mean feature is bit-identical to the identity "
                    "view. The accumulator did not accumulate; exporting it under the "
                    "post-TTA name would reintroduce blocker B3."
                )

            def _save(arr):
                def _w(p):
                    with open(p, "wb") as fh:
                        _d1_np.save(fh, arr)
                return _w
            _d1_atomic(_D1_OUT / f"{_ds}__rows.parquet", lambda p: _df.write_parquet(p))
            for _name, _a in _arrs.items():
                _d1_atomic(_D1_OUT / f"{_ds}__{_name}.npy", _save(_a))
            _gt = _df.filter(_pl.col("kind") == "gt_centre") if len(_df) else _df
            _par = _b.get("parity") or {}
            _d1_write_json(_term, {
                "dataset": _ds, "status": "complete", "pid": _d1_os.getpid(),
                "schema_version": "d1_v6",
                "fold": fold, "checkpoint_sha256": ckpt_hash,
                "split": _d1_os.environ.get("BIOHUB_D1_SPLIT"),
                "gt_load_error": _b.get("gt_load_error"),
                "n_rows": int(len(_df)), "gt_rows": int(len(_gt)),
                "row_id_contract": _D1_ROW_ID_CONTRACT,
                "feat_arrays": list(_D1_FEAT_ARRAYS),
                "feat_rows": int(_arrs["feat_tta_mean_gt"].shape[0]),
                "feat_dim": int(_arrs["feat_tta_mean_gt"].shape[1])
                if _arrs["feat_tta_mean_gt"].size else 0,
                # feat_finite keeps its v5 meaning for the parent aggregator: reaching this
                # line means the always-defined arrays are finite AND the *_max arrays satisfy
                # the NaN sentinel contract asserted above. Any violation raised already.
                "feat_finite": True,
                "feat_max_valid_rows": int(_valid.sum()),
                "feat_max_invalid_rows": int((~_valid).sum()),
                "tta_view_set": _b.get("tta_view_set"),
                "n_views": _b.get("n_views"),
                "grid_zyx": _b.get("grid_zyx"),
                "n_frames": _b.get("n_frames"),
                "n_frames_total": _b.get("n_frames_total"),
                "n_uniform_per_frame": _D1_N_UNIFORM,
                "n_subthr_per_frame": _D1_N_SUBTHR,
                "estimated_number_of_nodes": _b.get("estimated_number_of_nodes"),
                "n_local_max_total": _b.get("n_local_max_total"),
                "n_subthr_localmax_total": _b.get("n_subthr_localmax_total"),
                "parity": {
                    "gate_max_abs_err": _D1_PARITY_MAX_ABS,
                    "gate_max_sign_ratio": _D1_PARITY_MAX_SIGN_RATIO,
                    "gate_max_abs_corr": _D1_PARITY_MAX_CORR,
                    "n_frames_checked": _par.get("n_frames"),
                    "max_abs_err": _par.get("max_abs_err"),
                    "max_p999_abs_err": _par.get("max_p999_abs_err"),
                    "max_sign_ratio": _par.get("max_sign_ratio"),
                    "max_abs_corr": _par.get("max_abs_corr"),
                    "dtypes": _par.get("dtypes"),
                    "worst_frame": _par.get("worst"),
                },
                "memory": _b.get("memory"),
                "exception": None,
            })
        except Exception as _exc:
            _d1_write_json(_errp, {"dataset": _ds, "status": "error", "pid": _d1_os.getpid(),
                                   "fold": fold, "checkpoint_sha256": ckpt_hash,
                                   "exception": f"{type(_exc).__name__}: {_exc}",
                                   "traceback": _d1_tb.format_exc()[-1500:]})
            print(f"D1: FLUSH FAILED for {_ds}: {_exc}", flush=True)
    print(f"D1: wrote terminal records for {sorted(_D1_BUF)}", flush=True)
