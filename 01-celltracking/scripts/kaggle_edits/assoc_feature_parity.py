# =====================================================================================
# GATE 1 OF THE ASSOCIATION HARNESS: FEATURE PARITY  (PKT-0029)
#
# WHY. Every later gate - label contract, listwise evaluation, full-chain - depends on training
# association heads from CACHED frozen-trunk features rather than re-running the 3D UNet for each
# experiment. That shortcut is only legitimate if the cache is faithful to what the deployed head
# actually consumed. If it is not, a new head trains on subtly different inputs and every
# downstream number silently measures the cache instead of the model.
#
# THE TARGET NEEDS NO NEW GROUND TRUTH. The P30 pre-ILP export already records, per crop, every
# candidate edge the deployed pipeline kept: (source_id, target_id, edge_prob) after softmax over
# the SOURCE axis and the 0.5 threshold (FACT-0369), addressed by positional index into
# coords_so_far and verified against detector peak order (FACT-0373 check D). So the gate is exact:
# rebuild features, re-run the DEPLOYED predict_edges from the cache alone, apply the same softmax
# and threshold, and require the candidate set and probabilities to match what was recorded.
#
# WHY THIS RUNS ON GPU RATHER THAN LOCALLY. Two reasons, and the second is the scientific one.
# (1) tracking_cellmot.io.open_dataset calls pin_memory() and refuses to run without an
#     accelerator. That is a TRANSPORT optimisation, not numerical preprocessing, so it is an
#     implementation constraint and NOT evidence the pipeline is intrinsically GPU-only.
# (2) CPU and GPU floating-point paths can differ by more than the 1e-4 probability tolerance, so a
#     CPU smoke would confound a numeric-backend difference with a genuine cache error - precisely
#     the ambiguity this gate exists to remove. Running where the deployed pipeline runs keeps the
#     comparison honest.
#
# PASSIVE. This adds a diagnostic cell and changes no deployed behaviour: it neither writes a
# submission nor alters the graph. It runs on a couple of crops with a frame cap so the gate is
# cheap to prove before two full feature-cache sessions are spent.
# =====================================================================================
_afp_report = {"gate": "feature_parity", "crops": []}
try:
    import numpy as _afp_np
    import polars as _afp_pl
    import torch as _afp_torch
    from pathlib import Path as _AfpPath

    import predict_unet_transformer as _AFP

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

    _afp_dev = _afp_torch.device("cuda" if _afp_torch.cuda.is_available() else "cpu")
    _afp_weights = _AfpPath(WEIGHTS_RELATIVE) if "WEIGHTS_RELATIVE" in dir() else None
    _afp_model, _afp_W, _afp_ds = _AFP.load_model(_afp_primary_weights_path, _afp_dev)
    _afp_cfg = _AFP.PredictConfig(det_threshold=float(os.environ.get("BIOHUB_DET_THRESHOLD", "0.96875")))
    _afp_dsarr = _afp_np.asarray(_afp_ds, dtype=_afp_np.float32)
    _afp_dsarr_t = _afp_torch.tensor(_afp_dsarr, device=_afp_dev)
    _afp_voxel = _afp_np.asarray([1.625, 0.40625, 0.40625], dtype=_afp_np.float32)
    _afp_pool_k = _AFP.pool_kernel_from_um(_afp_cfg.pool_kernel_um, _afp_voxel)

    _afp_pre = _afp_pl.read_parquet(_afp_preilp)
    _afp_crops = [c for c in sorted(_afp_pre["dataset"].unique().to_list())][:int(
        os.environ.get("BIOHUB_AFP_CROPS", "2"))]
    _afp_maxframes = int(os.environ.get("BIOHUB_AFP_MAX_FRAMES", "8"))

    for _afp_crop in _afp_crops:
        _afp_dsobj = _AFP.open_dataset(TEST_DIR / f"{_afp_crop}.zarr")
        _afp_img = _afp_dsobj.image
        _afp_n = min(_afp_maxframes, _afp_img.shape[0])

        _afp_coord_lists, _afp_offset, _afp_feats, _afp_seen = [], {}, {}, set()
        _afp_total = 0
        for _afp_start in range(0, max(_afp_n - 1, 1), max(_afp_W - 1, 1)):
            _afp_fr = [t for t in range(_afp_start, _afp_start + _afp_W) if t < _afp_n]
            if len(_afp_fr) < 2:
                continue
            _afp_vol = _afp_np.stack([_afp_np.asarray(_afp_img[t]) for t in _afp_fr])[None]
            _afp_ten = _afp_torch.from_numpy(_afp_vol.astype(_afp_np.float32)).to(_afp_dev)
            with _afp_torch.no_grad():
                _afp_uo = _afp_model.unet(_afp_ten)
                _afp_dl = _afp_model.detection_head(_afp_uo)
            for _fi, _t in enumerate(_afp_fr):
                if _t in _afp_seen:
                    continue
                _arr = _AFP._detect_cells_pooled(_afp_dl[0][_fi][0], _t, _afp_cfg.det_threshold, _afp_pool_k)
                _afp_offset[_t] = (_afp_total, _afp_total + len(_arr))
                _afp_total += len(_arr)
                _afp_coord_lists.append(_arr)
                _afp_seen.add(_t)
            _afp_cs = _afp_np.concatenate(_afp_coord_lists) if _afp_coord_lists else _afp_np.empty((0, 4), dtype=_afp_np.int16)
            for _fi, _t in enumerate(_afp_fr):
                if _t not in _afp_offset or _t in _afp_feats:
                    continue
                _s, _e = _afp_offset[_t]
                if _e == _s:
                    continue
                _c = _afp_cs[_s:_e]
                _pc = _afp_torch.from_numpy(_c[:, 1:].astype(_afp_np.float32)).unsqueeze(0).to(_afp_dev)
                _pm = _afp_torch.ones(1, len(_c), dtype=_afp_torch.bool, device=_afp_dev)
                with _afp_torch.no_grad():
                    _afp_feats[_t] = _afp_model._index_features(_afp_uo[:, _fi], _pc, _pm)[0].cpu().numpy()
            del _afp_uo, _afp_dl

        _afp_coords = _afp_np.concatenate(_afp_coord_lists) if _afp_coord_lists else _afp_np.empty((0, 4), dtype=_afp_np.int16)
        _afp_shape = (_afp_W,) + tuple(_afp_img.shape[1:])
        _afp_got = {}
        _afp_frames = sorted(_afp_offset)
        for _ts, _tt in zip(_afp_frames[:-1], _afp_frames[1:]):
            if _tt != _ts + 1 or _ts not in _afp_feats or _tt not in _afp_feats:
                continue
            _ss, _se = _afp_offset[_ts]
            _tsq, _te = _afp_offset[_tt]
            if _se == _ss or _te == _tsq:
                continue
            _cs_, _ct_ = _afp_coords[_ss:_se], _afp_coords[_tsq:_te]
            _pcs = _afp_torch.from_numpy(_cs_[:, 1:].astype(_afp_np.float32)).unsqueeze(0).to(_afp_dev)
            _pct = _afp_torch.from_numpy(_ct_[:, 1:].astype(_afp_np.float32)).unsqueeze(0).to(_afp_dev)
            _csr, _ctr = _cs_.copy(), _ct_.copy()
            _csr[:, 0], _ctr[:, 0] = 0, 1
            _pps = _afp_torch.from_numpy(_AFP.extract_pos_features(_csr, _afp_shape)).unsqueeze(0).to(_afp_dev)
            _ppt = _afp_torch.from_numpy(_AFP.extract_pos_features(_ctr, _afp_shape)).unsqueeze(0).to(_afp_dev)
            _fs = _afp_torch.from_numpy(_afp_feats[_ts]).unsqueeze(0).to(_afp_dev)
            _ft = _afp_torch.from_numpy(_afp_feats[_tt]).unsqueeze(0).to(_afp_dev)
            _ms = _afp_torch.ones(1, len(_cs_), dtype=_afp_torch.bool, device=_afp_dev)
            _mt = _afp_torch.ones(1, len(_ct_), dtype=_afp_torch.bool, device=_afp_dev)
            with _afp_torch.no_grad():
                _lg = _afp_model.predict_edges(_fs, _ft, _pcs * _afp_dsarr_t, _pct * _afp_dsarr_t,
                                               _pps, _ppt, _ms, _mt)
            _pr = _afp_torch.softmax(_lg[0], dim=0).cpu().numpy()
            for _i in range(len(_cs_)):
                for _j in range(len(_ct_)):
                    if _pr[_i, _j] > 0.5:
                        _afp_got[(_ss + _i, _tsq + _j)] = float(_pr[_i, _j])

        _afp_ce = _afp_pre.filter((_afp_pl.col("dataset") == _afp_crop) & (_afp_pl.col("row_type") == "edge"))
        _afp_rec = {(int(a), int(b)): float(c) for a, b, c in
                    zip(_afp_ce["source_id"], _afp_ce["target_id"], _afp_ce["edge_prob"])}
        _afp_nt = {}
        for _t, (_s, _e) in _afp_offset.items():
            for _n in range(_s, _e):
                _afp_nt[_n] = _t
        _afp_rec = {k: v for k, v in _afp_rec.items()
                    if _afp_nt.get(k[0]) in _afp_offset and _afp_nt.get(k[1]) in _afp_offset}
        _afp_missing = sorted(set(_afp_rec) - set(_afp_got))
        _afp_extra = sorted(set(_afp_got) - set(_afp_rec))
        _afp_shared = set(_afp_rec) & set(_afp_got)
        _afp_md = max((abs(_afp_got[k] - _afp_rec[k]) for k in _afp_shared), default=0.0)
        _afp_report["crops"].append({
            "crop": _afp_crop, "frames": len(_afp_offset), "nodes": int(len(_afp_coords)),
            "recorded": len(_afp_rec), "reproduced": len(_afp_got),
            "missing": len(_afp_missing), "extra": len(_afp_extra),
            "max_abs_prob_delta": _afp_md,
            "passed": bool(not _afp_missing and not _afp_extra and _afp_md <= 1e-4),
        })
        print(f"AFP crop={_afp_crop} frames={len(_afp_offset)} nodes={len(_afp_coords)} "
              f"recorded={len(_afp_rec)} reproduced={len(_afp_got)} missing={len(_afp_missing)} "
              f"extra={len(_afp_extra)} max_dprob={_afp_md:.3e} "
              f"passed={_afp_report['crops'][-1]['passed']}", flush=True)

    _afp_report["all_passed"] = bool(_afp_report["crops"]) and all(c["passed"] for c in _afp_report["crops"])
except Exception as _afp_err:
    import traceback as _afp_tb
    _afp_report["error"] = f"{type(_afp_err).__name__}: {_afp_err}"
    _afp_report["traceback"] = _afp_tb.format_exc()[-2000:]
    _afp_report["all_passed"] = False
    print("AFP_FAILED", _afp_report["error"], flush=True)
    print(_afp_report["traceback"], flush=True)

import json as _afp_json
_AfpOut = __import__("pathlib").Path("/kaggle/working/assoc_feature_parity.json")
_AfpOut.write_text(_afp_json.dumps(_afp_report, indent=2))
print("ASSOC_FEATURE_PARITY_COMPLETE all_passed=", _afp_report.get("all_passed"), flush=True)
