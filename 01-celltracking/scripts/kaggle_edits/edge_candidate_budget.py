# =====================================================================================
# EDGE-CANDIDATE BUDGET  (LEVER-0037 acquisition + treatment surface)
#
# WHY. FACT-0369, verified at source and confirmed across all 2,162,040 fold-1 candidate
# edges: the deployed rule takes softmax over the SOURCE axis and keeps pairs above
# cfg.threshold = 0.5. A softmax over sources sums to one, so AT MOST ONE parent per target
# can survive - in-degree maximum is exactly 1, with zero exceptions, and 14.3% of nodes get
# no candidate parent at all. The edge model is deployed as a hard argmax with abstention,
# not as a ranker. FACT-0370 prices that: 15.6% of fold-1 GT edges have both endpoints
# detected yet no candidate edge ever offered, at a median separation of only 2.26 um, so a
# wider radius is not the fix - the one-parent threshold is.
#
# WHAT THIS PATCH DOES, AND WHAT IT REFUSES TO DO. It is PASSIVE BY DEFAULT. With no
# environment variables set it reproduces the deployed candidate list exactly, so the emitted
# graph and the submission stay bit-for-bit identical. It adds two SEPARABLE things:
#
#   1. AN EXPORT SURFACE (acquisition). Writes a per-crop sidecar of the richer candidate set
#      - every pair above a low floor, capped top-k per target - WITHOUT feeding it to the
#      graph. Nodes, edges and every downstream stage are untouched. This is what lets one
#      passive GPU run serve LEVER-0037 while remaining a valid P28 control.
#   2. A TREATMENT (deployment). `BIOHUB_EDGE_CANDIDATE_THRESHOLD` and
#      `BIOHUB_EDGE_CANDIDATE_TOPK` change the candidate list the pipeline actually consumes.
#
# THESE ARE DELIBERATELY NOT THE SAME SWITCH. The design is one acquisition run and two clean
# falsifiable experiments replayed on CPU, rather than one GPU run whose graph changes
# confound LEVER-0036 with LEVER-0037. Setting a treatment variable on an acquisition run
# would destroy the control, so the heartbeat records which mode was active and an audit can
# tell them apart after the fact.
#
# SIDECAR IDENTIFIERS. `idx_src`/`idx_tgt` are global indices into `coords_so_far`, which is
# exactly the node numbering the pre-ILP export uses, so a sidecar joins to that graph
# directly with no remapping.
#
# FAIL CLOSED. The primary anchor must match exactly once. If the predictor source ever
# changes shape this raises rather than silently exporting nothing - a silent no-op here
# would look exactly like "the richer candidates do not exist", which is the wrong conclusion
# to reach for free.
# =====================================================================================
import os as _ecb_patch_os

_ecb_src = _ps.read_text()

_ecb_config = '''
# --- LEVER-0037 edge-candidate budget (passive unless configured) --------------------
import os as _ecb_os

def _ecb_float(name):
    raw = _ecb_os.environ.get(name, "").strip()
    if not raw:
        return None
    value = float(raw)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value}")
    return value

def _ecb_int(name):
    raw = _ecb_os.environ.get(name, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value

# Treatment: changes the graph the pipeline consumes. Left unset on an acquisition run.
_ECB_THRESHOLD = _ecb_float("BIOHUB_EDGE_CANDIDATE_THRESHOLD")
_ECB_TOPK = _ecb_int("BIOHUB_EDGE_CANDIDATE_TOPK")
# Acquisition: passive sidecar only, never fed back into the graph.
_ECB_EXPORT_THRESHOLD = _ecb_float("BIOHUB_EDGE_CANDIDATE_EXPORT_THRESHOLD")
_ECB_EXPORT_TOPK = _ecb_int("BIOHUB_EDGE_CANDIDATE_EXPORT_TOPK") or 8
_ECB_EXPORT_DIR = _ecb_os.environ.get("BIOHUB_EDGE_CANDIDATE_EXPORT_DIR", "").strip() or None
_ECB_EXPORT_ON = _ECB_EXPORT_DIR is not None and _ECB_EXPORT_THRESHOLD is not None

_ECB_BUFFER = []                                  # per-crop, cleared at each flush
_ECB_CROP = {"pairs": 0, "frame_pairs": 0}        # per-crop, reset at each flush
_ECB_TOTAL = {"pairs": 0, "exported": 0, "crops": 0}


def _ecb_select(probs, n_src, n_tgt, threshold, topk):
    """Candidate (prob, i, j) triples above `threshold`, at most `topk` per TARGET.

    With `topk` None this is exactly the deployed comprehension, including its descending
    sort, so the default path is unchanged. The per-target cap is applied on the descending
    order, keeping the best-scoring parents for each target.
    """
    picked = [
        (probs[i, j], i, j)
        for i in range(n_src)
        for j in range(n_tgt)
        if probs[i, j] > threshold
    ]
    picked.sort(reverse=True)
    if topk is None:
        return picked
    per_target = {}
    capped = []
    for prob, i, j in picked:
        seen = per_target.get(j, 0)
        if seen >= topk:
            continue
        per_target[j] = seen + 1
        capped.append((prob, i, j))
    return capped
# --- end LEVER-0037 edge-candidate budget -------------------------------------------
'''

# NOTE the anchor includes the decorator. `predict_video` is wrapped in @torch.no_grad(), so
# inserting between the decorator and the `def` would apply that decorator to the config
# block's first statement and raise a SyntaxError at import.
_ecb_anchor_cfg = "@torch.no_grad()\ndef predict_video("
assert _ecb_src.count(_ecb_anchor_cfg) == 1, (
    f"edge-candidate budget: predict_video anchor count {_ecb_src.count(_ecb_anchor_cfg)}"
)
_ecb_src = _ecb_src.replace(_ecb_anchor_cfg, _ecb_config + "\n" + _ecb_anchor_cfg, 1)

# The PRIMARY candidate construction. Fail closed if it is not exactly where we expect.
_ecb_old = """            candidates = sorted(
                [
                    (probs[i, j], i, j)
                    for i in range(n_src)
                    for j in range(n_tgt)
                    if probs[i, j] > cfg.threshold
                ],
                reverse=True,
            )"""
_ecb_new = """            _ecb_threshold = cfg.threshold if _ECB_THRESHOLD is None else _ECB_THRESHOLD
            candidates = _ecb_select(probs, n_src, n_tgt, _ecb_threshold, _ECB_TOPK)
            _ECB_CROP["pairs"] += len(candidates)
            _ECB_CROP["frame_pairs"] += 1
            if _ECB_EXPORT_ON:
                # PASSIVE: recorded for offline replay, never fed back into `candidates`.
                _ecb_extra = _ecb_select(
                    probs, n_src, n_tgt, _ECB_EXPORT_THRESHOLD, _ECB_EXPORT_TOPK,
                )
                if _ecb_extra:
                    _ECB_BUFFER.append(np.array(
                        [(int(idx_src[i]), int(idx_tgt[j]), float(p)) for p, i, j in _ecb_extra],
                        dtype=np.float64,
                    ))"""
assert _ecb_src.count(_ecb_old) == 1, (
    f"edge-candidate budget: PRIMARY candidate anchor count {_ecb_src.count(_ecb_old)} "
    "(expected exactly 1) - the primary path was NOT patched, refusing to continue"
)
_ecb_src = _ecb_src.replace(_ecb_old, _ecb_new, 1)

# Flush one sidecar per crop at predict_video's return, then clear the per-crop state. The
# buffer is cleared HERE rather than initialised at entry because the only anchor available
# at entry sits inside the sliding-window loop and would reset once per window.
_ecb_ret_old = """    coords = coords.astype(np.int16)
    return coords, all_edges"""
_ecb_ret_new = """    coords = coords.astype(np.int16)
    if _ECB_EXPORT_ON:
        from pathlib import Path as _EcbPath
        _ecb_out = _EcbPath(_ECB_EXPORT_DIR)
        _ecb_out.mkdir(parents=True, exist_ok=True)
        _ecb_rows = (
            np.concatenate(_ECB_BUFFER) if _ECB_BUFFER else np.empty((0, 3), dtype=np.float64)
        )
        _ecb_tmp = _ecb_out / f"{ds_path.stem}.npz.tmp"
        np.savez_compressed(
            _ecb_tmp,
            source_id=_ecb_rows[:, 0].astype(np.int64),
            target_id=_ecb_rows[:, 1].astype(np.int64),
            edge_prob=_ecb_rows[:, 2].astype(np.float32),
            deployed_threshold=np.float64(cfg.threshold),
            export_threshold=np.float64(_ECB_EXPORT_THRESHOLD),
            export_topk=np.int64(_ECB_EXPORT_TOPK),
            deployed_candidate_count=np.int64(_ECB_CROP["pairs"]),
            frame_pairs=np.int64(_ECB_CROP["frame_pairs"]),
        )
        _ecb_tmp.replace(_ecb_out / f"{ds_path.stem}.npz")
        _ECB_TOTAL["crops"] += 1
        _ECB_TOTAL["pairs"] += _ECB_CROP["pairs"]
        _ECB_TOTAL["exported"] += len(_ecb_rows)
        print(
            f"ECB_EXPORT crop={ds_path.stem} deployed_threshold={cfg.threshold} "
            f"export_threshold={_ECB_EXPORT_THRESHOLD} export_topk={_ECB_EXPORT_TOPK} "
            f"deployed_candidates={_ECB_CROP['pairs']} exported_candidates={len(_ecb_rows)} "
            f"frame_pairs={_ECB_CROP['frame_pairs']} crops_done={_ECB_TOTAL['crops']} "
            f"total_exported={_ECB_TOTAL['exported']}",
            flush=True,
        )
        _ECB_BUFFER.clear()
        _ECB_CROP["pairs"] = 0
        _ECB_CROP["frame_pairs"] = 0
    return coords, all_edges"""
assert _ecb_src.count(_ecb_ret_old) == 1, (
    f"edge-candidate budget: predict_video return anchor count "
    f"{_ecb_src.count(_ecb_ret_old)} (expected exactly 1)"
)
_ecb_src = _ecb_src.replace(_ecb_ret_old, _ecb_ret_new, 1)

compile(_ecb_src, str(_ps), "exec")
_ps.write_text(_ecb_src)
print(
    "ECB_PATCH_APPLIED primary candidate path patched | "
    f"treatment_threshold={_ecb_patch_os.environ.get('BIOHUB_EDGE_CANDIDATE_THRESHOLD', '') or 'unset'} "
    f"treatment_topk={_ecb_patch_os.environ.get('BIOHUB_EDGE_CANDIDATE_TOPK', '') or 'unset'} "
    f"export_threshold={_ecb_patch_os.environ.get('BIOHUB_EDGE_CANDIDATE_EXPORT_THRESHOLD', '') or 'unset'} "
    f"export_topk={_ecb_patch_os.environ.get('BIOHUB_EDGE_CANDIDATE_EXPORT_TOPK', '') or '8 (default)'} "
    f"export_dir={_ecb_patch_os.environ.get('BIOHUB_EDGE_CANDIDATE_EXPORT_DIR', '') or 'unset'}",
    flush=True,
)
