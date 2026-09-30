"""H1-R sub-voxel refine kaggle_edit -- injected into the detector-patch cell.

Replaces extract_coords_from_logits int16 return with a parabolic per-axis sub-voxel
offset (float coords). See scripts/win_bet/h1r_subvoxel_refine.py for the math + self-test.
The patch body is embedded in the spec (below) as a `replace` before the shard launch.
"""

PATCH = r'''
# --- H1-R sub-voxel centroid refinement (parabolic per-axis offset; float coords) ---
_s = _ps.read_text()
_sv_old = "    coords = peak_idx.float().cpu().numpy()\n    t_col = np.full((len(coords), 1), t, dtype=np.float32)\n    return np.concatenate([t_col, coords], axis=1).astype(np.int16)"
_sv_new = (
    "    coords = peak_idx.float()\n"
    "    _lv = det_logits[0]\n"
    "    _pk = peak_idx.long()\n"
    "    _off = torch.zeros_like(coords)\n"
    "    for _a, _n in enumerate(_lv.shape):\n"
    "        _c = _pk[:, _a]\n"
    "        _int = (_c > 0) & (_c < _n - 1)\n"
    "        if _int.any():\n"
    "            _pm = _pk[_int].clone(); _pm[:, _a] -= 1\n"
    "            _pp = _pk[_int].clone(); _pp[:, _a] += 1\n"
    "            _p0 = _pk[_int]\n"
    "            _lm = _lv[_pm[:,0], _pm[:,1], _pm[:,2]]\n"
    "            _l0 = _lv[_p0[:,0], _p0[:,1], _p0[:,2]]\n"
    "            _lp = _lv[_pp[:,0], _pp[:,1], _pp[:,2]]\n"
    "            _den = _lm - 2.0*_l0 + _lp\n"
    "            _d = torch.where(_den < -1e-9, 0.5*(_lm - _lp)/_den, torch.zeros_like(_den)).clamp(-0.5, 0.5)\n"
    "            _off[_int, _a] = _d\n"
    "    coords = (coords + _off).cpu().numpy()\n"
    "    t_col = np.full((len(coords), 1), t, dtype=np.float32)\n"
    "    return np.concatenate([t_col, coords], axis=1).astype(np.float32)"
)
assert _s.count(_sv_old) == 1, f"H1-R subvoxel: extract_coords block count {_s.count(_sv_old)}"
_s = _s.replace(_sv_old, _sv_new, 1)
compile(_s, str(_ps), "exec")
_ps.write_text(_s)
print("H1-R sub-voxel refine patch applied (float sub-voxel centroids)")

'''
