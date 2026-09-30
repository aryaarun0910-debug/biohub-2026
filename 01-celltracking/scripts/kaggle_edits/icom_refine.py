# LEVER-0022 deployment edit (PKT-0018): intensity-centroid re-localisation of every exported node.
#
# Inserted into the deployed P3-harmonic/P9 notebook immediately before the submission writer, i.e.
# AFTER linefit smoothing and every graph operation, so it changes coordinates only - never topology.
# It is the exact operator measured offline by scripts/win_bet/dc_subvoxel_refine.py (icentroid arm
# `icom_155`): intensity-weighted centroid of the RAW frame in a z+-1, y+-5, x+-5 VOXEL window with a
# 20th-percentile baseline and a 2.8 um shift cap (non-binding for this window). Origin: the 0.927 public
# lineage's refine_centroids (arnav170/biohub-sdw60 cell13:74-113), which that lineage then ROUNDED
# away at export; FACT-0324 measured that the gain exists only with FLOAT coordinates, so the writer's
# int(round()) is replaced by a float format in the same spec (see p23_icom155.json).
#
# Env: BIOHUB_ICOM_REFINE (default "1"), BIOHUB_ICOM_WIN_Z/YX (1 / 5), BIOHUB_ICOM_BASELINE_PCT (20),
# BIOHUB_ICOM_MAX_SHIFT_UM (2.8). Counters land in `stats` so the run log proves it ran.
ICOM_REFINE = os.environ.get("BIOHUB_ICOM_REFINE", "1") != "0"
ICOM_WIN_Z = int(os.environ.get("BIOHUB_ICOM_WIN_Z", "1"))
ICOM_WIN_YX = int(os.environ.get("BIOHUB_ICOM_WIN_YX", "5"))
ICOM_BASELINE_PCT = float(os.environ.get("BIOHUB_ICOM_BASELINE_PCT", "20"))
ICOM_MAX_SHIFT_UM = float(os.environ.get("BIOHUB_ICOM_MAX_SHIFT_UM", "2.8"))


def icom_refine_frame(vol, coords, win_z, win_yx, baseline_pct, max_shift_um):
    """Vectorised-per-node intensity centroid on one raw frame; returns (N,3) float64 and a moved mask."""
    Z, Y, X = vol.shape
    out = coords.astype(np.float64).copy()
    moved = np.zeros(len(coords), dtype=bool)
    scale = np.asarray(VOXEL_SCALE_UM, dtype=np.float64)
    for i, original in enumerate(coords):
        z, y, x = [int(round(float(v))) for v in original]
        z0, z1 = max(0, z - win_z), min(Z, z + win_z + 1)
        y0, y1 = max(0, y - win_yx), min(Y, y + win_yx + 1)
        x0, x1 = max(0, x - win_yx), min(X, x + win_yx + 1)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue
        patch = vol[z0:z1, y0:y1, x0:x1].astype(np.float64)
        baseline = float(np.percentile(patch, baseline_pct))
        w = np.maximum(patch - baseline, 0.0)
        total = float(w.sum())
        if total <= 0.0:
            continue
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
        refined = np.array([(w * zz).sum() / total, (w * yy).sum() / total, (w * xx).sum() / total])
        if float(np.sqrt((((refined - original.astype(np.float64)) * scale) ** 2).sum())) <= max_shift_um:
            out[i] = refined
            moved[i] = True
    return out, moved


def icom_refine_output_graph(nodes_by_id, dataset, stats):
    """Re-localise every node of the FINAL graph from the raw frames. Coordinates only; ids/edges untouched."""
    if not ICOM_REFINE or not nodes_by_id:
        stats["icom_refine_skipped"] = int(stats.get("icom_refine_skipped", 0)) + 1
        return nodes_by_id
    frame_cache = {}
    by_t = {}
    for node_id, node in nodes_by_id.items():
        by_t.setdefault(int(node["t"]), []).append(node_id)
    n_moved = 0
    for t, ids in by_t.items():
        vol = read_test_frame(dataset, int(t), frame_cache)
        coords = np.array([[float(nodes_by_id[i]["z"]), float(nodes_by_id[i]["y"]), float(nodes_by_id[i]["x"])] for i in ids],
                          dtype=np.float64)
        refined, moved = icom_refine_frame(vol, coords, ICOM_WIN_Z, ICOM_WIN_YX, ICOM_BASELINE_PCT, ICOM_MAX_SHIFT_UM)
        for k, node_id in enumerate(ids):
            if moved[k]:
                nodes_by_id[node_id]["z"] = float(refined[k, 0])
                nodes_by_id[node_id]["y"] = float(refined[k, 1])
                nodes_by_id[node_id]["x"] = float(refined[k, 2])
                n_moved += 1
        if len(frame_cache) > 4:
            frame_cache.pop(next(iter(frame_cache)))
    stats["icom_refine_nodes"] = int(stats.get("icom_refine_nodes", 0)) + len(nodes_by_id)
    stats["icom_refine_moved"] = int(stats.get("icom_refine_moved", 0)) + n_moved
    print(f"icom_refine: {dataset} moved {n_moved}/{len(nodes_by_id)} nodes (win z{ICOM_WIN_Z} yx{ICOM_WIN_YX}, "
          f"baseline p{ICOM_BASELINE_PCT:g}, cap {ICOM_MAX_SHIFT_UM} um)")
    return nodes_by_id
