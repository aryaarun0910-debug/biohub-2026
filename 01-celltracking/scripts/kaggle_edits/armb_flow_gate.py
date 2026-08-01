# --- ARM B: flow-compensated gate quantity for `motion_relink_edges` -----------------
#
# THE ONLY SEMANTIC CHANGE from the P0-B deployment source.
#
# Deployed (arm A):  raw    = |target - source|                  ; if raw    > gate_um: continue
# Arm B:             gate_q = |target - (source + flow(source))| ; if gate_q > gate_um: continue
#
# The RADIUS is unchanged (tight 6.0 um / relaxed 10.0 um), the relink cost expression is
# unchanged (`motion + 0.05*raw - MOTION_RELINK_LEARNED_BONUS*prob`), the node population handed
# to the relink is unchanged, and `motion` is still measured against the velocity extrapolation.
# Only the eligibility predicate moves.
#
# `GATE_COMPENSATION` maps node_id -> 3-vector (um) added to the source position before the
# radius test. An EMPTY dict restores the deployed raw source-target distance bit-for-bit, which
# is what the baseline regression arm runs (BIOHUB_ARMB_FLOW_GATE=0).
#
# The flow definition is a literal port of the one WS-A/WS-F measured arm B with:
#   research agent_runs/ws_f_armB_p0b_2026-08-01/scripts/wsf_armB_f0.py::flow_vectors (:66-97)
#   research agent_runs/ws_f_armB_p0b_2026-08-01/scripts/wsf_armB_f0.py::comp_by_node (:100-121)
# It is GT-free, image-free, model-free and routes on nothing: no acquisition state, no family,
# no crop identity. It reads only the pre-wrapper prediction graph of the crop being processed.
#
# Evidence (do not re-derive; see reports/PRIMITIVE_MATRIX.md WS-A / WS-F):
#   P0-strict LOEO, complete wrapper, 144/144 crops parity-exact:
#     pooled +0.0079822 | 44b6 +0.0167567 | 6bba +0.0067055 | P(d>0)=1.000 on all three
#   E0c, 199/199 crops: +0.0088059 pooled.
# CAVEAT carried forward: at the tight radius arm B admits slightly MORE pairs than arm A
# (+0.169% on E0c, +0.388% on P0-strict). It is a RE-AIM with a small net widening, not monotone
# widening -- B is not a superset of A (7.2% churn). `ARMB_GATE_STATS` measures this on the
# deployment substrate rather than assuming it.

ARMB_FLOW_GATE = os.environ.get("BIOHUB_ARMB_FLOW_GATE", "1") != "0"
ARMB_KNN_K = 16
ARMB_KNN_MIN = 4

GATE_COMPENSATION: dict[int, np.ndarray] = {}
GATE_STATS: dict[str, int] = {}


def _gate_reset() -> None:
    GATE_STATS.clear()
    for _k in ("pairs_tight", "admit_raw_tight", "admit_arm_tight",
               "newly_admitted_tight", "newly_excluded_tight",
               "pairs_relaxed", "admit_raw_relaxed", "admit_arm_relaxed",
               "newly_admitted_relaxed", "newly_excluded_relaxed"):
        GATE_STATS[_k] = 0


def _armb_flow_compensation(
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
) -> dict[int, np.ndarray]:
    """GT-free kNN16 flow, node_id -> displacement vector in um.

    Verbatim in behaviour with WS-A/WS-F's `flow_vectors` + `comp_by_node`:
    raw prediction-graph edges of the same frame pair, restricted to out-degree 1 and
    in-degree 1 within the wrapper's own <= OUTPUT_EDGE_MAX_UM horizon; per-frame kD-tree;
    the flow of a node is the median of the k=16 nearest source displacements. Falls back to
    the frame median when a frame has fewer than 4 usable sources, and to the global median
    when a frame contributes no usable source at all.
    """
    pos_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}

    src_ids: list[int] = []
    dst_ids: list[int] = []
    for edge in raw_edges:
        a = int(edge["source_id"])
        b = int(edge["target_id"])
        node_a = nodes_by_id.get(a)
        node_b = nodes_by_id.get(b)
        if node_a is None or node_b is None:
            continue
        if int(node_b["t"]) != int(node_a["t"]) + 1:
            continue
        if float(np.linalg.norm(pos_um[b] - pos_um[a])) > OUTPUT_EDGE_MAX_UM:
            continue
        src_ids.append(a)
        dst_ids.append(b)

    outd: dict[int, int] = {}
    ind: dict[int, int] = {}
    for a, b in zip(src_ids, dst_ids):
        outd[a] = outd.get(a, 0) + 1
        ind[b] = ind.get(b, 0) + 1

    by_frame: dict[int, tuple[list, list]] = {}
    for a, b in zip(src_ids, dst_ids):
        if outd[a] != 1 or ind[b] != 1:
            continue
        f = int(nodes_by_id[a]["t"])
        s, d = by_frame.setdefault(f, ([], []))
        s.append(pos_um[a])
        d.append(pos_um[b] - pos_um[a])

    all_d = [d for value in by_frame.values() for d in value[1]]
    global_median = np.median(np.stack(all_d), axis=0) if all_d else np.zeros(3)

    built: dict[int, tuple] = {}
    for f, (s, d) in by_frame.items():
        S = np.stack(s)
        D = np.stack(d)
        built[f] = (cKDTree(S), S, D, np.median(D, axis=0))

    frames: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        frames.setdefault(int(node["t"]), []).append(node_id)

    out: dict[int, np.ndarray] = {}
    for f, ids in frames.items():
        ids_sorted = sorted(ids)
        P = np.stack([pos_um[i] for i in ids_sorted])
        entry = built.get(f)
        if entry is None:
            fl = np.tile(global_median, (len(ids_sorted), 1))
        else:
            tree, S, D, frame_median = entry
            if len(S) < ARMB_KNN_MIN:
                fl = np.tile(frame_median, (len(ids_sorted), 1))
            else:
                k = min(ARMB_KNN_K, len(S))
                _, jj = tree.query(P, k=k)
                fl = np.median(D[np.atleast_2d(jj)], axis=1)
        for node_id, v in zip(ids_sorted, fl):
            out[int(node_id)] = np.asarray(v, dtype=np.float64)
    return out


_gate_reset()          # counters accumulate over every crop of the run
