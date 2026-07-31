# --- P1 node-budget stage (deployable port of scripts/win_bet/phaseb_node_budget.py) ---
#
# PROVISIONAL.  The corpus measurement behind this stage (+0.00157 pooled) was taken on
# E0c arm A, whose node ratio is +0.0832 (over-prediction).  The stage's realised gain is
# dominated by the count multiplier  adj_J = J * (1 - 0.1 * (N_pred - N_est) / N_est),
# whose first-order sensitivity to a node cut of fraction f is  +0.1 * f * (1 + r0) * J.
# On an under-predicting substrate (r0 < 0) that term SHRINKS but does not change sign;
# what decides the net effect is the edge-Jaccard cost of the components removed.  The
# P0-B substrate already runs `filter_short_track_components`, so its weakest surviving
# components are stronger than E0c's and the cost term is expected to be larger.
# DO NOT quote +0.00157 as this stage's expected effect on P0-B.
#
# Mechanism, byte-for-byte the offline ranking:
#   * weakly connected components over the final emitted graph;
#   * a component containing any fork (out-degree >= 2) is protected -- dropped last;
#   * otherwise weakest-first = shortest component first;
#   * ties broken by smallest node id (the offline script relies on stable sort over
#     components enumerated in ascending node-id order; this is the same order, made
#     explicit so the result does not depend on sort stability);
#   * whole components are dropped until the surviving node count reaches the budget;
#   * edges with a dropped endpoint are dropped with them.
#
# No ground truth, no family/crop identity, no image data, no threshold fitted here.

NODE_BUDGET_KEEP_FRAC = 0.975


def _nb_components(node_ids, edges):
    """Weakly connected components, enumerated in ascending node-id order."""
    adj = {n: set() for n in node_ids}
    for e in edges:
        a = int(e["source_id"])
        b = int(e["target_id"])
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen = set()
    comps = []
    for n in node_ids:
        if n in seen:
            continue
        stack = [n]
        cur = []
        seen.add(n)
        while stack:
            u = stack.pop()
            cur.append(u)
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(cur)
    return comps


def apply_node_budget(nodes_by_id, edges, keep_frac):
    """Drop the weakest whole components until <= keep_frac of the nodes remain.

    Returns (nodes_by_id, edges, stats).  keep_frac >= 1.0 is an exact no-op.
    """
    node_ids = sorted(nodes_by_id)
    n_nodes = len(node_ids)
    stats = {
        "nb_keep_frac": float(keep_frac),
        "nb_nodes_before": n_nodes,
        "nb_edges_before": len(edges),
        "nb_components": 0,
        "nb_components_dropped": 0,
        "nb_nodes_dropped": 0,
        "nb_edges_dropped": 0,
        "nb_forks_dropped": 0,
    }
    if keep_frac >= 1.0 or n_nodes == 0:
        return nodes_by_id, edges, stats

    outdeg = {}
    for e in edges:
        a = int(e["source_id"])
        outdeg[a] = outdeg.get(a, 0) + 1
    forks_before = sum(1 for v in outdeg.values() if v >= 2)

    comps = _nb_components(node_ids, edges)
    stats["nb_components"] = len(comps)
    comps.sort(key=lambda c: (1 if any(outdeg.get(n, 0) >= 2 for n in c) else 0,
                              len(c), min(c)))

    target = int(round(keep_frac * n_nodes))
    drop = set()
    i = 0
    while n_nodes - len(drop) > target and i < len(comps):
        drop.update(comps[i])
        i += 1
    if not drop:
        return nodes_by_id, edges, stats

    keep = {n: nodes_by_id[n] for n in node_ids if n not in drop}
    kept_edges = [e for e in edges
                  if int(e["source_id"]) in keep and int(e["target_id"]) in keep]

    outdeg_after = {}
    for e in kept_edges:
        a = int(e["source_id"])
        outdeg_after[a] = outdeg_after.get(a, 0) + 1
    forks_after = sum(1 for v in outdeg_after.values() if v >= 2)

    stats["nb_components_dropped"] = i
    stats["nb_nodes_dropped"] = len(drop)
    stats["nb_edges_dropped"] = len(edges) - len(kept_edges)
    stats["nb_forks_dropped"] = forks_before - forks_after
    return keep, kept_edges, stats
# --- end P1 node-budget stage ---
