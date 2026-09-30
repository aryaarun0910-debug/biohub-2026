# =====================================================================================
# L3 -- STOP THE WRAPPER FROM DELETING THE ILP'S DIVISIONS
#
# WHY. Measured 2026-08-18 (kernel biohub-p4-preilp-loeo-f1 v2): the ILP is offered 176,835
# sources with candidate out-degree >= 2 on fold 1 and emits ZERO divisions, because our own
# BIOHUB_ILP_APPEARANCE_WEIGHT="0.0" makes conversion cost
#     division_weight - appearance_weight - p = 1.0 - p >= 0   for every p <= 1.
# Of 125 GT divisions, 29 had BOTH true daughters offered to the solver, at median edge_prob
# 0.9188. Setting BIOHUB_ILP_DIVISION_WEIGHT=0.55 (L1) makes the solver emit them.
#
# But L1 alone changes NOTHING downstream, because `motion_relink_edges` returns a strict
# one-to-one Hungarian matching and REPLACES `edges` wholesale (wrapper.py:1158-1161). Every ILP
# division dies there. This stage re-admits them after the relink.
#
# SAFETY. Additive only -- it can never remove an edge. Every re-admitted edge satisfies the
# wrapper's own invariants (out-degree <= 2, in-degree <= 1), so `assert_degree_invariants`
# still passes. Verified inert when the flag is off.
#
# Downstream filters that could otherwise undo this were checked and are all OFF by default and
# not overridden by the deployed notebook:
#   OUTPUT_SINGLE_CHILD_REPAIR       default "0"  (would keep only the best child per source)
#   OUTPUT_DIVISION_GEOMETRY_FILTER  default "0"  (would re-filter multi-child sources on geometry)
# OUTPUT_SINGLE_PARENT_REPAIR is ON, but only enforces in-degree <= 1, which divisions respect.
#
# Ships default-OFF behind BIOHUB_RESTORE_LEARNED_DIVISIONS, matching the flag convention at
# wrapper.py:40-56.
# =====================================================================================
RESTORE_LEARNED_DIVISIONS = os.environ.get("BIOHUB_RESTORE_LEARNED_DIVISIONS", "0") != "0"


def restore_learned_divisions(pre_relink_edges, relinked_edges, nodes_by_id, stats):
    """Re-admit second-child edges the ILP emitted and the motion relink discarded.

    An edge (s, t) from the pre-relink (ILP) graph is re-admitted iff ALL hold:
      * s had out-degree >= 2 before the relink   -- the ILP genuinely chose a division
      * s has exactly one child after the relink  -- we are restoring the SECOND child only
      * t has no parent after the relink          -- keeps in-degree <= 1, no contention
      * t is at frame(s) + 1                      -- the wrapper's consecutive-frame rule
      * (s, t) is not already present             -- no duplicates
    """
    if not pre_relink_edges or not relinked_edges:
        return relinked_edges

    pre_out: dict[int, list] = {}
    for _e in pre_relink_edges:
        pre_out.setdefault(int(_e["source_id"]), []).append(_e)

    post_out: dict[int, int] = {}
    post_in: set[int] = set()
    present: set[tuple[int, int]] = set()
    for _e in relinked_edges:
        _s, _t = int(_e["source_id"]), int(_e["target_id"])
        post_out[_s] = post_out.get(_s, 0) + 1
        post_in.add(_t)
        present.add((_s, _t))

    restored: list = []
    for _src, _cands in pre_out.items():
        if len(_cands) < 2:
            continue
        stats["restore_div_sources_seen"] = stats.get("restore_div_sources_seen", 0) + 1
        if post_out.get(_src, 0) != 1:
            stats["restore_div_skipped_outdeg"] = stats.get("restore_div_skipped_outdeg", 0) + 1
            continue
        _source = nodes_by_id.get(_src)
        if _source is None:
            continue
        # strongest first, so a capped restore keeps the ILP's best second child
        for _e in sorted(_cands, key=lambda q: float(q.get("edge_prob") or 0.0), reverse=True):
            _tgt = int(_e["target_id"])
            if (_src, _tgt) in present:
                continue
            if _tgt in post_in:
                stats["restore_div_skipped_target_taken"] = (
                    stats.get("restore_div_skipped_target_taken", 0) + 1)
                continue
            _target = nodes_by_id.get(_tgt)
            if _target is None:
                continue
            if int(_target["t"]) != int(_source["t"]) + 1:
                continue
            restored.append(_e)
            post_in.add(_tgt)
            present.add((_src, _tgt))
            post_out[_src] = post_out.get(_src, 0) + 1
            break   # at most ONE second child per source -> out-degree <= 2

    stats["restore_div_restored"] = len(restored)
    if restored:
        print(f"restore_learned_divisions: re-admitted {len(restored)} ILP divisions "
              f"(sources with pre-relink out-degree>=2: "
              f"{stats.get('restore_div_sources_seen', 0)})", flush=True)
    return relinked_edges + restored
