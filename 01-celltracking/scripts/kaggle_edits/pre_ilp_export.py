# =====================================================================================
# PRE-ILP CANDIDATE-GRAPH EXPORT  (falsification measurement for the linker-division lane)
#
# WHY. Measured 2026-08-18: the deployed pipeline emits divisions ONLY from the post-hoc
# `add_safe_divisions_postlink` patch -- the linker itself never produces out-degree 2
# (pre-wrapper histogram is {1: 1,976,653}). The root cause is economic, not structural:
# we set BIOHUB_ILP_APPEARANCE_WEIGHT="0.0" (vendor default 0.1), so under the minimised
# min-cost-flow objective, converting a link into a division costs
#     division_weight - appearance_weight - p = 1.0 - p >= 0    for every p <= 1,
# i.e. division is strictly dominated and the solver always prefers "daughter appears
# from nothing". See research/06-knowledge-system/internal-reports/
# linker_division_capacity_2026-08-18.md and the 2026-08-18 experimental record.
#
# Before spending a pilot on changing those weights, ONE number decides whether the ILP is
# discarding real divisions at all: how many sources carry >= 2 CANDIDATE edges in the graph
# the solver is handed, and how many of those sit at a true dividing mother. That graph
# already exists in memory one line before the solver; nothing needs recomputing.
#
# KILL CRITERION (stated before the run): if fewer than ~15 of the 31 fold-1 orphan cases had
# a declined candidate, the ILP is not throwing real divisions away and lane L1 is dead --
# the second daughter simply never scored above the hard 0.5 candidate threshold.
#
# TRAP THIS AVOIDS. The notebook globs `predictions/*/{METHOD}/split_0/*.geff` and hard-fails
# unless the count equals len(test_stems). Writing `*.preilp.geff` beside the real outputs
# would double that count and kill the run at the assertion. So the export goes to a separate
# /kaggle/working/preilp directory, outside the glob.
#
# Cost: ~+2% kernel wall-clock, no extra GPU, no change to the emitted graph -- the ILP still
# consumes exactly the object it always did, so the submission is bit-identical to the base arm.
# =====================================================================================
_pi_old = """        graph = build_graph(coords, edges)
        if cfg.use_ilp and graph.num_edges() > 0:"""
_pi_new = """        graph = build_graph(coords, edges)
        try:
            _pi_dir = Path("/kaggle/working/preilp")
            _pi_dir.mkdir(parents=True, exist_ok=True)
            save_graph(graph, _pi_dir / f"{name}.geff")
        except Exception as _pi_err:
            print(f"PREILP export failed for {name}: {_pi_err}", flush=True)
        if cfg.use_ilp and graph.num_edges() > 0:"""

_s = _ps.read_text()
assert _s.count(_pi_old) == 1, f"pre-ILP export: anchor count {_s.count(_pi_old)}"
_s = _s.replace(_pi_old, _pi_new, 1)
compile(_s, str(_ps), "exec")
_ps.write_text(_s)
print("pre-ILP candidate-graph export patch applied (-> /kaggle/working/preilp)")
