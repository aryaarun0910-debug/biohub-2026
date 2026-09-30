# --- OUT-DEGREE INVARIANT: fail at the stage that breaks it, not at the submission guard ---
#
# The pipeline assumes lineage degrees in <= 1 and out <= 2 everywhere downstream, but no
# stage enforced out <= 2. `add_safe_divisions_postlink` dedupes its proposals by TARGET
# (`used_targets` / `incoming`) and never by SOURCE, and never advances `out_by_source` as
# it admits. A source with one existing child can therefore receive TWO safe divisions in
# the same frame and reach out-degree 3.
#
# That is what blocked the first arm-B deployment run: exactly one node in 121,003
# (`6bba_05db0fb1`) at out-degree 3, zero in-degree violations -- the signature of a
# target-side dedupe with no source-side dedupe.
#
# NOTE the relink is NOT the site. `motion_relink_edges` replaces the whole edge list and
# assigns one-to-one per frame pair, so it emits out-degree <= 1 by construction; a guard
# at relink admission is a no-op. Arm B changes WHICH targets are left unlinked, which
# shifts the safe-division candidate population and makes the latent defect fire.


def assert_degree_invariants(edges, stage):
    """Lineage degrees the whole pipeline assumes: in <= 1, out <= 2."""
    _out_degree = {}
    _in_degree = {}
    for _edge in edges:
        _s = int(_edge["source_id"])
        _t = int(_edge["target_id"])
        _out_degree[_s] = _out_degree.get(_s, 0) + 1
        _in_degree[_t] = _in_degree.get(_t, 0) + 1
    _bad_out = sorted(n for n, d in _out_degree.items() if d > 2)
    _bad_in = sorted(n for n, d in _in_degree.items() if d > 1)
    if _bad_out or _bad_in:
        raise RuntimeError(
            "degree invariant violated after " + str(stage)
            + ": out-degree>2 on " + repr(_bad_out[:8])
            + " (n=" + str(len(_bad_out)) + "), in-degree>1 on "
            + repr(_bad_in[:8]) + " (n=" + str(len(_bad_in)) + ")"
        )
