"""Build s07: move gap2 recovery AFTER safe-division repair, and prove it.

The finding (artifacts/other_stages.log, scripts/91_other_stages.py):

    ANCHOR raw + safe_div                 J 0.93223   div 5/2/7  divJ 0.3571
    + gap2 total=10.2 (BEFORE safe_div)   J 0.93355   div 4/3/8  divJ 0.2667  -0.00802
    + safe_div then gap2  (AFTER)         J 0.93371   div 5/2/7  divJ 0.3571  +0.00121
    BEST gc8+sd+gap2+lf0.4/w3             J 0.94059   div 5/2/7  divJ 0.3571  +0.00761

gap2 joins a track end at t to a start at t+3. That start is an orphan -- no
incoming edge -- which is exactly the pool add_safe_divisions_postlink draws
its second daughters from (`candidate_ids = [... if node_id not in incoming]`).
Run gap2 first and it eats a daughter: one TP division becomes an FP + an FN.
Run safe_div first and gap2 still gets its +0.0015 J of edge gain, because the
daughter it loses is one node out of ~750.

linefit is a separate case: it rewrites node z/y/x in place (lines 2936-2938
of cell 2), and safe_div's SAFE_DIV_MAX_UM / SISTER_MAX_UM / DIVERGE_UM gates
read those coordinates. Smoothing first collapses divisions to 3/1/9 or 2/3/10.
In the deployed notebook linefit is ALREADY the last stage in the chain, so it
needs no move -- this script verifies that rather than assuming it.

This is a pure statement reorder inside filter_output_graph. It is checked
three ways before the notebook is written: a line diff, an AST comparison that
requires the new statement sequence to be a permutation of the old one with no
statement added/removed/altered anywhere in the cell, and a compile().
"""
import ast
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "public notebooks/biohub-0-947-lb-runnable-with-public-datasets.ipynb"
S05 = ROOT / "submissions/s05_no_relink/biohub-s05-no-relink.ipynb"

# Two build targets. s07 puts the reorder on the unmodified base (relink ON);
# docs/HANDOFF.md section 6 says do not push that one -- gap2 competes with
# safe_div for orphan nodes and relink is what manufactures the orphan pool, so
# a delta measured with relink off need not survive with it on. s08 puts the
# same reorder on top of s05 (relink OFF), which is the topology the +0.00922
# three-stage delta and the +0.00748 full-chain delta were both measured in
# (scripts/91_other_stages.py, scripts/98_reorder_on_norelink.py).
TARGETS = {
    "s07": dict(out="submissions/s07_reorder", slug="biohub-s07-reorder",
                title="Biohub S07 gap2 after safe-division", env=False),
    # NOTE: Kaggle derives the live kernel slug by slugifying the TITLE, not the
    # `id` in kernel-metadata.json. s05 shipped as id=biohub-s05-no-relink but
    # went live at biohub-s05-no-motion-relink (= slugify("Biohub S05 no motion
    # relink")); s06's log is biohub-s06-linefit-weight-0-4.log from "Biohub S06
    # linefit weight 0.4". So keep title and slug in sync or the push lands
    # somewhere you did not predict and `kernels status` 404s.
    "s08": dict(out="submissions/s08_reorder_on_s05",
                slug="biohub-s08-reorder-on-s05",
                title="Biohub S08 reorder on s05",
                env=True),
}

# The s05 change, inserted at exactly the position s05 uses so that s08 differs
# from the shipped s05 notebook by the reorder and nothing else.
ENV_ANCHOR = "os.environ['BIOHUB_OUTPUT_GAP2_RECOVERY'] = '1'"
ENV_LINE = "os.environ['BIOHUB_OUTPUT_MOTION_RELINK'] = '0'\n"

FUNC = "filter_output_graph"
CODE_CELL = 2

# The two statements that move, and the statement they land behind. Matched on
# a distinctive substring; each must match exactly one line or we abort.
GAP2_CALL = "recover_strict_gap2("
GAP2_PRINT = "after gap-closing (single-frame + gap2)"
LANDING = "after safe-division repair:"

SAFE_DIV = "add_safe_divisions_postlink("
GAP_CLOSE = "close_single_frame_gaps("
LINEFIT = "linefit_smooth_output_graph("

GUARD_KEY = "_EXPECTED_NUMERIC"


def die(msg):
    sys.exit(f"ABORT: {msg}")


def only(lines, needle, what):
    """Index of the unique line containing `needle`, ignoring def statements."""
    hits = [i for i, l in enumerate(lines)
            if needle in l and not l.lstrip().startswith("def ")]
    if len(hits) != 1:
        die(f"{what}: {needle!r} matched {len(hits)} lines, expected 1")
    return hits[0]


def stage_order(src):
    """Stage call sites in source order, as (line_no, stage name)."""
    out = []
    for i, line in enumerate(src.split("\n"), 1):
        for name in (GAP_CLOSE, GAP2_CALL, SAFE_DIV, LINEFIT):
            if name in line and "def " not in line:
                out.append((i, name.rstrip("(")))
    return out


def func_body(src):
    """Top-level statements of filter_output_graph, as ast dumps."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == FUNC:
            return [ast.dump(s) for s in node.body]
    die(f"{FUNC} not found")


STAGES = ((GAP_CLOSE, "close_single_frame_gaps"),
          (GAP2_CALL, "recover_strict_gap2"),
          (SAFE_DIV, "add_safe_divisions_postlink"),
          (LINEFIT, "linefit_smooth_output_graph"))


def show_sequence(body, title):
    print(f"    {title}")
    for k, dump in enumerate(body):
        for needle, short in STAGES:
            if f"id='{needle.rstrip('(')}'" in dump:
                print(f"      [{k:2d}] {short}")


def main(target="s08"):
    spec = TARGETS[target]
    out_dir, slug, title = ROOT / spec["out"], spec["slug"], spec["title"]
    print(f"building {target}: {spec['title']}\n"
          f"  reorder on {'the s05 no-relink base' if spec['env'] else 'the unmodified base'}\n")
    nb = json.loads(BASE.read_text())
    before_lines = [l for c in nb["cells"] for l in c.get("source", [])]
    cell = nb["cells"][CODE_CELL]
    if cell["cell_type"] != "code":
        die(f"cell {CODE_CELL} is {cell['cell_type']}, expected code")
    src_list = list(cell["source"])
    old_src = "".join(src_list)

    # ---------------------------------------------------------- before state
    print(f"base notebook: {len(before_lines)} source lines across "
          f"{len(nb['cells'])} cells; cell {CODE_CELL} has {len(src_list)} lines")
    print("\ncurrent stage order in filter_output_graph:")
    for ln, name in stage_order(old_src):
        print(f"  line {ln}: {name}")

    # ------------------------------------------------------------- the move
    i_call = only(src_list, GAP2_CALL, "gap2 call")
    i_print = only(src_list, GAP2_PRINT, "gap2 summary print")
    i_land = only(src_list, LANDING, "safe-division summary print")
    i_sd = only(src_list, SAFE_DIV, "safe-division call")

    if i_print != i_call + 1:
        die("gap2 call and its summary print are not adjacent")
    if not (i_call < i_sd < i_land):
        die("unexpected layout: expected gap2 call, then safe_div call, then its print")
    i_lf = only(src_list, LINEFIT, "linefit call")
    if i_lf < i_land:
        die("linefit is not already after safe-division; this patch assumes it is")
    print(f"\nlinefit already sits at line {i_lf + 1}, after safe-division at "
          f"line {i_sd + 1} -- no move needed for it.")

    block = src_list[i_call:i_print + 1]
    rest = src_list[:i_call] + src_list[i_print + 1:]
    j_land = only(rest, LANDING, "safe-division summary print (after removal)")
    new_list = rest[:j_land + 1] + block + rest[j_land + 1:]

    if len(new_list) != len(src_list):
        die("line count changed")
    if sorted(new_list) != sorted(src_list):
        die("line multiset changed -- this is not a pure reorder")

    cell["source"] = new_list
    new_src = "".join(new_list)

    # ------------------------------------------------------- proof (a) diff
    print("\n--- (a) line diff vs original -------------------------------------")
    diff = [l for l in difflib.unified_diff(src_list, new_list, lineterm="", n=0)
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    for l in diff:
        print("   ", l.rstrip()[:160])
    moved = len(diff) // 2
    if len(diff) != 4:
        die(f"expected 4 diff lines (2 moved statements), got {len(diff)}")
    print(f"    => {moved} statements moved, {len(new_list) - moved} lines untouched")

    print("\nnew stage order in filter_output_graph:")
    for ln, name in stage_order(new_src):
        print(f"  line {ln}: {name}")

    # -------------------------------------------------------- proof (b) AST
    print("\n--- (b) AST verification ------------------------------------------")
    old_tree, new_tree = ast.parse(old_src), ast.parse(new_src)
    old_top = [ast.dump(n) for n in old_tree.body]
    new_top = [ast.dump(n) for n in new_tree.body]
    if len(old_top) != len(new_top):
        die("top-level statement count changed")
    outer_diff = [i for i, (a, b) in enumerate(zip(old_top, new_top)) if a != b]
    print(f"    module has {len(old_top)} top-level statements; "
          f"{len(outer_diff)} differ: {outer_diff}")

    old_body, new_body = func_body(old_src), func_body(new_src)
    if len(old_body) != len(new_body):
        die("filter_output_graph statement count changed")
    if sorted(old_body) != sorted(new_body):
        die("filter_output_graph statements are not a permutation -- content changed")
    perm = [old_body.index(s) for s in new_body]
    print(f"    {FUNC} has {len(old_body)} top-level statements")
    print(f"    new order as indices into the old order: {perm}")
    if perm == sorted(perm):
        die("nothing actually moved")

    # every other top-level statement must be byte-identical
    for i in outer_diff:
        node = new_tree.body[i]
        if not (isinstance(node, ast.FunctionDef) and node.name == FUNC):
            die(f"a statement outside {FUNC} changed: top-level index {i}")
    print(f"    the only changed top-level statement is {FUNC}: PASS")

    print()
    show_sequence(old_body, "statement sequence BEFORE (stage calls only):")
    show_sequence(new_body, "statement sequence AFTER (stage calls only):")

    # ---------------------------------------------------- proof (c) compile
    print("\n--- (c) compile ----------------------------------------------------")
    compile(new_src, "<cell2>", "exec")
    print("    compile(cell 2) OK")

    # ------------------------------------------------- proof (d) drift guard
    print("\n--- (d) configuration drift guard ---------------------------------")
    guard_line = src_list[only(src_list, f"{GUARD_KEY} = ", "drift guard")]
    guarded = ast.literal_eval(guard_line.split("=", 1)[1].strip())
    print(f"    {GUARD_KEY} asserts {len(guarded)} env vars:")
    for k, v in guarded.items():
        print(f"      {k} == {v}")
    touched = [k for k in guarded if any(k in l for l in [x for x in diff])]
    if touched:
        die(f"the reorder touches guarded keys {touched}; update the guard too")
    if guard_line not in new_list:
        die("guard line changed")
    print("    the diff sets no os.environ value and touches no guarded key: PASS")
    print("    (the moved statements are two calls/prints inside a function; the "
          "guard runs at import time on os.environ only)")

    # --------------------------------------- (f) the s05 change, if building s08
    if spec["env"]:
        print("\n--- (f) adding the s05 change on top of the reorder ---------------")
        k = only(new_list, ENV_ANCHOR, "gap2-recovery env line")
        final_list = new_list[:k + 1] + [ENV_LINE] + new_list[k + 1:]
        if len(final_list) != len(new_list) + 1:
            die("env insertion changed more than one line")
        if sorted(final_list) != sorted(new_list + [ENV_LINE]):
            die("env insertion perturbed an existing line")
        cell["source"] = final_list
        final_src = "".join(final_list)
        compile(final_src, "<cell2>", "exec")

        # exactly one added top-level statement, and it is that os.environ set
        f_top = [ast.dump(n) for n in ast.parse(final_src).body]
        if len(f_top) != len(new_top) + 1:
            die(f"top-level statement count moved by {len(f_top) - len(new_top)}, expected +1")
        added = [i for i in range(len(f_top)) if f_top[i] not in new_top]
        node = ast.parse(final_src).body[added[0]] if len(added) == 1 else None
        if node is None or ast.unparse(node) != ENV_LINE.strip():
            die(f"the added statement is not the expected env set: {added}")
        print(f"    inserted after line {k + 1} ({ENV_ANCHOR})")
        print(f"    one added top-level statement at index {added[0]}: "
              f"{ast.unparse(node)}")
        if sorted(func_body(final_src)) != sorted(old_body):
            die("filter_output_graph changed when the env line went in")
        print(f"    {FUNC} still a pure permutation of the base: PASS")
        new_list = final_list

    # ------------------------------------------------------------- write out
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    after_lines = [l for c in nb["cells"] for l in c.get("source", [])]
    grew = 1 if spec["env"] else 0
    if len(after_lines) != len(before_lines) + grew:
        die(f"whole-notebook line count moved by "
            f"{len(after_lines) - len(before_lines)}, expected +{grew}")
    whole = [l for l in difflib.unified_diff(before_lines, after_lines,
                                             lineterm="", n=0)
             if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    if len(whole) != 4 + grew:
        die(f"whole-notebook diff shows {len(whole)} lines, expected {4 + grew}")

    # ------- (g) one change relative to s05, demonstrated not asserted -------
    if spec["env"]:
        print("\n--- (g) diff against the SHIPPED s05 notebook ---------------------")
        if not S05.exists():
            die(f"{S05} not found; cannot prove one-change-vs-s05")
        s05_lines = [l for c in json.loads(S05.read_text())["cells"]
                     for l in c.get("source", [])]
        vs05 = [l for l in difflib.unified_diff(s05_lines, after_lines,
                                                lineterm="", n=0)
                if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
        for l in vs05:
            print("   ", l.rstrip()[:160])
        if len(vs05) != 4:
            die(f"s08 differs from s05 by {len(vs05)} lines, expected 4 "
                "(the two moved statements)")
        if sorted(s05_lines) != sorted(after_lines):
            die("s08 is not a pure reorder of s05")
        print("    s08 == s05 with two statements moved, nothing else: PASS")
        print("    => ONE change relative to the configuration that was submitted")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.ipynb").write_text(json.dumps(nb))
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"aryaarun07/{slug}", "title": title, "code_file": f"{slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1",
                            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
                            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1"],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [], "model_sources": []}, indent=2))
    print(f"\nwrote {out_dir}/{slug}.ipynb  "
          f"({len(after_lines)} source lines)")
    print(f"wrote {out_dir}/kernel-metadata.json")

    simulate(old_src)


# --------------------------------------------------------------------------
# (e) behavioural proof: run the notebook's OWN two stage functions, under the
# notebook's OWN env config, in both orders on two toy graphs.
# --------------------------------------------------------------------------
NEED = {"edge_distance_um", "node_point", "_position_um",
        "_single_successor_map", "_single_predecessor_map", "_next_node_id",
        "recover_strict_gap2", "add_safe_divisions_postlink"}
CONST_PREFIXES = ("GAP2_", "SAFE_DIV_", "OUTPUT_GAP2_RECOVERY",
                  "OUTPUT_SAFE_DIVISIONS", "DEEPCENTER_SAFE_DIV_",
                  "VOXEL_SCALE_UM")
XS = 0.40625  # um per voxel in x; VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def _load_stages(src):
    """Exec just the stage functions + their constants, verbatim from cell 2."""
    import math
    import os as _os

    import numpy as np
    from scipy.spatial import cKDTree

    ns = {"os": _os, "np": np, "cKDTree": cKDTree, "math": math, "sys": sys}
    for node in ast.parse(src).body:
        keep = False
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.value, ast.Constant)
                and isinstance(node.targets[0], ast.Subscript)
                and ast.unparse(node.targets[0]).startswith("os.environ[")):
            keep = True                      # the notebook's own configuration
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id.startswith(CONST_PREFIXES)):
            keep = True                      # constants derived from it
        elif isinstance(node, ast.FunctionDef) and node.name in NEED:
            keep = True
        if keep:
            exec(compile(ast.Module([node], []), "<cell2>", "exec"), ns)
    # the two image-backed helpers are stubbed: no movie on disk here, and the
    # deepcenter veto is a separate axis that this test deliberately isolates
    ns["refine_synthetic_midpoint"] = lambda ds, t, mid, cache, stats: mid
    ns["deepcenter_accept_repair_point"] = lambda *a, **k: True
    if NEED - set(ns):
        die(f"could not load {NEED - set(ns)}")
    return ns


def _n(i, t, x_um):
    return {"node_id": i, "t": t, "z": 0.0, "y": 0.0, "x": x_um / XS}


def _e(s, d):
    return {"source_id": s, "target_id": d, "edge_prob": 0.9, "distance_um": 0.0}


def _contested():
    """Orphan Q(7) at t=4 is BOTH safe_div's only candidate daughter for the
    parent S(3) at t=3 AND gap2's only candidate start for the dead end E(2)
    at t=1. Everything else is on a fully linked track, so Q is the only
    orphan in the graph."""
    nodes = {n["node_id"]: n for n in [
        _n(11, 0, 0.0), _n(12, 1, 0.0), _n(6, 2, 0.0),   # main track head
        _n(3, 3, 0.0),                                    # S: parent, 1 child
        _n(4, 4, -3.0), _n(5, 5, -5.0),                   # C, Cg
        _n(1, 0, -9.0), _n(2, 1, -6.0),                   # E0 -> E, dead end
        _n(7, 4, 3.0), _n(8, 5, 5.0),                     # Q (orphan), Qg
    ]}
    return nodes, [_e(11, 12), _e(12, 6), _e(6, 3), _e(3, 4), _e(4, 5),
                   _e(1, 2), _e(7, 8)]


def _uncontested():
    """The same gap2 pair, with no parent at t=3 for safe_div to fork."""
    nodes = {n["node_id"]: n for n in
             [_n(1, 0, -9.0), _n(2, 1, -6.0), _n(7, 4, 3.0), _n(8, 5, 5.0)]}
    return nodes, [_e(1, 2), _e(7, 8)]


def simulate(src):
    ns = _load_stages(src)
    gap2, sd = ns["recover_strict_gap2"], ns["add_safe_divisions_postlink"]
    keys = ("gap2_candidates", "gap2_pairs_selected", "gap2_added_nodes",
            "gap2_added_edges", "gap2_skipped_cap", "safe_division_candidates",
            "safe_division_geometric_candidates", "safe_divisions_added",
            "safe_division_skipped_cap", "safe_division_mutual_nn_rejected",
            "safe_division_divergence_rejected",
            "safe_division_symmetry_rejected")

    def run(make, order):
        nodes, edges = make()
        st = dict.fromkeys(keys, 0)
        if order == "gap2_first":
            nodes, edges = gap2(nodes, edges, st, dataset="toy")
            edges = sd(nodes, edges, st, dataset="toy")
        else:
            edges = sd(nodes, edges, st, dataset="toy")
            nodes, edges = gap2(nodes, edges, st, dataset="toy")
        div = sorted((int(e["source_id"]), int(e["target_id"]))
                     for e in edges if e.get("safe_division"))
        new = sorted(set(nodes) - set(make()[0]))
        return st, div, new

    print("\n--- (e) both orders, on the notebook's own stage functions --------")
    print(f"    config: OUTPUT_GAP2_RECOVERY={ns['OUTPUT_GAP2_RECOVERY']} "
          f"OUTPUT_SAFE_DIVISIONS={ns['OUTPUT_SAFE_DIVISIONS']} "
          f"SAFE_DIV_MAX_UM={ns['SAFE_DIV_MAX_UM']} "
          f"GAP2_MAX_TOTAL_UM={ns['GAP2_MAX_TOTAL_UM']}")
    results = {}
    for name, make in (("CONTESTED", _contested), ("UNCONTESTED", _uncontested)):
        print(f"    {name}:")
        for order in ("gap2_first", "safediv_first"):
            st, div, new = run(make, order)
            results[name, order] = (st["gap2_pairs_selected"],
                                    st["safe_divisions_added"], tuple(new))
            print(f"      {order:14s} gap2_pairs={st['gap2_pairs_selected']} "
                  f"gap2_nodes={st['gap2_added_nodes']} new_ids={new} "
                  f"safe_divisions_added={st['safe_divisions_added']} "
                  f"division_edges={div}")

    # gap2 first eats the daughter: the division is lost
    assert results["CONTESTED", "gap2_first"][:2] == (1, 0), results
    # safe_div first keeps the division; gap2 gives up only that one pair
    assert results["CONTESTED", "safediv_first"][:2] == (0, 1), results
    # where nothing is contested, the reorder changes nothing at all --
    # including the synthetic node ids, because safe_div adds edges, not nodes
    assert (results["UNCONTESTED", "gap2_first"]
            == results["UNCONTESTED", "safediv_first"]), results
    print("    PASS: contested orphan goes to safe_div under the new order; "
          "uncontested gap2 pairs are bit-identical in both orders")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "s08")
