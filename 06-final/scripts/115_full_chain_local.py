"""THE MISSING INSTRUMENT: the notebook's REAL post-processing chain, on the
REAL scored films, locally.

s05 scored 0.945 against a 0.947 baseline while every offline tier said +0.02
to +0.04. The diagnosis: no instrument ever combined the full chain with the
test films.

    in-kernel validator   FULL chain  |  TRAIN films   -> +0.0224
    our harness (ports)   PARTIAL     |  TEST films    -> +0.02707
    board                 FULL        |  TEST          -> -0.002

Our ports omit single-parent repair, single-CHILD repair, the short-track rescue
and the DeepCenter vetoes -- and those stages repair the same damage relink
causes, so they are partially redundant with removing it.

This runs `filter_output_graph` -- the notebook's own entry point for the whole
chain, extracted verbatim from cell 2 -- on the 4 test films' ILP graphs.

CALIBRATION IS THE POINT, not the absolute number. The board handed us one hard
fact: relink OFF is WORSE by about 0.002. An instrument that reproduces that
sign can be trusted. One that still says +0.027 means the chain was never the
problem and the GROUND TRUTH we score against is.
"""
import ast
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

NB = ROOT / "public notebooks/biohub-0-947-lb-runnable-with-public-datasets.ipynb"
SCORED = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
# NB: exec-ing scripts/91 into globals() clobbers a bare `PRED`, so name it
# distinctly. That bug cost a run.
TEST_PRED = ROOT / ("artifacts/s05_output/tracking_repo/predictions/unknown/"
                    "unet_transformer/split_0")


def load_cell2(relink: bool):
    """Definitions + constants from cell 2, with env set BEFORE exec.

    Module-level constants are read from os.environ at definition time, so the
    env must be in place first. Driver statements (If/For/With/Expr at module
    level) are skipped -- we want the functions, not the notebook's run.
    """
    src = "".join(json.loads(NB.read_text())["cells"][2]["source"])
    tree = ast.parse(src)
    keep = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef,
            ast.Assign, ast.AnnAssign)
    ns = {"__name__": "nb_cell2"}
    skipped = 0
    for node in tree.body:
        if not isinstance(node, keep):
            skipped += 1
            continue
        try:
            exec(compile(ast.Module([node], []), "<cell2>", "exec"), ns)
        except Exception:
            skipped += 1
        # apply our override the moment the notebook's env block has run
        if isinstance(node, ast.Assign) and "OUTPUT_GAP2_RECOVERY" in ast.unparse(node):
            os.environ["BIOHUB_OUTPUT_MOTION_RELINK"] = "1" if relink else "0"
    return ns, skipped


def to_nodes_edges(P):
    nodes = {i: {"node_id": int(i), "t": int(t), "z": float(z), "y": float(y),
                 "x": float(x)}
             for i, (t, (z, y, x)) in enumerate(zip(P["t"], P["zyx"]))}
    edges = [{"source_id": int(s), "target_id": int(d), "edge_prob": float(p),
              "distance_um": 0.0}
             for (s, d), p in zip(P["edges"], P["prob"])]
    return nodes, edges


def main():
    print(__doc__.split("CALIBRATION IS THE POINT")[0].rstrip())
    print("=" * 92)
    exec((ROOT / "scripts/91_other_stages.py").read_text().split("\nALL = []")[0],
         globals())
    out = {}
    for relink in (True, False):
        ns, skipped = load_cell2(relink)
        fog = ns.get("filter_output_graph")
        if fog is None:
            sys.exit("ABORT: filter_output_graph not defined after extraction")
        print(f"\n--- relink {'ON (base)' if relink else 'OFF (s05)'} --- "
              f"(OUTPUT_MOTION_RELINK={ns.get('OUTPUT_MOTION_RELINK')}, "
              f"{skipped} driver statements skipped)")
        rows = []
        for stem in SCORED:
            P = load_pred(TEST_PRED / f"{stem}.geff")
            GT = load_gt(stem)
            nodes, edges = to_nodes_edges(P)
            n2, e2, st = fog(nodes, edges, stem, None)
            ids = sorted(n2)
            idx = {k: i for i, k in enumerate(ids)}
            t = np.array([n2[k]["t"] for k in ids], np.int64)
            zyx = np.array([[n2[k]["z"], n2[k]["y"], n2[k]["x"]] for k in ids], float)
            ee = [(idx[int(e["source_id"])], idx[int(e["target_id"])]) for e in e2
                  if int(e["source_id"]) in idx and int(e["target_id"]) in idx]
            r = M2.score(t, zyx, ee, GT["t"], GT["zyx"], GT["edges"], GT["n_est"])
            rows.append(r)
            print(f"    {stem:<18} nodes {len(ids):>7}  edges {len(ee):>7}  "
                  f"J {r['J_edge']:.5f}  div {r['dtp']}/{r['dfp']}/{r['dfn']}")
        agg = M2.aggregate(rows)
        out[relink] = agg
        print(f"    AGGREGATE proxy {agg['proxy']:.5f}  J {agg['J']:.5f}  "
              f"divJ {agg['divJ']:.4f}")

    d = out[False]["proxy"] - out[True]["proxy"]
    print("\n" + "=" * 92)
    print(f"FULL CHAIN on the SCORED films: relink OFF - relink ON = {d:+.5f}")
    print(f"  board says                                            = -0.00200")
    print(f"  our PORTS said                                        = +0.02707")
    print(f"  in-kernel validator (train films) said                = +0.02240")
    print()
    if d < 0:
        print("  => SIGN REPRODUCED. The chain was the problem: the stages our ports")
        print("     omit are partially redundant with removing relink. This instrument")
        print("     can be calibrated and trusted.")
    else:
        print("  => SIGN NOT REPRODUCED. The full chain still says relink removal helps,")
        print("     so the chain is NOT the explanation -- suspect the GROUND TRUTH we")
        print("     score against differs from what Kaggle scores.")


if __name__ == "__main__":
    main()
