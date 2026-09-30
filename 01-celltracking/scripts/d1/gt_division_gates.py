"""Which of the three LIVE safe-division gates actually binds on true divisions?

Pure GT geometry -- no wrapper run, no GPU. This is the cheap falsification gate that must be
passed BEFORE any sister-gate work is funded.

The three live gates in `add_safe_divisions_postlink` (deployment values from cell02):

    SAFE_DIV_MAX_UM                parent -> NEW daughter
    SAFE_DIV_EXISTING_CHILD_MAX_UM parent -> EXISTING child
    SAFE_DIV_SISTER_MAX_UM         daughter <-> daughter

Their VALUES are read from the operational base at import time via
`scripts/core/baseline_contract.py` and are deliberately not written here - the numbers this
docstring used to quote were `p3_harmonic`'s, not the champion's.

NOTE the gate that correction 6 killed is a DIFFERENT constant: `DIV_SISTER_MAX_UM = 8.0`
inside `OUTPUT_DIVISION_GEOMETRY_FILTER`, which defaults "0" and is never enabled. Designing
against that one would repeat correction 6.
"""
import argparse
import glob
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(r"c:\Users\aryaa\Documents\Biohub-CellTracking-2026")
sys.path.insert(0, str(REPO / "src"))

SCALE = (1.625, 0.40625, 0.40625)
# THE THREE LIVE GATES, READ FROM THE OPERATIONAL BASE - not copied.
# These were 4.66 / 7.65 / 8.50, which is `p3_harmonic`'s geometry and NOT the champion's
# (CLAUDE.md names p3_harmonic as explicitly not the champion). The same stale triple was
# independently hard-coded in constant_audit.py and tests/test_lineage_degree_invariants.py.
# Three copies, three chances to go stale, all three taken. baseline_contract regenerates
# from the built notebook and has a --check drift lock, so there is now one copy and it moves
# when the notebook moves.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "core"))
import baseline_contract as _BC  # noqa: E402

_GEOM = _BC.safe_division()
G_PARENT_NEW = _GEOM["BIOHUB_SAFE_DIV_MAX_UM"]
G_PARENT_EXISTING = _GEOM["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"]
G_SISTER = _GEOM["BIOHUB_SAFE_DIV_SISTER_MAX_UM"]


def main():
    import tracksdata as td
    from biotrack.metric import load_graph

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()

    geffs = sorted(glob.glob(str(REPO / "data" / "train" / "*.geff")))[: a.limit]
    print(f"GT crops: {len(geffs)}\n", flush=True)

    sister, d_near, d_far, fams = [], [], [], []
    n_mothers = n_nodes = 0
    for gi, gp in enumerate(geffs, 1):
        name = pathlib.Path(gp).stem
        g = load_graph(gp)
        na = g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
        pos = {}
        for iid, z, y, x in zip(na[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
                                na["z"].to_list(), na["y"].to_list(), na["x"].to_list()):
            pos[iid] = np.array([float(z) * SCALE[0], float(y) * SCALE[1], float(x) * SCALE[2]])
        n_nodes += len(pos)
        if g.num_edges() == 0:
            continue
        ea = g.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                                     td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
        kids = {}
        for s, t in zip(ea[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(),
                        ea[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()):
            kids.setdefault(s, []).append(t)
        for m, ch in kids.items():
            if len(ch) != 2:
                continue
            n_mothers += 1
            a1, b1 = pos[ch[0]], pos[ch[1]]
            p = pos[m]
            da, db = float(np.linalg.norm(a1 - p)), float(np.linalg.norm(b1 - p))
            sister.append(float(np.linalg.norm(a1 - b1)))
            d_near.append(min(da, db))
            d_far.append(max(da, db))
            fams.append(name.split("_")[0])
        if gi % 10 == 0:
            print(f"  {gi}/{len(geffs)}", flush=True)

    sister = np.array(sister); d_near = np.array(d_near); d_far = np.array(d_far)
    fams = np.array(fams)
    n = len(sister)
    print("\n" + "=" * 78)
    print(f"GT DIVISION GEOMETRY — {n} true divisions over {n_nodes} GT nodes")
    print("=" * 78)

    def pct(v, q):
        return float(np.percentile(v, q))

    print(f"\n  {'quantity':26s} {'p50':>8s} {'p90':>8s} {'p95':>8s} {'p99':>8s} {'max':>9s}")
    for lbl, v in (("sister (daughter<->daughter)", sister),
                   ("parent->nearer daughter", d_near),
                   ("parent->farther daughter", d_far)):
        print(f"  {lbl:26s} {pct(v,50):8.3f} {pct(v,90):8.3f} {pct(v,95):8.3f} "
              f"{pct(v,99):8.3f} {v.max():9.3f}")

    print("\n  WHICH GATE EXCLUDES TRUE DIVISIONS? (share of GT divisions rejected)")
    ex_sister = sister > G_SISTER
    # our proposer pairs an EXISTING child with a NEW candidate; the new one must clear 4.66
    # and the existing one 7.65. Best case: assign the nearer daughter as 'new'.
    ex_parent_best = np.minimum(d_near, d_far) > G_PARENT_NEW
    ex_exist_best = np.maximum(d_near, d_far) > G_PARENT_EXISTING
    print(f"    sister      > {G_SISTER:4.2f} um : {ex_sister.sum():5d}  ({100*ex_sister.mean():6.2f}%)")
    print(f"    nearer kid  > {G_PARENT_NEW:4.2f} um : {ex_parent_best.sum():5d}  "
          f"({100*ex_parent_best.mean():6.2f}%)   <- SAFE_DIV_MAX_UM")
    print(f"    farther kid > {G_PARENT_EXISTING:4.2f} um : {ex_exist_best.sum():5d}  "
          f"({100*ex_exist_best.mean():6.2f}%)   <- SAFE_DIV_EXISTING_CHILD_MAX_UM")
    any_ex = ex_sister | ex_parent_best | ex_exist_best
    print(f"    ANY gate rejects        : {any_ex.sum():5d}  ({100*any_ex.mean():6.2f}%)")
    print(f"    ALL gates pass          : {(~any_ex).sum():5d}  ({100*(~any_ex).mean():6.2f}%)")

    print("\n  MARGINAL VALUE OF RELAXING THE SISTER GATE ALONE")
    print("  (true divisions unlocked that ALSO clear both parent gates)")
    parent_ok = ~ex_parent_best & ~ex_exist_best
    base = (parent_ok & ~ex_sister).sum()
    print(f"    {'sister gate':>12s} {'admissible':>11s} {'vs 8.50':>9s}")
    for gate in (7.2, 8.0, 8.5, 9.0, 10.0, 12.0, 1e9):
        adm = int((parent_ok & (sister <= gate)).sum())
        lbl = "inf" if gate > 1e8 else f"{gate:.2f}"
        print(f"    {lbl:>12s} {adm:11d} {adm-base:+9d}")

    print("\n  SAME, BY FAMILY")
    for fam in sorted(set(fams.tolist())):
        m = fams == fam
        po = parent_ok & m
        b = int((po & (sister <= G_SISTER)).sum())
        inf = int(po.sum())
        print(f"    {fam}: divisions={int(m.sum()):5d}  parent-gates-pass={int(po.sum()):5d}  "
              f"sister<=8.5={b:5d}  headroom if sister=inf: {inf-b:+d}")


if __name__ == "__main__":
    main()
