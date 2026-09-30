r"""Pre-ILP candidate audit -- does the ILP DECLINE real divisions, or were they never proposed?

WHY THIS EXISTS
---------------
Root cause established 2026-08-18: the deployed pipeline sets `BIOHUB_ILP_APPEARANCE_WEIGHT=0.0`
(vendor default 0.1). Under the minimised min-cost-flow objective, turning a link into a division
costs `division_weight - appearance_weight - p = 1.0 - p >= 0` for every `p <= 1`, so division is
strictly dominated and the solver always prefers "daughter appears from nothing". The post-ILP
graph has out-degree {1: N} everywhere; every fork we emit comes from the post-hoc
`add_safe_divisions_postlink` patch.

That tells us the solver never OUTPUTS a division. It does NOT tell us whether the solver was ever
OFFERED one. Two worlds:
  (a) the candidate graph contains second-daughter edges above the 0.5 threshold and the ILP
      declined them on cost  -> fixing the weights (lane L1) recovers real divisions;
  (b) the candidate graph never contained them -> the second daughter never scored above 0.5,
      L1 is dead, and the lane collapses to the threshold (L2) or the training fix.

Kernel `biohub-p4-preilp-loeo-f1` exports the candidate graph handed to the solver, one line
before `solver.solve`. This script reads that roll-up and decides between (a) and (b).

KILL CRITERION, stated before the run: of the 31 fold-1 GT divisions whose second daughter was
detected but left ORPHAN by the linker, fewer than ~15 having a declined candidate edge means the
ILP is not discarding real divisions -> lane L1 is DEAD.

Note the measurement is a COUNT, deliberately: our score instrument is not validated
(LOEO +0.0144/+0.0090 -> LB +0.000) and fold noise floors are +/-0.003 to +/-0.004, so no score
delta at this scale would be interpretable. See internal-reports/instrument_repair_2026-08-18.md.

Usage:
  .venv\Scripts\python.exe scripts\core\kaggle_factory.py fetch \
      --spec scripts\kaggle_specs\p4_preilp_loeo_f1.json --files preilp_split1.parquet \
      --dest c:\temp\preilp_f1
  .venv\Scripts\python.exe scripts\win_bet\preilp_div_audit.py \
      --parquet c:\temp\preilp_f1\preilp_split1.parquet --gt-dir data\train
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])   # z, y, x um per level-0 voxel
MATCH_UM = 7.0                                 # scorer matching radius
KILL_BAR = 15                                  # of 31 fold-1 orphan cases


def gt_divisions(geff_path: Path):
    """Return [(mother_id, d1_id, d2_id)] plus {node_id: (t, xyz_um)}."""
    g = zarr.open(str(geff_path), mode="r")
    nid = np.asarray(g["nodes/ids"][:])
    t = np.asarray(g["nodes/props/t/values"][:])
    z = np.asarray(g["nodes/props/z/values"][:])
    y = np.asarray(g["nodes/props/y/values"][:])
    x = np.asarray(g["nodes/props/x/values"][:])
    pos = {int(n): (int(t[i]), np.array([z[i], y[i], x[i]]) * SCALE) for i, n in enumerate(nid)}
    e = np.asarray(g["edges/ids"][:])
    out = collections.defaultdict(list)
    if e.ndim == 2 and len(e):
        for a, b in e:
            out[int(a)].append(int(b))
    divs = []
    for src, tg in out.items():
        if len(tg) >= 2 and src in pos:
            d = [q for q in tg[:2] if q in pos]
            if len(d) == 2:
                divs.append((src, d[0], d[1]))
    return divs, pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="preilp_split*.parquet from the kernel")
    ap.add_argument("--gt-dir", default="data/train")
    args = ap.parse_args()

    df = pl.read_parquet(args.parquet)
    nodes = df.filter(pl.col("row_type") == "node")
    edges = df.filter(pl.col("row_type") == "edge")
    print(f"pre-ILP candidate graph: {nodes.height:,} nodes, {edges.height:,} candidate edges, "
          f"{df['dataset'].n_unique()} crops")

    probs = edges["edge_prob"].drop_nulls().to_numpy()
    if probs.size:
        print(f"  edge_prob: min {probs.min():.6f}  median {np.median(probs):.4f}  max {probs.max():.4f}")

    tot_ge2 = tot_div = 0
    matched_mother = declined_at_true = 0
    declined_probs: list[float] = []

    for ds, ge in edges.group_by("dataset"):
        ds = ds[0] if isinstance(ds, tuple) else ds
        gn = nodes.filter(pl.col("dataset") == ds)
        npos = {int(r[0]): np.array([r[2], r[3], r[4]]) * SCALE
                for r in gn.select(["node_id", "t", "z", "y", "x"]).iter_rows()}
        nt = {int(r[0]): int(r[1]) for r in gn.select(["node_id", "t"]).iter_rows()}

        out = collections.defaultdict(list)
        for s, t_, p in ge.select(["source_id", "target_id", "edge_prob"]).iter_rows():
            out[int(s)].append((int(t_), p))
        ge2 = {s: v for s, v in out.items() if len(v) >= 2}
        tot_ge2 += len(ge2)

        geff = Path(args.gt_dir) / f"{ds}.geff"
        if not geff.exists():
            continue
        divs, gpos = gt_divisions(geff)
        tot_div += len(divs)
        if not divs or not npos:
            continue

        by_t = collections.defaultdict(list)
        for n_, tt in nt.items():
            by_t[tt].append(n_)
        trees = {tt: (cKDTree(np.stack([npos[i] for i in ids])), ids)
                 for tt, ids in by_t.items() if ids}

        def match(tt, xyz):
            ent = trees.get(tt)
            if ent is None:
                return None
            tree, ids = ent
            d, j = tree.query(xyz)
            return ids[j] if d <= MATCH_UM else None

        for src, d1, d2 in divs:
            mt, mxyz = gpos[src]
            m = match(mt, mxyz)
            if m is None or m not in ge2:
                continue
            matched_mother += 1
            # do the >=2 candidates of this matched mother include BOTH true daughters?
            t1, p1 = gpos[d1]
            t2, p2 = gpos[d2]
            a, b = match(t1, p1), match(t2, p2)
            if a is None or b is None:
                continue
            tgts = {t_ for t_, _ in ge2[m]}
            if a in tgts and b in tgts:
                declined_at_true += 1
                declined_probs += [p for t_, p in ge2[m] if t_ in (a, b) and p is not None]

    print("\n=== PRE-ILP CANDIDATE AUDIT ===")
    print(f"  sources with candidate out-degree >= 2 : {tot_ge2:,}")
    print(f"  GT division events in these crops      : {tot_div}")
    print(f"  ... whose mother matched a >=2 source  : {matched_mother}")
    print(f"  ... with BOTH true daughters offered   : {declined_at_true}   <-- the number")
    if declined_probs:
        dp = np.array(declined_probs)
        print(f"  declined candidate edge_prob: median {np.median(dp):.4f}  min {dp.min():.4f}")

    print(f"\n  KILL BAR: < {KILL_BAR} of the 31 fold-1 orphan cases => lane L1 DEAD")
    if declined_at_true < KILL_BAR:
        print(f"  VERDICT: {declined_at_true} < {KILL_BAR}  ->  **L1 DEAD**. The ILP is not discarding")
        print("           real divisions; the second daughter never cleared the 0.5 candidate")
        print("           threshold. Escalate to L2 (threshold) or the training fix.")
    else:
        print(f"  VERDICT: {declined_at_true} >= {KILL_BAR}  ->  **L1 LIVE**. The solver was offered")
        print("           real divisions and declined them on cost. Changing")
        print("           BIOHUB_ILP_DIVISION_WEIGHT (0.55) or restoring appearance_weight=0.1")
        print("           should recover them -- but L3 is still required, because")
        print("           motion_relink_edges replaces the ILP's edges wholesale.")


if __name__ == "__main__":
    main()
