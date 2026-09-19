"""Third opinion on s10's linefit weight, across all 199 films.

s10 ships OUTPUT_LINEFIT_WEIGHT 0.6, chosen because it is positive on BOTH the 4
scored films (+0.00471) and the 8 validator films (+0.00338). Twelve films is
still twelve films, and s09 is the standing reminder of what a single film set
can do.

artifacts/graphs/*.npz has all 199 with ground truth. linefit() is coordinate
-system agnostic -- it fits a line per axis in whatever space it is handed and
blends -- so it applies unchanged to these isotropic grid coordinates, where
metric.py (NOT metric2) is the correct scorer.

Applied to raw_edges, i.e. the relink-OFF arm, which is the s05/s10 base. Only
linefit varies; nothing else in the chain is applied, which ISOLATES the knob
rather than approximating the chain with ports that do not exist for this grid.

CALIBRATION FIRST, as in scripts/105: the 4 scored films are in this set and
their deployed-graph answer is known from scripts/106-107 -- w=0.6 is +0.00471
and w=0.3 is -0.00086. If the rebuild reproduces that ordering, its verdict on
the other 195 is worth having. If it does not, this script is noise and says so.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from biohub import metric as MT

exec((ROOT / "scripts/91_other_stages.py").read_text()
     .split("def linefit")[1].join(["def linefit", ""]).split("\n\n\n")[0])

GRAPHS = ROOT / "artifacts/graphs"
SCORED = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
WS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def one(path):
    d = np.load(path, allow_pickle=True)
    dt = d["det_t"].astype(np.int64)
    dz = d["det_zyx"].astype(np.float64)
    nodes = list(range(len(dt)))
    gt_t, gt_grid = d["gt_t"].astype(np.int64), d["gt_grid"].astype(np.float64)
    idx = {int(i): k for k, i in enumerate(d["gt_ids"])}
    gt_e = np.array([[idx[int(a)], idx[int(b)]] for a, b in d["gt_edges"]],
                    np.int64).reshape(-1, 2)
    n_est = float(d["est_nodes"])
    edges = [(int(a), int(b)) for a, b in d["raw_edges"][:, :2].astype(np.int64)]
    out = {}
    for w in WS:
        G = dict(t=dt, zyx=dz, edges=edges)
        if w > 0:
            G, _ = linefit(G, w=w, window=2)
        out[w] = MT.score_film(nodes, G["edges"], G["zyx"], dt, gt_grid, gt_t,
                               gt_e, n_est)
    return path.stem, out


def main():
    paths = sorted(GRAPHS.glob("*.npz"))
    print(__doc__.split("CALIBRATION FIRST")[0].rstrip())
    print("=" * 92)
    print(f"199 films x {len(WS)} weights on 16 workers ...")
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = dict(ex.map(one, paths, chunksize=2))
    print(f"done: {len(res)} films\n")

    def prox(r):
        return r["adj_J_edge"] + 0.1 * (r["div_tp"] /
                                        max(r["div_tp"] + r["div_fp"] + r["div_fn"], 1))

    print("--- CALIBRATION: the 4 SCORED films, deployed-graph truth known ---")
    print("    deployed (scripts/106-107): w=0.6 -> +0.00471,  w=0.3 -> -0.00086")
    print(f"{'film':<18}{'w=0.3 d':>11}{'w=0.6 d':>11}   reproduces ordering?")
    ok = 0
    for s in SCORED:
        r = res[s]
        b = prox(r[0.8])
        d3, d6 = prox(r[0.3]) - b, prox(r[0.6]) - b
        good = d6 > d3
        ok += good
        print(f"{s:<18}{d3:>+11.5f}{d6:>+11.5f}   {'yes' if good else 'NO'}")
    print(f"    rebuild puts 0.6 above 0.3 on {ok}/4 scored films")

    print(f"\n--- ALL {len(res)} FILMS: delta vs deployed w=0.8 ---")
    print(f"{'w':>5}{'mean':>11}{'median':>11}{'films>0':>10}{'weighted agg':>15}")
    best = None
    for w in WS:
        ds = np.array([prox(res[s][w]) - prox(res[s][0.8]) for s in res])
        agg_w = MT.aggregate([res[s][w] for s in res])
        agg_b = MT.aggregate([res[s][0.8] for s in res])
        delta = agg_w["adj_J_edge"] - agg_b["adj_J_edge"]
        if best is None or delta > best[1]:
            best = (w, delta)
        print(f"{w:>5.1f}{ds.mean():>+11.5f}{np.median(ds):>+11.5f}"
              f"{f'{int((ds > 0).sum())}/{len(ds)}':>10}{delta:>+15.5f}")
    print(f"\n199-film optimum: w={best[0]:g}  ({best[1]:+.5f} weighted)")
    print(f"s10 ships w=0.6.  s09 shipped w=0.3.")

    out = ROOT / "artifacts/linefit_across_199.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["film"] + [f"w{w:g}" for w in WS])
        for s in sorted(res):
            wr.writerow([s] + [f"{prox(res[s][w]):.6f}" for w in WS])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
