"""Does the s05 result (remove motion relink) hold across all 199 films?

Every decision so far rests on 8 validator films or, since scripts/102-103, on
the 4 films the board actually scores. Four films and ~2,273 GT edges is a thin
base for the biggest change in the project.

artifacts/graphs/*.npz holds a rebuild of ALL 199 films that stores BOTH arms:

    raw_edges     greedy on softmax probs, forks allowed   -> relink OFF
    linked_edges  1:1 Hungarian, tight 6um then relaxed 10 -> relink ON

which is exactly the s05 contrast, on 199 films instead of 4.

TWO CAVEATS, both load-bearing:

 1. This is the weaker rebuild, not the deployed ILP -- greedy instead of a
    global ILP. By the transfer lesson (docs/HANDOFF.md section 6) its absolute
    numbers do not carry. Only the DIRECTION and its CONSISTENCY are claimed.
 2. These graphs live on the DOWNSAMPLED ISOTROPIC grid, so they need
    src/biohub/metric.py -- NOT metric2. Using metric2 here would be 4x wrong
    in y and x and would fail silently. This is the documented trap, applied in
    the opposite direction to usual.

The design that makes caveat 1 tolerable: the 4 scored films are IN this set and
we already know their true answer from scripts/103 (+0.02707 for relink off).
So the rebuild can be CALIBRATED against ground truth on those 4 before its
verdict on the other 195 is believed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from biohub import metric as MT

GRAPHS = ROOT / "artifacts/graphs"
SCORED = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
VALIDATOR = ["44b6_12dfb391", "44b6_267148e4", "44b6_2a2eff9f", "44b6_341df25f",
             "6bba_062c8d37", "6bba_07e24132", "6bba_085bf656", "6bba_09961292"]


def one(path):
    d = np.load(path, allow_pickle=True)
    dt, dz = d["det_t"].astype(np.int64), d["det_zyx"].astype(np.float64)
    nodes = list(range(len(dt)))
    gt_t, gt_grid = d["gt_t"].astype(np.int64), d["gt_grid"].astype(np.float64)
    # gt_edges stores NODE IDs, not indices into gt_t/gt_grid. score_film wants
    # indices. Getting this wrong yields J = 0.0 on every film, silently.
    idx = {int(i): k for k, i in enumerate(d["gt_ids"])}
    gt_e = np.array([[idx[int(a)], idx[int(b)]] for a, b in d["gt_edges"]],
                    np.int64).reshape(-1, 2)
    n_est = float(d["est_nodes"])
    out = {}
    for arm, key in (("off", "raw_edges"), ("on", "linked_edges")):
        e = [(int(a), int(b)) for a, b in d[key][:, :2].astype(np.int64)]
        out[arm] = MT.score_film(nodes, e, dz, dt, gt_grid, gt_t, gt_e, n_est)
    return path.stem, out


def main():
    paths = sorted(GRAPHS.glob("*.npz"))
    print(__doc__.split("The design that makes")[0].rstrip())
    print("=" * 96)
    print(f"scoring {len(paths)} films x 2 arms on 16 workers ...")
    res = {}
    with ProcessPoolExecutor(max_workers=16) as ex:
        for stem, out in ex.map(one, paths, chunksize=4):
            res[stem] = out
    print(f"done: {len(res)} films\n")

    def delta(stem):
        a, b = res[stem]["off"], res[stem]["on"]
        pa = a["adj_J_edge"] + 0.1 * 0.0
        pb = b["adj_J_edge"] + 0.1 * 0.0
        return pa - pb, a, b

    # ---- calibration on the 4 films whose TRUE answer we know ----------
    print("--- CALIBRATION: the 4 SCORED films, where scripts/103 measured the truth ---")
    print("    scripts/103 on deployed ILP graphs: relink OFF is +0.02707 aggregate,")
    print("    and positive on all four individually.")
    print(f"{'film':<18}{'rebuild dJ(off-on)':>20}{'off J':>9}{'on J':>9}  sign")
    agree = 0
    for s in SCORED:
        if s not in res:
            print(f"{s:<18}{'MISSING':>20}")
            continue
        d, a, b = delta(s)
        agree += d > 0
        print(f"{s:<18}{d:>+20.5f}{a['J_edge']:>9.5f}{b['J_edge']:>9.5f}"
              f"  {'agrees' if d > 0 else 'DISAGREES'}")
    print(f"    rebuild agrees with the deployed-graph direction on {agree}/4 scored films")

    # ---- the full 199 -------------------------------------------------
    rows = []
    for stem in sorted(res):
        d, a, b = delta(stem)
        rows.append((stem, stem.split("_")[0], d, a, b))
    ds = np.array([r[2] for r in rows])
    print(f"\n--- ALL {len(rows)} FILMS: proxy(relink OFF) - proxy(relink ON) ---")
    print(f"    films where removing relink HELPS : {int((ds > 0).sum())}/{len(ds)} "
          f"({100 * (ds > 0).mean():.1f}%)")
    print(f"    mean {ds.mean():+.5f}   median {np.median(ds):+.5f}   "
          f"std {ds.std():.5f}")
    print(f"    p05 {np.percentile(ds, 5):+.5f}   p95 {np.percentile(ds, 95):+.5f}")
    print(f"    worst {ds.min():+.5f} ({rows[int(ds.argmin())][0]})   "
          f"best {ds.max():+.5f} ({rows[int(ds.argmax())][0]})")

    for emb in ("44b6", "6bba"):
        sub = np.array([r[2] for r in rows if r[1] == emb])
        print(f"    {emb}: n={len(sub):>3}  helps {int((sub > 0).sum()):>3}"
              f" ({100 * (sub > 0).mean():>5.1f}%)  mean {sub.mean():+.5f}"
              f"  median {np.median(sub):+.5f}")

    agg_off = MT.aggregate([r[3] for r in rows])
    agg_on = MT.aggregate([r[4] for r in rows])
    print(f"\n    weighted aggregate over all {len(rows)} films:")
    print(f"      relink ON   adj_J {agg_on['adj_J_edge']:.5f}  J {agg_on['J_edge']:.5f}")
    print(f"      relink OFF  adj_J {agg_off['adj_J_edge']:.5f}  J {agg_off['J_edge']:.5f}")
    print(f"      delta       {agg_off['adj_J_edge'] - agg_on['adj_J_edge']:+.5f}")

    out = ROOT / "artifacts/relink_across_199.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["film", "embryo", "delta_off_minus_on", "off_J", "on_J",
                    "off_adj", "on_adj", "off_weight"])
        for stem, emb, d, a, b in rows:
            w.writerow([stem, emb, f"{d:.6f}", f"{a['J_edge']:.6f}",
                        f"{b['J_edge']:.6f}", f"{a['adj_J_edge']:.6f}",
                        f"{b['adj_J_edge']:.6f}", a["weight"]])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
