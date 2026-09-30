"""G5: null-equivalence test, then measure consequence-scored forks on LOEO.

Test 1 -- with the fork disabled the augmented solver must return a
BYTE-IDENTICAL edge list. Not similar. Identical.
Test 3 -- accept only if dJ_edge + 0.1*dJ_div > 0 AND dJ_edge >= 0.
"""
import sys, json, hashlib, itertools, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd
from biohub import metric as MT, division as DV

G = Path("artifacts/graphs")
split = json.loads(Path("artifacts/loeo_split.json").read_text())
fold_of = {f: e for e, d in split["folds"].items() for f in d["held_out_films"]}
films = sorted(p.stem for p in G.glob("*.npz"))


def load(f):
    z = np.load(G / f"{f}.npz")
    ids = z["gt_ids"]; idx = {int(i): k for k, i in enumerate(ids)}
    gt_e = np.array([[idx[int(a)], idx[int(b)]] for a, b in z["gt_edges"]], np.int64).reshape(-1, 2)
    return (z["det_t"], z["det_zyx"].astype(np.float32), z["gt_t"], z["gt_grid"],
            float(z["est_nodes"]), [(int(a), int(b)) for a, b, *_ in z["linked_edges"]], gt_e)

DATA = {f: load(f) for f in films}
ehash = lambda e: hashlib.sha256(repr(sorted(e)).encode()).hexdigest()

# ---------------- G5 TEST 1: null equivalence ----------------
print("=== G5 test 1: null equivalence (fork disabled must be byte-identical) ===")
bad = 0
for f in films:
    dt, dz, *_rest, base, _ = DATA[f]
    out, added = DV.augment(base, dt, dz, DV.DivConfig(enabled=False))
    if ehash(out) != ehash(base) or added:
        bad += 1
print(f"  films checked {len(films)} | mismatches {bad} -> {'PASS' if bad == 0 else 'FAIL'}\n")
if bad:
    sys.exit("G5 test 1 FAILED -- no fork solver. (ABORT_RULES G5)")


def evaluate(cfg, label):
    rows = {}
    for f in films:
        dt, dz, gt_t, gt_grid, n_est, base, gt_e = DATA[f]
        e, _ = DV.augment(base, dt, dz, cfg)
        rows[f] = MT.score_film(list(range(len(dt))), e, dz, dt, gt_grid, gt_t, gt_e, n_est)
    out = {}
    for fold in ["ALL"] + sorted(set(fold_of.values())):
        sel = [rows[f] for f in films if fold == "ALL" or fold_of[f] == fold]
        out[fold] = MT.aggregate(sel)
    return label, out


base_lbl, BASE = evaluate(DV.DivConfig(enabled=False), "baseline")
b = BASE["ALL"]
print(f"baseline  ALL  score {b['score']:.4f}  adjJ {b['adj_J_edge']:.4f}  "
      f"divJ {b['div_jaccard']:.4f}  div {b['div_tp']}/{b['div_fp']}/{b['div_fn']}\n")

print("=== consequence sweep: how long must both daughters persist? ===")
print(f"{'min_track_len':>13} {'score':>8} {'d_score':>9} {'adjJ':>8} {'d_adjJ':>9} "
      f"{'divJ':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'added':>7}")
res = []
for L in (1, 2, 4, 6, 8, 12, 16, 24):
    cfg = DV.DivConfig(min_track_len=L)
    _, R = evaluate(cfg, f"L{L}")
    a = R["ALL"]
    n_add = sum(len(DV.augment(DATA[f][5], DATA[f][0], DATA[f][1], cfg)[1]) for f in films)
    d_s, d_a = a["score"] - b["score"], a["adj_J_edge"] - b["adj_J_edge"]
    flag = "" if (d_s > 0 and d_a >= -1e-6) else "  <- fails G5 test 3"
    print(f"{L:>13} {a['score']:>8.4f} {d_s:>+9.4f} {a['adj_J_edge']:>8.4f} {d_a:>+9.4f} "
          f"{a['div_jaccard']:>7.4f} {a['div_tp']:>4} {a['div_fp']:>4} {a['div_fn']:>4} "
          f"{n_add:>7}{flag}")
    res.append({"min_track_len": L, **{k: a[k] for k in
                ("score", "adj_J_edge", "div_jaccard", "div_tp", "div_fp", "div_fn")},
                "d_score": d_s, "d_adjJ": d_a, "added": n_add,
                **{f"{fd}_score": R[fd]["score"] for fd in R}})
pd.DataFrame(res).to_csv("artifacts/division_sweep.csv", index=False)

best = max(res, key=lambda r: r["score"])
print(f"\nbest min_track_len={best['min_track_len']}: score {best['score']:.4f} "
      f"({best['d_score']:+.4f}), divJ {best['div_jaccard']:.4f}, "
      f"edge {best['d_adjJ']:+.5f}")
print(f"  per fold: " + "  ".join(f"{k.replace('_score','')}={v:.4f}"
      for k, v in best.items() if k.endswith('_score') and k != 'd_score'))
