"""Fix 1 (fork BEFORE prune) + fix 2 (contested targets). Measured on 199 films."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd
from biohub import metric as MT, postprocess as PP, division2 as D2

G = Path("artifacts/graphs")
split = json.loads(Path("artifacts/loeo_split.json").read_text())
fold_of = {f: e for e, d in split["folds"].items() for f in d["held_out_films"]}
films = sorted(p.stem for p in G.glob("*.npz"))

D = {}
for f in films:
    z = np.load(G / f"{f}.npz")
    ids = z["gt_ids"]; idx = {int(i): k for k, i in enumerate(ids)}
    le = z["linked_edges"]
    D[f] = dict(det_t=z["det_t"], det_zyx=z["det_zyx"].astype(np.float32),
                gt_t=z["gt_t"], gt_grid=z["gt_grid"], n_est=float(z["est_nodes"]),
                edges=[(int(a), int(b)) for a, b, *_ in le],
                meta={(int(a), int(b)): (float(p), float(dd)) for a, b, p, dd in le},
                gt_e=np.array([[idx[int(a)], idx[int(b)]] for a, b in z["gt_edges"]],
                              np.int64).reshape(-1, 2))

# ---- G5 test 1 ----
h = lambda e: hashlib.sha256(repr(sorted(e)).encode()).hexdigest()
bad = sum(1 for f in films
          if h(D2.augment(D[f]["edges"], D[f]["det_t"], D[f]["det_zyx"],
                          D2.DivConfig2(enabled=False), D[f]["meta"])[0]) != h(D[f]["edges"]))
print(f"G5 test 1 (null equivalence): {len(films)} films, {bad} mismatches -> "
      f"{'PASS' if bad == 0 else 'FAIL'}\n")
if bad:
    sys.exit("FAIL")


def run(cfg, fork_first=True, min_len=6):
    rows, nadd, ndrop = [], 0, 0
    for f in films:
        d = D[f]
        e, dt, dz = d["edges"], d["det_t"], d["det_zyx"]
        if fork_first:
            e, a, dr = D2.augment(e, dt, dz, cfg, d["meta"]); nadd += len(a); ndrop += len(dr)
            e, dt, dz, _, _ = PP.prune_and_filter(e, dt, dz, min_track_len=min_len, keep_forks=True)
        else:
            e, dt, dz, _, _ = PP.prune_and_filter(e, dt, dz, min_track_len=min_len, keep_forks=True)
            e, a, dr = D2.augment(e, dt, dz, cfg, d["meta"]); nadd += len(a); ndrop += len(dr)
        rows.append(MT.score_film(list(range(len(dt))), e, dz, dt,
                                  d["gt_grid"], d["gt_t"], d["gt_e"], d["n_est"]))
    a = MT.aggregate(rows)
    per = {fd: MT.aggregate([rows[i] for i, f in enumerate(films) if fold_of[f] == fd])
           for fd in sorted(set(fold_of.values()))}
    return a, per, nadd, ndrop


OFF = D2.DivConfig2(enabled=False)
b, bper, _, _ = run(OFF)
print(f"baseline (prune only)        score {b['score']:.4f}  adjJ {b['adj_J_edge']:.4f}  "
      f"divJ {b['div_jaccard']:.4f}  div {b['div_tp']}/{b['div_fp']}/{b['div_fn']}\n")

cfgs = [
    ("prune-then-fork, no contest", D2.DivConfig2(contest=False), False),
    ("FORK-FIRST, no contest",      D2.DivConfig2(contest=False), True),
    ("FORK-FIRST + contest",        D2.DivConfig2(contest=True),  True),
    ("  + wider parent 14um",       D2.DivConfig2(contest=True, parent_max_um=14.0), True),
    ("  + k=5 candidates",          D2.DivConfig2(contest=True, n_candidates=5), True),
    ("  + cap 2%",                  D2.DivConfig2(contest=True, global_frac_cap=0.02), True),
    ("  + cap 0.5%",                D2.DivConfig2(contest=True, global_frac_cap=0.005), True),
    ("  + contest margin 0",        D2.DivConfig2(contest=True, contest_margin_um=0.0), True),
    ("  + min_track_len 8",         D2.DivConfig2(contest=True, min_track_len=8), True),
]
print(f"{'variant':<30} {'score':>8} {'d':>8} {'d_adjJ':>8} {'divJ':>7} {'TP':>4} {'FP':>4} "
      f"{'FN':>4} {'added':>6} {'drop':>6} {'44b6':>7} {'6bba':>7}")
res = []
for lbl, cfg, ff in cfgs:
    a, per, na, nd = run(cfg, fork_first=ff)
    ds, da = a["score"] - b["score"], a["adj_J_edge"] - b["adj_J_edge"]
    ok = ds > 0 and da >= -1e-6
    print(f"{lbl:<30} {a['score']:>8.4f} {ds:>+8.4f} {da:>+8.4f} {a['div_jaccard']:>7.4f} "
          f"{a['div_tp']:>4} {a['div_fp']:>4} {a['div_fn']:>4} {na:>6} {nd:>6} "
          f"{per['44b6']['score']:>7.4f} {per['6bba']['score']:>7.4f}{'' if ok else '  FAIL-G5.3'}")
    res.append(dict(variant=lbl, score=a["score"], d_score=ds, d_adjJ=da,
                    divJ=a["div_jaccard"], tp=a["div_tp"], fp=a["div_fp"], fn=a["div_fn"],
                    added=na, dropped=nd, **{f"{k}_score": v["score"] for k, v in per.items()}))
pd.DataFrame(res).to_csv("artifacts/division2_sweep.csv", index=False)
