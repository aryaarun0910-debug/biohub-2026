"""Clean 2x2: ordering x contest, at the ORIGINAL tight gate settings.

The previous sweep bundled three changes and is uninterpretable. Here only
ordering and contest vary; every gate is ported verbatim from the config that
produced divJ 0.067.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from biohub import metric as MT, postprocess as PP, division2 as D2

G = Path("artifacts/graphs"); films = sorted(p.stem for p in G.glob("*.npz"))
split = json.loads(Path("artifacts/loeo_split.json").read_text())
fold_of = {f: e for e, d in split["folds"].items() for f in d["held_out_films"]}
D = {}
for f in films:
    z = np.load(G / f"{f}.npz"); ids = z["gt_ids"]
    idx = {int(i): k for k, i in enumerate(ids)}; le = z["linked_edges"]
    D[f] = dict(det_t=z["det_t"], det_zyx=z["det_zyx"].astype(np.float32), gt_t=z["gt_t"],
                gt_grid=z["gt_grid"], n_est=float(z["est_nodes"]),
                edges=[(int(a), int(b)) for a, b, *_ in le],
                meta={(int(a), int(b)): (float(p), float(dd)) for a, b, p, dd in le},
                gt_e=np.array([[idx[int(a)], idx[int(b)]] for a, b in z["gt_edges"]],
                              np.int64).reshape(-1, 2))

TIGHT = dict(parent_max_um=9.0, sister_max_um=14.0, existing_child_max_um=10.0,
             symmetry_tau=0.6, min_track_len=8, diverge_um=2.25, n_candidates=1,
             frame_frac_cap=0.0076, global_frac_cap=0.00375)


def run(cfg, fork_first):
    rows, na = [], 0
    for f in films:
        d = D[f]; e, dt, dz = d["edges"], d["det_t"], d["det_zyx"]
        if fork_first:
            e, a, _ = D2.augment(e, dt, dz, cfg, d["meta"]); na += len(a)
            e, dt, dz, _, _ = PP.prune_and_filter(e, dt, dz, min_track_len=6, keep_forks=True)
        else:
            e, dt, dz, _, _ = PP.prune_and_filter(e, dt, dz, min_track_len=6, keep_forks=True)
            e, a, _ = D2.augment(e, dt, dz, cfg, d["meta"]); na += len(a)
        rows.append(MT.score_film(list(range(len(dt))), e, dz, dt,
                                  d["gt_grid"], d["gt_t"], d["gt_e"], d["n_est"]))
    a = MT.aggregate(rows)
    per = {fd: MT.aggregate([rows[i] for i, f in enumerate(films) if fold_of[f] == fd])
           for fd in sorted(set(fold_of.values()))}
    return a, per, na


b, _, _ = run(D2.DivConfig2(enabled=False, **TIGHT), False)
print(f"baseline  score {b['score']:.4f}  adjJ {b['adj_J_edge']:.4f}  divJ {b['div_jaccard']:.4f}\n")
print(f"{'ordering':<13}{'contest':<9}{'score':>8}{'d':>9}{'d_adjJ':>9}{'divJ':>8}"
      f"{'TP':>4}{'FP':>5}{'FN':>5}{'add':>7}{'44b6':>8}{'6bba':>8}")
for ff, lbl in ((False, "prune-first"), (True, "FORK-FIRST")):
    for ct in (False, True):
        a, per, na = run(D2.DivConfig2(contest=ct, **TIGHT), ff)
        ds, da = a["score"] - b["score"], a["adj_J_edge"] - b["adj_J_edge"]
        ok = ds > 0 and da >= -1e-6
        print(f"{lbl:<13}{str(ct):<9}{a['score']:>8.4f}{ds:>+9.4f}{da:>+9.4f}"
              f"{a['div_jaccard']:>8.4f}{a['div_tp']:>4}{a['div_fp']:>5}{a['div_fn']:>5}"
              f"{na:>7}{per['44b6']['score']:>8.4f}{per['6bba']['score']:>8.4f}"
              f"{'' if ok else '  FAIL'}", flush=True)
