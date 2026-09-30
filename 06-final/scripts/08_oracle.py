"""G3: oracle headroom decomposition. Built to kill the division thesis.

Baseline is the LINKED graph -- the one-to-one Hungarian that the deployed
pipeline actually ships, which by construction has no forks.

  O1  division ceiling   force the correct fork wherever parent and both
                         daughters are already matched; drop wrong forks
  O2  association ceiling force every GT edge whose endpoints are already
                         matched (perfect linking, current detections)
  O3  detection ceiling  insert a node at every unmatched GT node, then O2
  O4  node-count         set n_pred = n_est (measured, never tuned)

Read O1 to three decimals at most: fold 44b6 has D=26, one event = 0.0038.
O2 rests on ~129k labelled edges and is far better resolved.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd
from biohub import metric as MT

G = Path("artifacts/graphs")
split = json.loads(Path("artifacts/loeo_split.json").read_text())


def load(film):
    z = np.load(G / f"{film}.npz")
    gt_ids = z["gt_ids"]
    idx = {int(i): k for k, i in enumerate(gt_ids)}
    gt_e = np.array([[idx[int(a)], idx[int(b)]] for a, b in z["gt_edges"]
                     if int(a) in idx and int(b) in idx], dtype=np.int64).reshape(-1, 2)
    return z, gt_e


def force(edges, add, drop_in=True):
    """Add edges, removing any conflicting incoming edge on the targets."""
    tgt = {int(b) for _, b in add}
    keep = [(int(s), int(t)) for s, t in edges if not (drop_in and int(t) in tgt)]
    return keep + [(int(a), int(b)) for a, b in add]


rows = {k: [] for k in ("base", "O1", "O2", "O3", "O4")}
reach = {"edges_total": 0, "edges_both_matched": 0, "nodes_total": 0, "nodes_matched": 0}
films = sorted(p.stem for p in G.glob("*.npz"))
t0 = time.perf_counter()

for n, film in enumerate(films):
    z, gt_e = load(film)
    det_t, det_zyx = z["det_t"], z["det_zyx"].astype(np.float32)
    gt_t, gt_grid, n_est = z["gt_t"], z["gt_grid"], float(z["est_nodes"])
    nodes = list(range(len(det_t)))
    base_e = [(int(a), int(b)) for a, b, *_ in z["linked_edges"]]

    p2g, g2p = MT.match_nodes(gt_grid, gt_t, det_zyx, det_t)
    reach["nodes_total"] += len(gt_t); reach["nodes_matched"] += len(g2p)
    both = sum(1 for a, b in gt_e if a in g2p and b in g2p)
    reach["edges_total"] += len(gt_e); reach["edges_both_matched"] += both

    S = lambda e, npred=None: MT.score_film(nodes, e, det_zyx, det_t, gt_grid, gt_t,
                                            gt_e, n_est, npred)
    rows["base"].append(S(base_e))

    # O1 -- perfect divisions, everything else fixed
    gt_out = {}
    for a, b in gt_e:
        gt_out.setdefault(int(a), []).append(int(b))
    add = []
    for g, ch in gt_out.items():
        if len(ch) >= 2 and g in g2p and all(c in g2p for c in ch[:2]):
            add += [(g2p[g], g2p[c]) for c in ch[:2]]
    rows["O1"].append(S(force(base_e, add)))

    # O2 -- perfect association given current detections
    add2 = [(g2p[int(a)], g2p[int(b)]) for a, b in gt_e if int(a) in g2p and int(b) in g2p]
    rows["O2"].append(S(force([], add2)))

    # O3 -- perfect detection then perfect association
    extra = [k for k in range(len(gt_t)) if k not in g2p]
    d_t = np.concatenate([det_t, gt_t[extra].astype(det_t.dtype)])
    d_z = np.concatenate([det_zyx, gt_grid[extra]])
    g2p3 = dict(g2p); g2p3.update({k: len(det_t) + i for i, k in enumerate(extra)})
    add3 = [(g2p3[int(a)], g2p3[int(b)]) for a, b in gt_e]
    rows["O3"].append(MT.score_film(list(range(len(d_t))), force([], add3), d_z, d_t,
                                    gt_grid, gt_t, gt_e, n_est))

    # O4 -- node-count term neutralised
    r = dict(rows["base"][-1]); r["multiplier"] = 1.0
    r["adj_J_edge"] = r["J_edge"]; rows["O4"].append(r)

    if (n + 1) % 25 == 0:
        print(f"  {n+1}/{len(films)}  {time.perf_counter()-t0:.0f}s", flush=True)

fold_of = {f: e for e, d in split["folds"].items() for f in d["held_out_films"]}
out = []
for fold in sorted(set(fold_of.values())) + ["ALL"]:
    sel = [i for i, f in enumerate(films) if fold == "ALL" or fold_of[f] == fold]
    agg = {k: MT.aggregate([rows[k][i] for i in sel]) for k in rows}
    b = agg["base"]
    for k in ("base", "O1", "O2", "O3", "O4"):
        a = agg[k]
        out.append({"fold": fold, "oracle": k, "films": len(sel),
                    "score": a["score"], "adj_J": a["adj_J_edge"], "J": a["J_edge"],
                    "mult": a["multiplier"], "divJ": a["div_jaccard"],
                    "div_tp": a["div_tp"], "div_fp": a["div_fp"], "div_fn": a["div_fn"],
                    "d_score": a["score"] - b["score"],
                    "d_adjJ": a["adj_J_edge"] - b["adj_J_edge"],
                    "d_div_w": 0.1 * (a["div_jaccard"] - b["div_jaccard"])})
df = pd.DataFrame(out)
df.to_csv("artifacts/oracle.csv", index=False)
pd.set_option("display.width", 200)
print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nreachability: GT nodes matched {reach['nodes_matched']}/{reach['nodes_total']}"
      f" = {reach['nodes_matched']/reach['nodes_total']:.4f}")
print(f"              GT edges with BOTH endpoints matched {reach['edges_both_matched']}"
      f"/{reach['edges_total']} = {reach['edges_both_matched']/reach['edges_total']:.4f}")
