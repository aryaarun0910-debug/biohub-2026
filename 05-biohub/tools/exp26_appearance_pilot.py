#!/usr/bin/env python3
"""EXP-26 -- does APPEARANCE separate true from false divisions, where geometry cannot?

EXP-24 trained a discriminator on nine geometric features over 1.28M candidates and failed:
AUC 0.566-0.622, precision 0.0% at top-10 against a 24% break-even. But it had no appearance
term, while the 0.947 pipeline's discriminating power comes from exactly that -- the DeepCenter
heatmap score it uses as a veto.

This is the cheap version of the decisive test. Rather than score 1.28M candidates (19,900 frames
of inference, ~16h on CPU), it takes a handful of division-bearing datasets and asks whether the
DeepCenter score at a candidate point separates true divisions from false ones AT ALL.

If appearance does not move AUC here, it will not at scale, and divisions are closed.
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch, zarr, tracksdata as td
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
PACK = Path("data/pubweights/biohub-deepcenter-unet3d-center-prior-v1")
sys.path.insert(0, str(PACK / "source_scripts"))
import div_sweep as D
from train_full_frame_center_detector import DeepCenterUNet3D

GT = Path("data/train_geff"); GRAPHS = Path("work/train_graphs"); IMG = Path("data/images/train")
TOL = 7.0
FEATS = ["parent_dist", "sister_dist", "child_dist", "cos", "diverge",
         "arc_max", "arc_min", "arc_asym", "arc_sum"]
WIDE = D.DivCfg(max_um=12.0, sister_max_um=20.0, existing_child_max_um=12.0,
                symmetry_tau=0.0, diverge_um=-99.0, require_divergence=False,
                require_mutual_nn=True, frame_frac_cap=1.0, global_frac_cap=1.0)


def load_model():
    ck = torch.load(PACK / "weights/full_frame_center/best.pt", map_location="cpu",
                    weights_only=False)
    cfg = ck["config"]
    base = int(cfg["base_channels"] if isinstance(cfg, dict) else getattr(cfg, "base_channels", 24))
    m = DeepCenterUNet3D(base_channels=base)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m, (cfg if isinstance(cfg, dict) else vars(cfg))


def pool_xy(v, f):
    z, y, x = v.shape
    return v.reshape(z, y // f, f, x // f, f).max(axis=(2, 4))


def normalise(v, cfg):
    lo = np.percentile(v, cfg.get("norm_lo_pct", 50.0))
    hi = np.percentile(v, cfg.get("norm_hi_pct", 99.5))
    out = (v - lo) / max(hi - lo, 1e-6)
    return np.clip(out, cfg.get("norm_clip_lo", -0.5), cfg.get("norm_clip_hi", 6.0)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", type=int, default=8)
    a = ap.parse_args()
    model, cfg = load_model()
    pf = int(cfg.get("pool_factor", 4))
    print(f"  DeepCenter loaded: base_channels {cfg.get('base_channels')}, pool_factor {pf}\n",
          flush=True)

    # pick division-bearing datasets: the question is only meaningful where positives exist
    cands = []
    for p in sorted(GRAPHS.rglob("*.geff")):
        gp = GT / f"{p.stem}.geff"
        if not gp.exists() or not (IMG / f"{p.stem}.zarr").exists():
            continue
        g = td.graph.IndexedRXGraph.from_geff(gp); g = g[0] if isinstance(g, tuple) else g
        out = {}
        for r in g.edge_attrs().iter_rows(named=True):
            out[r["source_id"]] = out.get(r["source_id"], 0) + 1
        nd = sum(1 for v in out.values() if v >= 2)
        if nd:
            cands.append((nd, p))
    cands.sort(reverse=True)
    picked = [p for _, p in cands[:a.datasets]]
    print(f"  {len(cands)} division-bearing datasets; using the {len(picked)} richest\n", flush=True)

    X, y, app = [], [], []
    for p in picked:
        t0 = time.time()
        gp = GT / f"{p.stem}.geff"
        g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
        n = g.node_attrs(); e = g.edge_attrs()
        nodes = {int(r["node_id"]): {"t": int(r["t"]), "z": float(r["z"]),
                                     "y": float(r["y"]), "x": float(r["x"])}
                 for r in n.iter_rows(named=True)}
        edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"]),
                  "edge_prob": r.get("edge_prob"), "distance_um": 0.0}
                 for r in e.iter_rows(named=True)]
        gg = td.graph.IndexedRXGraph.from_geff(gp); gg = gg[0] if isinstance(gg, tuple) else gg
        gpos = {int(r["node_id"]): np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
                for r in gg.node_attrs().iter_rows(named=True)}
        gout = {}
        for r in gg.edge_attrs().iter_rows(named=True):
            gout.setdefault(int(r["source_id"]), []).append(int(r["target_id"]))
        gtd = [(gpos[s], gpos[k[0]], gpos[k[1]]) for s, k in gout.items()
               if len(k) == 2 and s in gpos and k[0] in gpos and k[1] in gpos]

        props = D.collect_proposals(nodes, edges, WIDE)
        need = sorted({pr[3] + 1 for pr in props})          # candidate lives at frame t+1
        z = zarr.open(str(IMG / f"{p.stem}.zarr"), mode="r"); arr = z["0"] if "0" in z else z
        heat = {}
        for t in need:
            if t >= arr.shape[0]:
                continue
            v = normalise(pool_xy(np.asarray(arr[t], np.float32), pf), cfg)
            with torch.no_grad():
                heat[t] = torch.sigmoid(model(torch.from_numpy(v[None, None]))[0, 0]).numpy()

        first_child = {}
        for e_ in edges:
            first_child.setdefault(int(e_["source_id"]), int(e_["target_id"]))
        got = 0
        for f, sid, qid, t in props:
            hm = heat.get(t + 1)
            if hm is None:
                continue
            q = nodes[qid]
            zz = int(round(q["z"])); yy = int(round(q["y"] / pf)); xx = int(round(q["x"] / pf))
            z0, z1 = max(0, zz - 1), min(hm.shape[0], zz + 2)
            y0, y1 = max(0, yy - 1), min(hm.shape[1], yy + 2)
            x0, x1 = max(0, xx - 1), min(hm.shape[2], xx + 2)
            patch = hm[z0:z1, y0:y1, x0:x1]
            if patch.size == 0:
                continue
            src = D._pos(nodes[sid]); cand = D._pos(nodes[qid])
            kc = first_child.get(sid)
            if kc is None:
                continue
            kid = D._pos(nodes[kc])
            lab = 0
            for P, A, B in gtd:
                if np.linalg.norm(src - P) > TOL:
                    continue
                if ((np.linalg.norm(kid - A) <= TOL and np.linalg.norm(cand - B) <= TOL) or
                        (np.linalg.norm(kid - B) <= TOL and np.linalg.norm(cand - A) <= TOL)):
                    lab = 1; break
            X.append([f[k] for k in FEATS]); y.append(lab); app.append(float(patch.max())); got += 1
        print(f"    {p.stem}: {got:,} scored candidates, {sum(y[-got:])} true, "
              f"{len(need)} frames, {time.time()-t0:.0f}s", flush=True)

    X, y, app = np.array(X), np.array(y), np.array(app)
    print(f"\n  TOTAL {len(y):,} candidates, {y.sum()} true (base rate {y.mean():.4%})\n")
    if y.sum() < 3:
        print("  too few positives to judge"); return
    print("  per-feature AUC (higher = more separable):")
    for i, f in enumerate(FEATS):
        a_ = roc_auc_score(y, X[:, i]); print(f"    {f:<14} {max(a_, 1-a_):.3f}")
    a_app = roc_auc_score(y, app)
    print(f"    {'DEEPCENTER':<14} {max(a_app, 1-a_app):.3f}   <-- the appearance term")
    print(f"\n  appearance score: true divisions median {np.median(app[y==1]):.4f}, "
          f"false median {np.median(app[y==0]):.4f}")
    order = np.argsort(-app)
    for K in (10, 25, 50, 100):
        if K <= len(y):
            print(f"    top-{K:<4} by appearance alone: precision {y[order][:K].mean():>6.1%}"
                  f"{'  <-- PAYS' if y[order][:K].mean() > 0.24 else ''}")


if __name__ == "__main__":
    main()
