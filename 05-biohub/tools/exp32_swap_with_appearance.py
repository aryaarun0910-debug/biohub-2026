#!/usr/bin/env python3
"""EXP-32 -- does APPEARANCE lift swap precision past the thin margin?

EXP-31 on all 199 graphs: the swap discriminator holds 49-52% (logistic) and 50-54% (grad-boost)
at top-1000 against a 48.1% break-even. Real signal, AUC ~0.81, but the min-across-embryos margin
is nil: -0.0003 at 49%, exactly 0.0000 at 50%. Only higher precision changes the arithmetic --
fixing the 2,887-edge pool at 65% would be worth +0.0128.

Appearance is the untested route. EXP-26 found the DeepCenter score the strongest single feature
(0.657) where geometry sat at 0.52-0.58, and the swap model uses geometry only.

Runs LOCALLY on CPU against our own image volumes: no Kaggle GPU slot (capped at 2, both busy)
and no credential on a third-party VM. Scores are taken at BOTH the assigned target and the
rival's, because the swap question is comparative.
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch, zarr, tracksdata as td
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
PACK = Path("data/pubweights/biohub-deepcenter-unet3d-center-prior-v1")
sys.path.insert(0, str(PACK / "source_scripts"))
import div_sweep as D
from train_full_frame_center_detector import DeepCenterUNet3D

GT = Path("data/train_geff"); GRAPHS = Path("work/train_graphs"); IMG = Path("data/images/train")
TOL = 7.0
GEO = ["dist", "rival_dist", "rival_margin", "n_rivals", "src_outdeg", "disp_ratio"]
APP = ["app_tgt", "app_rival_tgt", "app_src", "app_margin"]


def load_model():
    ck = torch.load(PACK / "weights/full_frame_center/best.pt", map_location="cpu",
                    weights_only=False)
    cfg = ck["config"] if isinstance(ck["config"], dict) else vars(ck["config"])
    m = DeepCenterUNet3D(base_channels=int(cfg["base_channels"])); m.load_state_dict(ck["model_state"])
    m.eval(); return m, cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", type=int, default=40)
    a = ap.parse_args()
    model, cfg = load_model(); pf = int(cfg.get("pool_factor", 4))
    X, y, emb = [], [], []
    # BALANCE THE EMBRYOS. sorted() puts all 71 44b6 first, so a plain [:N] slice is one embryo
    # and leave-one-embryo-out becomes impossible. This bug has now appeared three times in this
    # campaign (EXP-18, EXP-26, here); interleave instead of slicing.
    avail = [p for p in sorted(GRAPHS.rglob("*.geff"))
             if (GT / f"{p.stem}.geff").exists() and (IMG / f"{p.stem}.zarr").exists()]
    A = [p for p in avail if p.stem.startswith("44b6")]
    B = [p for p in avail if p.stem.startswith("6bba")]
    half = a.datasets // 2
    files = A[:half] + B[:a.datasets - half]
    print(f"  balanced: {len([f for f in files if f.stem.startswith('44b6')])} x 44b6, "
          f"{len([f for f in files if f.stem.startswith('6bba')])} x 6bba")
    print(f"  {len(files)} datasets, DeepCenter on cpu\n", flush=True)

    def score_at(hm, node, pf):
        z = int(round(node["z"])); yy = int(round(node["y"]/pf)); xx = int(round(node["x"]/pf))
        z0,z1 = max(0,z-1), min(hm.shape[0], z+2)
        y0,y1 = max(0,yy-1), min(hm.shape[1], yy+2)
        x0,x1 = max(0,xx-1), min(hm.shape[2], xx+2)
        p = hm[z0:z1, y0:y1, x0:x1]
        return float(p.max()) if p.size else 0.0

    for fi, p in enumerate(files):
        t0 = time.time()
        g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g,tuple) else g
        n = g.node_attrs(); e = g.edge_attrs()
        nodes, pos, tt = {}, {}, {}
        for r in n.iter_rows(named=True):
            i = int(r["node_id"])
            nodes[i] = {"z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])}
            pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM; tt[i] = int(r["t"])
        pred, inbound, outbound = [], {}, {}
        for r in e.iter_rows(named=True):
            s_, d_ = int(r["source_id"]), int(r["target_id"])
            pred.append((s_,d_)); inbound[d_] = s_; outbound.setdefault(s_,[]).append(d_)
        gg = td.graph.IndexedRXGraph.from_geff(GT/f"{p.stem}.geff"); gg = gg[0] if isinstance(gg,tuple) else gg
        gpos, gtt = {}, {}
        for r in gg.node_attrs().iter_rows(named=True):
            i=int(r["node_id"]); gpos[i]=np.array([r["z"],r["y"],r["x"]])*D.VOXEL_SCALE_UM; gtt[i]=int(r["t"])
        gedges = [(int(r["source_id"]),int(r["target_id"])) for r in gg.edge_attrs().iter_rows(named=True)]
        if not gedges: continue
        by_t = {}
        for i,t in tt.items(): by_t.setdefault(t,[]).append(i)
        trees = {t:(cKDTree(np.stack([pos[i] for i in ids])), ids) for t,ids in by_t.items()}
        def match(pt,t):
            ent=trees.get(t)
            if ent is None: return None
            tr_,ids=ent; d,j=tr_.query(pt); return ids[int(j)] if d<=TOL else None
        gt_target_of = {}
        for s,d in gedges:
            if s in gpos and d in gpos:
                ms,md = match(gpos[s],gtt[s]), match(gpos[d],gtt[d])
                if ms is not None and md is not None: gt_target_of.setdefault(ms,set()).add(md)
        if not gt_target_of: continue

        need = sorted({tt[d_] for s_,d_ in pred if s_ in gt_target_of} |
                      {tt[s_] for s_,d_ in pred if s_ in gt_target_of})
        z = zarr.open(str(IMG/f"{p.stem}.zarr"), mode="r"); arr = z["0"] if "0" in z else z
        heat = {}
        for t in need:
            if t >= arr.shape[0]: continue
            v = np.asarray(arr[t], np.float32)
            zz,yy,xx = v.shape
            pl = v.reshape(zz, yy//pf, pf, xx//pf, pf).max(axis=(2,4))
            lo,hi = np.percentile(pl, cfg.get("norm_lo_pct",50.0)), np.percentile(pl, cfg.get("norm_hi_pct",99.5))
            im = np.clip((pl-lo)/max(hi-lo,1e-6), cfg.get("norm_clip_lo",-0.5), cfg.get("norm_clip_hi",6.0)).astype(np.float32)
            with torch.no_grad():
                heat[t] = torch.sigmoid(model(torch.from_numpy(im[None,None]))[0,0]).numpy()

        got = 0
        for s_, d_ in pred:
            if s_ not in gt_target_of: continue
            t = tt[s_]; ent = trees.get(t)
            if ent is None or t+1 not in heat or t not in heat: continue
            tr_, ids = ent
            rivals = [ids[j] for j in tr_.query_ball_point(pos[d_], 10.0) if ids[j]!=s_ and tt[ids[j]]==t]
            if not rivals: continue
            dd = float(np.linalg.norm(pos[d_]-pos[s_]))
            rv = min(rivals, key=lambda r: np.linalg.norm(pos[d_]-pos[r]))
            rd = float(np.linalg.norm(pos[d_]-pos[rv]))
            rt = outbound.get(rv, [None])[0]
            a_t = score_at(heat[t+1], nodes[d_], pf)
            a_rt = score_at(heat[t+1], nodes[rt], pf) if rt is not None and tt.get(rt)==t+1 else 0.0
            a_s = score_at(heat[t], nodes[s_], pf)
            X.append([dd, rd, dd-rd, len(rivals), len(outbound.get(s_,[])), dd/max(rd,1e-6),
                      a_t, a_rt, a_s, a_t-a_rt])
            y.append(0 if d_ in gt_target_of[s_] else 1); emb.append(p.stem[:4]); got += 1
        print(f"    {fi+1}/{len(files)} {p.stem}: {got} contested, {sum(y[-got:]) if got else 0} wrong, {time.time()-t0:.0f}s", flush=True)

    X, y, emb = np.array(X), np.array(y), np.array(emb)
    print(f"\n  contested {len(y):,}  wrong {y.sum():,}  base {y.mean():.2%}\n")
    names = GEO + APP
    print("  per-feature AUC:")
    for i,f in enumerate(names):
        a_ = roc_auc_score(y, X[:,i]); print(f"    {f:<14} {max(a_,1-a_):.3f}{'   <-- appearance' if f in APP else ''}")
    print("\n  === GEOMETRY ONLY vs GEOMETRY + APPEARANCE, leave-one-embryo-out ===")
    for tag, cols in (("geometry", list(range(len(GEO)))), ("geo+appearance", list(range(len(names))))):
        for held in sorted(set(emb)):
            tr, te = emb!=held, emb==held
            if te.sum()<50 or len(set(y[tr]))<2 or len(set(y[te]))<2: continue
            Xtr, Xte = X[tr][:,cols], X[te][:,cols]
            mu,sd = Xtr.mean(0), Xtr.std(0)+1e-9
            m = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0).fit((Xtr-mu)/sd, y[tr])
            s = m.predict_proba((Xte-mu)/sd)[:,1]
            o = np.argsort(-s); yy = y[te][o]
            line = f"    {tag:<15} held-out {held}: AUC {roc_auc_score(y[te],s):.3f}  "
            for K in (200, 1000):
                if K<=len(yy):
                    pr=yy[:K].mean(); line += f"top{K} {pr:.0%}{'*' if pr>0.481 else ''}  "
            print(line)
    print("    (* clears the 48.1% edge break-even)")


if __name__ == "__main__":
    main()
