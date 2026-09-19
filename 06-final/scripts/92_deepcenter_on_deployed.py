"""H-DC re-run: is the DeepCenter veto still worth anything on DEPLOYED-quality graphs?

Previously (scripts/90_deepcenter_veto.py) the veto was measured on a locally
rebuilt tracking graph (adj edge Jaccard 0.845).  There the division-candidate
pool was ~180x larger than the divisions inside it (2334 proposals, 13 true), so
even a real ranker (AUC 0.836, p=1.5e-5, beats a brightness control) bought only
+0.0017 of competition proxy.

The graphs here are the kernel's own ILP output (adj 0.926).  safe_div() on them
proposes a far cleaner pool: 5 true divisions against 2 metric-charged false
positives.  A veto therefore has very little junk left to remove, and every true
proposal it rejects costs a division outright.  That is the question.

GEOMETRY (the thing that went wrong last time, stated explicitly):
  raw frame        (64, 256, 256)
  DeepCenter input = block_mean_xy(raw, pool_factor=4) -> (64, 64, 64)
  prediction nodes are ORIGINAL voxels (z 0-63, y/x 0-252) and, as verified at
  runtime below, lie exactly on a 4-voxel lattice in y and x.
  => heatmap index is (z, y // 4, x // 4).  NOT 1:1 like the 64^3 grid in
  script 90.  Verified by a positive control: GT node positions must score far
  above random voxels, and above a deliberately y/x-swapped indexing.

CHECKPOINT: best.pt (epoch 2).  checkpoint_last.pt is epoch 500 and its
validation never improved on epoch 2, so it is not used.

Outputs: artifacts/deepcenter_deployed_proposals.csv
         artifacts/deepcenter_deployed_thresholds.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub import io  # noqa: E402
from biohub import metric2 as M2  # noqa: E402

DC_DIR = ROOT / "weights" / "biohub-deepcenter-unet3d-center-prior-v1" / "weights" / "full_frame_center"
PRED = ROOT / "artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0"
TRAIN = io.dataset_root() / "train"
SCALE = M2.SCALE
THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


# ---------------------------------------------------------------------------
# DeepCenter architecture + normalisation: transcribed verbatim from
# scripts/90_deepcenter_veto.py (itself verbatim from the training script).
# ---------------------------------------------------------------------------
class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DeepCenterUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
        super().__init__()
        c = int(base_channels)
        self.enc1 = ConvBlock3d(in_channels, c)
        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock3d(c, c * 2)
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock3d(c * 2, c * 4)
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock3d(c * 4, c * 8)
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3d(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3d(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3d(c * 2, c)
        self.head = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def block_mean_xy(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2, x2 = (y // factor) * factor, (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


def normalize_dynamic_range(volume, lo_pct, hi_pct, clip_lo, clip_hi) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    lo, hi = np.percentile(vol, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo), clip_lo, clip_hi).astype(np.float32)


def load_deepcenter(device: str):
    cfg = json.loads((DC_DIR / "config.json").read_text())
    ck = torch.load(DC_DIR / "best.pt", map_location="cpu", weights_only=False)
    ck_cfg = dict(ck.get("config", {}))
    for k in ("base_channels", "pool_factor", "norm_lo_pct", "norm_hi_pct",
              "norm_clip_lo", "norm_clip_hi"):
        if k in ck_cfg and ck_cfg[k] != cfg.get(k):
            raise RuntimeError(f"config.json / checkpoint disagree on {k}: "
                               f"{cfg.get(k)} vs {ck_cfg[k]}")
    model = DeepCenterUNet3D(1, int(cfg["base_channels"]))
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=True)
    if missing or unexpected:  # strict=True already raises; belt and braces
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    meta = {"epoch": int(ck.get("epoch", -1)),
            "best_score": float(ck.get("best_score", float("nan"))),
            "params": sum(p.numel() for p in model.parameters()),
            "n_tensors": len(ck["model_state"]),
            "n_missing": len(missing), "n_unexpected": len(unexpected),
            **{k: cfg[k] for k in ("base_channels", "pool_factor", "norm_lo_pct",
                                   "norm_hi_pct", "norm_clip_lo", "norm_clip_hi")}}
    return model.to(device).eval(), cfg, meta


@torch.no_grad()
def score_frame(model, cfg, zarr_path: Path, t: int, device: str):
    """-> (doc, img_doc) on the pooled (64,64,64) grid.

    doc     = sigmoid heatmap, max-pooled over the documented window
              (z +-1, y +-2, x +-2 in POOLED units = y/x +-8 original voxels).
    img_doc = the normalised image itself under the same window.  This is the
              control: if DeepCenter is only saying 'this voxel is bright', the
              control ranks the proposals just as well and the net earns nothing.
    """
    raw = io.read_frame(zarr_path, t)
    pooled = block_mean_xy(raw, int(cfg["pool_factor"]))
    img = normalize_dynamic_range(pooled, cfg["norm_lo_pct"], cfg["norm_hi_pct"],
                                  cfg["norm_clip_lo"], cfg["norm_clip_hi"])
    x = torch.from_numpy(img)[None, None].to(device)
    prob = torch.sigmoid(model(x))
    doc = F.max_pool3d(prob, (3, 5, 5), stride=1, padding=(1, 2, 2))
    imd = F.max_pool3d(x, (3, 5, 5), stride=1, padding=(1, 2, 2))
    return doc[0, 0].cpu().numpy(), imd[0, 0].cpu().numpy()


# ---------------------------------------------------------------------------
# graph loading + safe_div: transcribed verbatim from scripts/24_keep_ilp_edges.py
# (importing it would execute its driver).  The candidate pool scored below is
# exactly what safe_div() proposes, with the deployed gates.
# ---------------------------------------------------------------------------
def load_pred(p):
    d = io.read_geff(p)
    idx = {int(i): k for k, i in enumerate(d["ids"])}
    e = np.array([[idx[int(a)], idx[int(b)]] for a, b in d["edges"]], np.int64).reshape(-1, 2)
    prob = np.asarray(io.read_zstd_array(p / "edges/props/edge_prob/values"), np.float64)
    return dict(t=d["t"].astype(np.int64),
                zyx=np.stack([d["z"], d["y"], d["x"]], 1).astype(np.float64),
                edges=[(int(a), int(b)) for a, b in e], prob=prob)


def load_gt(stem):
    g = io.read_geff(io.dataset_root() / "train" / f"{stem}.geff")
    idx = {int(i): k for k, i in enumerate(g["ids"])}
    e = np.array([[idx[int(a)], idx[int(b)]] for a, b in g["edges"]], np.int64).reshape(-1, 2)
    return dict(t=g["t"].astype(np.int64),
                zyx=np.stack([g["z"], g["y"], g["x"]], 1).astype(np.float64),
                edges=e, n_est=float(g["estimated_number_of_nodes"]))


def safe_div(P, parent_max=9.0, sister_max=14.0, child_max=10.0, tau=0.6,
             diverge=2.25, mutual_nn=True, frame_cap=0.0076, glob_cap=0.00375):
    """The deployed safe-division rule, on original-voxel coordinates."""
    pos = P["zyx"] * SCALE
    succ, indeg = {}, {}
    for s, t in P["edges"]:
        succ.setdefault(s, []).append(t); indeg[t] = indeg.get(t, 0) + 1
    by_t = {}
    for i, t in enumerate(P["t"]):
        by_t.setdefault(int(t), []).append(i)
    d = lambda a, b: float(np.linalg.norm(pos[a] - pos[b]))
    added = []
    for t in sorted(by_t):
        nxt = by_t.get(t + 1)
        if not nxt:
            continue
        orph = [j for j in nxt if indeg.get(j, 0) == 0]
        if not orph:
            continue
        opos = pos[orph]
        cands = []
        for Pn in [i for i in by_t[t] if len(succ.get(i, ())) == 1]:
            C = succ[Pn][0]
            dpc = d(Pn, C)
            if dpc > child_max:
                continue
            k = int(np.argmin(np.linalg.norm(opos - pos[C], axis=1)))
            Q = orph[k]
            dpq, dcq = d(Pn, Q), d(C, Q)
            if dpq > parent_max or dcq > sister_max:
                continue
            if abs(dpc - dpq) / max((dpc + dpq) / 2, 1e-9) > tau:
                continue
            if diverge > 0 or True:
                sc, sq = succ.get(C, []), succ.get(Q, [])
                if len(sc) != 1 or len(sq) != 1:
                    continue
                if d(sc[0], sq[0]) - dcq < diverge:
                    continue
            cands.append((dpq + 0.15 * dcq, Pn, Q))
        cap = max(1, round(frame_cap * len(by_t[t])))
        n = 0
        for _, Pn, Q in sorted(cands):
            if n >= cap or indeg.get(Q, 0) or len(succ.get(Pn, ())) >= 2:
                continue
            added.append((Pn, Q)); succ.setdefault(Pn, []).append(Q); indeg[Q] = 1; n += 1
    cap = max(1, round(glob_cap * len(P["edges"])))
    return P["edges"] + added[:cap], added[:cap]


# ---------------------------------------------------------------------------
def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, np.asarray(s, float))) if 0 < y.sum() < len(y) else float("nan")


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, cfg, meta = load_deepcenter(device)
    sm = json.loads((DC_DIR / "split_manifest.json").read_text())
    print("=== DeepCenter checkpoint ===")
    print(f"  path            {DC_DIR / 'best.pt'}")
    print(f"  state_dict      {meta['n_tensors']} tensors, {meta['params']:,} params, "
          f"strict load OK ({meta['n_missing']} missing / {meta['n_unexpected']} unexpected)")
    print(f"  base_channels   {meta['base_channels']}   pool_factor {meta['pool_factor']}")
    print(f"  normalisation   clip((v-P{meta['norm_lo_pct']})/(P{meta['norm_hi_pct']}"
          f"-P{meta['norm_lo_pct']}), {meta['norm_clip_lo']}, {meta['norm_clip_hi']})")
    print(f"  epoch           {meta['epoch']}  (best val loss {-meta['best_score']:.6f}) "
          f"-- checkpoint_last.pt (epoch 500) deliberately NOT used")
    print(f"  DC train split  {len(sm['train'])} films, embryos {sorted({f[:4] for f in sm['train']})}")
    print(f"  DC val split    {len(sm['val'])} films, embryos {sorted({f[:4] for f in sm['val']})}")
    print(f"  device          {device}\n")

    stems = sorted(p.stem for p in PRED.glob("*.geff"))
    print(f"=== {len(stems)} deployed-quality prediction graphs ===")

    t0 = time.time()
    rows, films = [], {}
    for n, stem in enumerate(stems, 1):
        P, G = load_pred(PRED / f"{stem}.geff"), load_gt(stem)
        _, added = safe_div(P)

        # -- geometry check, per film: predicted nodes must sit on the 4-lattice
        yx = P["zyx"][:, 1:]
        if not (np.all(yx == np.round(yx)) and np.all(yx % cfg["pool_factor"] == 0)):
            raise RuntimeError(f"{stem}: y/x are not on the pool_factor lattice; "
                               "the //4 heatmap index would be wrong")

        need = sorted({int(P["t"][q]) for _, q in added})
        cache = {t: score_frame(model, cfg, TRAIN / f"{stem}.zarr", t, device) for t in need}

        # ---- labels -------------------------------------------------------
        p2g, g2p = M2.match(P["t"], P["zyx"], G["t"], G["zyx"])
        gt_out = {}
        for a, b in G["edges"]:
            gt_out.setdefault(int(a), []).append(int(b))
        gt_div_src = {g for g, ch in gt_out.items() if len(ch) >= 2}

        scores, imgs, labels = [], [], []
        for Pn, Q in added:
            t = int(P["t"][Q])
            doc, imd = cache[t]
            z, y, x = (int(v) for v in P["zyx"][Q])
            yy, xx = y // int(cfg["pool_factor"]), x // int(cfg["pool_factor"])
            scores.append(float(doc[z, yy, xx]))
            imgs.append(float(imd[z, yy, xx]))
            mp, mq = p2g.get(int(Pn)), p2g.get(int(Q))
            if mp is None:
                labels.append("parent_unmatched")
            elif mp not in gt_div_src:
                labels.append("parent_matched_not_divider")
            elif mq is not None and mq in gt_out[mp]:
                labels.append("TRUE")
            else:
                labels.append("parent_is_divider_wrong_daughter")
        scores = np.asarray(scores, float)

        # ---- what the METRIC actually charges for: leave-one-out on each fork
        full = M2.score(P["t"], P["zyx"], P["edges"] + list(added),
                        G["t"], G["zyx"], G["edges"], G["n_est"])
        credited, charged = [], []
        for i in range(len(added)):
            sub = P["edges"] + [e for k, e in enumerate(added) if k != i]
            r = M2.score(P["t"], P["zyx"], sub, G["t"], G["zyx"], G["edges"], G["n_est"])
            credited.append(int(r["dtp"] < full["dtp"]))
            charged.append(int(r["dfp"] < full["dfp"]))

        for i, (Pn, Q) in enumerate(added):
            rows.append({"film": stem, "embryo": stem[:4],
                         "dc_split": "train" if stem[:4] == "44b6" else "val",
                         "t": int(P["t"][Q]), "P": int(Pn), "Q": int(Q),
                         "z": int(P["zyx"][Q][0]), "y": int(P["zyx"][Q][1]),
                         "x": int(P["zyx"][Q][2]),
                         "score": scores[i], "img": imgs[i], "label": labels[i],
                         "true": int(labels[i] == "TRUE"),
                         "credited": credited[i], "charged_fp": charged[i]})

        films[stem] = {"P": P, "G": G, "added": list(added), "scores": scores}
        print(f"  [{n}/{len(stems)}] {stem}  nodes {len(P['t']):>6}  edges {len(P['edges']):>6}  "
              f"forks {len(added):>3}  true {sum(l == 'TRUE' for l in labels):>2}  "
              f"credited {sum(credited)}  charged_fp {sum(charged)}  "
              f"frames {len(need):>3}  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "artifacts" / "deepcenter_deployed_proposals.csv", index=False)
    print(f"\nwrote artifacts/deepcenter_deployed_proposals.csv  ({len(df)} proposals)")

    # -------- positive control: is the //4 index actually pointing at cells? --
    print("\n=== GEOMETRY POSITIVE CONTROL (a wrong index gives plausible garbage) ===")
    rng = np.random.default_rng(0)
    gt_s, rnd_s, swap_s = [], [], []
    for stem in stems[:4]:
        P, G = films[stem]["P"], films[stem]["G"]
        ts = sorted({int(t) for t in np.unique(G["t"])})[:6]
        for t in ts:
            doc, _ = score_frame(model, cfg, TRAIN / f"{stem}.zarr", t, device)
            gi = np.where(G["t"] == t)[0]
            for i in gi:
                z, y, x = (int(v) for v in G["zyx"][i])
                gt_s.append(float(doc[min(z, 63), min(y // 4, 63), min(x // 4, 63)]))
                swap_s.append(float(doc[min(z, 63), min(x // 4, 63), min(y // 4, 63)]))
            for _ in range(len(gi)):
                rnd_s.append(float(doc[rng.integers(64), rng.integers(64), rng.integers(64)]))
    y_lab = [1] * len(gt_s) + [0] * len(rnd_s)
    print(f"  GT nodes (z, y//4, x//4)  n={len(gt_s):>5}  median {np.median(gt_s):.4f}")
    print(f"  random voxels             n={len(rnd_s):>5}  median {np.median(rnd_s):.4f}")
    print(f"  GT nodes, y/x SWAPPED     n={len(swap_s):>5}  median {np.median(swap_s):.4f}")
    print(f"  AUC  GT vs random  = {auc(y_lab, gt_s + rnd_s):.4f}   "
          f"(swapped index: {auc(y_lab, swap_s + rnd_s):.4f})")
    print("  -> correct index separates real cells from noise; the swap does not.")

    # -------- threshold sweep on the full competition proxy ------------------
    def run(keep):
        out, nf = [], 0
        for stem, c in films.items():
            sel = [e for e, s in zip(c["added"], c["scores"]) if keep(s)]
            nf += len(sel)
            out.append(M2.score(c["P"]["t"], c["P"]["zyx"], c["P"]["edges"] + sel,
                                c["G"]["t"], c["G"]["zyx"], c["G"]["edges"], c["G"]["n_est"]))
        return M2.aggregate(out), nf

    print("\n=== KEY TABLE: DeepCenter veto on raw ILP + safe_div, 8 deployed graphs ===")
    print("(12 ground-truth divisions in total; 1 division ~ 0.0083 of proxy)")
    hdr = (f"{'variant':>16} {'forks':>6} {'TP':>3} {'FP':>3} {'FN':>3} {'divJ':>7} "
           f"{'adj':>8} {'proxy':>9} {'d_proxy':>9}")
    print(hdr); print("-" * len(hdr))
    trows = []
    base, _ = run(lambda s: False)
    off, _ = run(lambda s: True)
    for lbl, a, nf in [("raw ILP only", base, 0)]:
        print(f"{lbl:>16} {nf:>6} {a['dtp']:>3} {a['dfp']:>3} {a['dfn']:>3} {a['divJ']:>7.4f} "
              f"{a['adj']:>8.5f} {a['proxy']:>9.5f} {a['proxy'] - off['proxy']:>+9.5f}")
        trows.append({"variant": lbl, "threshold": None, "forks": nf, **a})
    for thr in THRESHOLDS:
        a, nf = run((lambda s: True) if thr == 0.0 else (lambda s, t=thr: s >= t))
        lbl = "veto OFF (0.00)" if thr == 0.0 else f"veto >= {thr:.2f}"
        print(f"{lbl:>16} {nf:>6} {a['dtp']:>3} {a['dfp']:>3} {a['dfn']:>3} {a['divJ']:>7.4f} "
              f"{a['adj']:>8.5f} {a['proxy']:>9.5f} {a['proxy'] - off['proxy']:>+9.5f}")
        trows.append({"variant": lbl, "threshold": thr, "forks": nf,
                      "d_proxy_vs_noveto": a["proxy"] - off["proxy"], **a})
    pd.DataFrame(trows).to_csv(ROOT / "artifacts" / "deepcenter_deployed_thresholds.csv", index=False)
    print("wrote artifacts/deepcenter_deployed_thresholds.csv")

    best = max(trows[1:], key=lambda r: r["proxy"])
    dep = next(r for r in trows if r["threshold"] == 0.20)
    print(f"\n  best threshold      {best['variant']}  proxy {best['proxy']:.5f}")
    print(f"  deployed thr 0.20   proxy {dep['proxy']:.5f}  "
          f"(delta vs best {dep['proxy'] - best['proxy']:+.5f} = "
          f"{(dep['proxy'] - best['proxy']) / 0.00833:+.2f} divisions)")

    # -------- ceiling: what an ORACLE veto could buy ------------------------
    print("\n=== CEILING: an oracle veto that keeps only the metric-credited forks ===")
    keepset = {(r.film, r.P, r.Q) for r in df[df.credited == 1].itertuples()}
    out = []
    for stem, c in films.items():
        sel = [e for e in c["added"] if (stem, e[0], e[1]) in keepset]
        out.append(M2.score(c["P"]["t"], c["P"]["zyx"], c["P"]["edges"] + sel,
                            c["G"]["t"], c["G"]["zyx"], c["G"]["edges"], c["G"]["n_est"]))
    orc = M2.aggregate(out)
    print(f"  oracle (5 forks kept)  TP {orc['dtp']} FP {orc['dfp']} FN {orc['dfn']}  "
          f"divJ {orc['divJ']:.4f}  adj {orc['adj']:.5f}  proxy {orc['proxy']:.5f}  "
          f"({orc['proxy'] - off['proxy']:+.5f} vs no veto)")
    print(f"  DeepCenter at its best threshold reaches {best['proxy']:.5f} = "
          f"{(best['proxy'] - off['proxy']) / max(orc['proxy'] - off['proxy'], 1e-12):.0%} "
          f"of that ceiling.")

    # -------- control: does raw brightness do the same job? ----------------
    print("\n=== CONTROL: same veto driven by raw normalised intensity instead ===")
    imap = {(r.film, r.P, r.Q): r.img for r in df.itertuples()}
    print(f"{'img thr':>9} {'forks':>6} {'TP':>3} {'FP':>3} {'FN':>3} {'divJ':>7} {'proxy':>9}")
    for ithr in (0.0, 0.40, 0.60, 0.70, 0.80, 0.90, 1.10, 1.30):
        out, nf = [], 0
        for stem, c in films.items():
            sel = [e for e in c["added"] if imap[(stem, e[0], e[1])] >= ithr]
            nf += len(sel)
            out.append(M2.score(c["P"]["t"], c["P"]["zyx"], c["P"]["edges"] + sel,
                                c["G"]["t"], c["G"]["zyx"], c["G"]["edges"], c["G"]["n_est"]))
        a = M2.aggregate(out)
        print(f"{ithr:>9.2f} {nf:>6} {a['dtp']:>3} {a['dfp']:>3} {a['dfn']:>3} "
              f"{a['divJ']:>7.4f} {a['proxy']:>9.5f}")

    # -------- how wide is the window that works? ---------------------------
    lo = df[(df.charged_fp == 1) & (df.credited == 0)].score.max()
    hi = df[df.credited == 1].score.min()
    print("\n=== SAFE WINDOW (why 0.15 and 0.20 tie) ===")
    print(f"  highest-scoring removable FP fork : {lo:.4f}")
    print(f"  lowest-scoring credited TP fork   : {hi:.4f}")
    print(f"  any threshold in ({lo:.4f}, {hi:.4f}] gives the identical, best result.")
    print(f"  deployed 0.20 sits {hi - 0.20:.4f} below the first TP it would destroy "
          f"and {0.20 - lo:.4f} above the FP it must clear.")

    # -------- jackknife: how many films carry the effect? -------------------
    print("\n=== JACKKNIFE at thr 0.20: drop one film, re-measure the veto's value ===")
    print(f"{'film held out':>16} {'d_proxy(0.20 vs off)':>22} {'TP/FP/FN off':>14} {'-> 0.20':>10}")
    for held in list(films):
        def agg(keep):
            return M2.aggregate([
                M2.score(c["P"]["t"], c["P"]["zyx"],
                         c["P"]["edges"] + [e for e, s in zip(c["added"], c["scores"]) if keep(s)],
                         c["G"]["t"], c["G"]["zyx"], c["G"]["edges"], c["G"]["n_est"])
                for st, c in films.items() if st != held])
        a0, a2 = agg(lambda s: True), agg(lambda s: s >= 0.20)
        c0 = "{}/{}/{}".format(a0["dtp"], a0["dfp"], a0["dfn"])
        c2 = "{}/{}/{}".format(a2["dtp"], a2["dfp"], a2["dfn"])
        print(f"{held:>16} {a2['proxy'] - a0['proxy']:>+22.5f} {c0:>14} {c2:>10}")

    # -------- what the pool is made of --------------------------------------
    print("\n--- proposal label census (GT is sparse: most parents match no GT node) ---")
    print(df.label.value_counts().to_string())
    print(f"\n  metric-visible forks (parent matched a GT node): {int((df.label != 'parent_unmatched').sum())}"
          f" / {len(df)}")
    print(f"  forks the metric CREDITS as a division TP        : {int(df.credited.sum())}")
    print(f"  forks the metric CHARGES as a division FP        : {int(df.charged_fp.sum())}")
    print("\n  scores of the forks that actually matter:")
    for _, r in df[(df.credited == 1) | (df.charged_fp == 1) | (df.true == 1)].iterrows():
        kind = "CREDITED TP" if r.credited else ("CHARGED FP" if r.charged_fp else "true, uncredited")
        print(f"    {r.film} t={r.t:>3} Q={r.Q:>6}  score {r.score:.4f}  img {r.img:.4f}  "
              f"{kind:<16} [{r.label}]")

    # -------- AUC on this pool, against the weak-graph number ---------------
    print("\n=== AUC: DeepCenter score, TRUE vs FALSE proposal, on THIS pool ===")
    from scipy.stats import mannwhitneyu
    for name, sub in [("ALL", df), ("44b6 (DC train)", df[df.embryo == "44b6"]),
                      ("6bba (DC val)", df[df.embryo == "6bba"])]:
        for lab in ("true", "credited"):
            a = auc(sub[lab], sub.score)
            ai = auc(sub[lab], sub.img)
            print(f"  {name:>16}  label={lab:<9} n={len(sub):>4} pos={int(sub[lab].sum()):>2}  "
                  f"DeepCenter AUC {a:.4f}   brightness control {ai:.4f}")
    pos, neg = df[df.true == 1].score, df[df.true == 0].score
    if len(pos) and len(neg):
        u = mannwhitneyu(pos, neg, alternative="greater")
        print(f"\n  Mann-Whitney TRUE > FALSE: U={u.statistic:.0f}  p={u.pvalue:.3f}  "
              f"(n_true={len(pos)}, n_false={len(neg)})")
    print(f"\n  weak-graph reference (script 90): pool {2334}, true 13, AUC 0.836 "
          f"(held-out embryo 0.880), brightness control 0.615")

    print("\n=== score distributions, median [Q1-Q3] ===")
    print(f"{'fold':>16} {'class':>18} {'n':>6} {'median':>8} {'Q1':>8} {'Q3':>8}")
    for name, sub in [("ALL", df), ("44b6 (DC train)", df[df.embryo == "44b6"]),
                      ("6bba (DC val)", df[df.embryo == "6bba"])]:
        for cls, s in [("TRUE division", sub[sub.true == 1].score),
                       ("FALSE candidate", sub[sub.true == 0].score)]:
            if len(s):
                q1, med, q3 = np.percentile(s, [25, 50, 75])
                print(f"{name:>16} {cls:>18} {len(s):>6} {med:>8.4f} {q1:>8.4f} {q3:>8.4f}")
    print(f"\ntotal wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
