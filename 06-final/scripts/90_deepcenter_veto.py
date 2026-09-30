"""H-DC: can an independently trained cell-PRESENCE detector veto false division candidates?

Hypothesis under test: the division-candidate pool is mostly junk because the
candidate DAUGHTER node Q is a false-positive detection from the primary
detector.  If so, DeepCenter (a separately trained 3D UNet center-heatmap
detector) should score those Q low, and thresholding on that score should raise
division precision.

This is a cell-PRESENCE check, NOT a mitosis classifier.  G3b already showed the
primary encoder cannot linearly decode mitotic state at the split frame
(AUC 0.456).  Different question, different model, do not conflate.

Geometry note (verified, not assumed):
  raw frame                       (64, 256, 256)
  primary grid  = raw[::1,::4,::4] -> (64, 64, 64), index j <-> raw y = 4j
  DeepCenter    = block_mean_xy(4) -> (64, 64, 64), block k covers raw [4k,4k+4)
  => det_zyx (primary grid) maps 1:1 onto the DeepCenter pooled grid.  The
  documented "y/pool_factor" step is already applied by the primary downsample.

Outputs: artifacts/deepcenter_veto_proposals.csv
         artifacts/deepcenter_veto_thresholds.csv
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

from biohub import io as BIO  # noqa: E402
from biohub import division as DV  # noqa: E402
from biohub import metric as MT  # noqa: E402
from biohub import postprocess as PP  # noqa: E402

DC_DIR = ROOT / "weights" / "biohub-deepcenter-unet3d-center-prior-v1" / "weights" / "full_frame_center"
GRAPHS = ROOT / "artifacts" / "graphs"
TRAIN = BIO.dataset_root() / "train"
N_FILMS_PER_EMBRYO = 20
MIN_TRACK_LEN = 6
THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70]


# --------------------------------------------------------------------------
# architecture: transcribed verbatim from
# source_scripts/train_full_frame_center_detector.py (ConvBlock3d / DeepCenterUNet3D)
# --------------------------------------------------------------------------
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
    n_par = sum(p.numel() for p in model.parameters())
    meta = {
        "epoch": int(ck.get("epoch", -1)),
        "best_score": float(ck.get("best_score", float("nan"))),
        "history_len": len(ck.get("history", [])),
        "params": n_par,
        "n_tensors": len(ck["model_state"]),
        **{k: cfg[k] for k in ("base_channels", "pool_factor", "norm_lo_pct",
                               "norm_hi_pct", "norm_clip_lo", "norm_clip_hi")},
    }
    return model.to(device).eval(), cfg, meta


@torch.no_grad()
def score_frame(model, cfg, zarr_path: Path, t: int, device: str):
    """-> (prob, prob_doc, prob_iso, img_doc), each (64,64,64) on the pooled grid.

    img_doc is the CONTROL: the normalised image itself, max-pooled over the same
    window.  If DeepCenter is only reporting "this voxel is bright", the control
    separates the classes just as well and the network earns nothing.
    """
    raw = BIO.read_frame(zarr_path, t)
    pooled = block_mean_xy(raw, int(cfg["pool_factor"]))
    img = normalize_dynamic_range(pooled, cfg["norm_lo_pct"], cfg["norm_hi_pct"],
                                  cfg["norm_clip_lo"], cfg["norm_clip_hi"])
    x = torch.from_numpy(img)[None, None].to(device)
    prob = torch.sigmoid(model(x))
    # documented scoring window: max over z+-1, y+-2, x+-2 on the pooled grid
    doc = F.max_pool3d(prob, (3, 5, 5), stride=1, padding=(1, 2, 2))
    iso = F.max_pool3d(prob, 3, stride=1, padding=1)
    imd = F.max_pool3d(x, (3, 5, 5), stride=1, padding=(1, 2, 2))
    return (prob[0, 0].cpu().numpy(), doc[0, 0].cpu().numpy(),
            iso[0, 0].cpu().numpy(), imd[0, 0].cpu().numpy())


def pick_films() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "artifacts" / "films.csv")
    have = {p.stem for p in GRAPHS.glob("*.npz")}
    df = df[df.film.isin(have)].copy()
    out = []
    for emb, g in df.groupby("embryo"):
        g = g.sort_values(["divisions", "film"], ascending=[False, True])
        out.append(g.head(N_FILMS_PER_EMBRYO))
    return pd.concat(out).reset_index(drop=True)


def gt_index_edges(z) -> np.ndarray:
    idx = {int(i): k for k, i in enumerate(z["gt_ids"])}
    return np.array([[idx[int(a)], idx[int(b)]] for a, b in z["gt_edges"]],
                    np.int64).reshape(-1, 2)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, cfg, meta = load_deepcenter(device)
    print("=== DeepCenter checkpoint ===")
    print(f"  path            {DC_DIR / 'best.pt'}")
    print(f"  state_dict      {meta['n_tensors']} tensors, {meta['params']:,} params, "
          f"strict load OK (0 missing / 0 unexpected)")
    print(f"  base_channels   {meta['base_channels']}   pool_factor {meta['pool_factor']}")
    print(f"  normalisation   clip((v-P{meta['norm_lo_pct']})/(P{meta['norm_hi_pct']}"
          f"-P{meta['norm_lo_pct']}), {meta['norm_clip_lo']}, {meta['norm_clip_hi']})")
    print(f"  epoch           {meta['epoch']}  (best val loss {-meta['best_score']:.6f}, "
          f"history in best.pt = {meta['history_len']} epochs)")
    sm = json.loads((DC_DIR / "split_manifest.json").read_text())
    tr_emb = sorted({f[:4] for f in sm["train"]})
    va_emb = sorted({f[:4] for f in sm["val"]})
    print(f"  train split     {len(sm['train'])} films, embryos {tr_emb}")
    print(f"  val split       {len(sm['val'])} films, embryos {va_emb}")
    print(f"  device          {device}\n")

    sel = pick_films()
    print(f"=== subset: {len(sel)} films ===")
    print(sel.groupby("embryo").agg(films=("film", "size"), divisions=("divisions", "sum")))
    print()

    rows = []          # per proposal
    det_rows = []      # per detection in scored frames (presence-detector sanity check)
    per_film = []
    films_cache = {}   # everything score_film needs, so the veto can be re-scored
    t_start = time.time()

    for n, rec in enumerate(sel.itertuples(), 1):
        film, emb = rec.film, rec.embryo
        z = np.load(GRAPHS / f"{film}.npz")
        det_t = z["det_t"].astype(np.int64)
        det_zyx = z["det_zyx"].astype(np.int64)
        base = [(int(a), int(b)) for a, b, *_ in z["linked_edges"]]

        edges, dt, dz, keep, stats = PP.prune_and_filter(
            base, det_t, det_zyx, min_track_len=MIN_TRACK_LEN)
        _, proposals = DV.augment(edges, dt, dz, DV.DivConfig(min_track_len=MIN_TRACK_LEN))

        gt_e = gt_index_edges(z)
        p2g, g2p = MT.match_nodes(z["gt_grid"], z["gt_t"].astype(np.int64),
                                  dz.astype(np.float32), dt)
        gt_children: dict[int, list[int]] = {}
        for a, b in gt_e:
            gt_children.setdefault(int(a), []).append(int(b))
        gt_div_src = {g for g, ch in gt_children.items() if len(ch) >= 2}

        need = sorted({int(dt[Q]) for _, Q in proposals})
        zpath = TRAIN / f"{film}.zarr"
        cache = {}
        for t in need:
            cache[t] = score_frame(model, cfg, zpath, t, device)

        # per-proposal rows
        prop_scores = []
        for P, Q in proposals:
            t = int(dt[Q])
            prob, doc, iso, imd = cache[t]
            zz, yy, xx = (int(v) for v in dz[Q])
            prop_scores.append(float(doc[zz, yy, xx]))
            mg = p2g.get(int(P))
            mq = p2g.get(int(Q))
            if mg is None:
                label = "parent_unmatched"
            elif mg not in gt_div_src:
                label = "parent_matched_not_divider"
            elif mq is not None and mq in gt_children[mg]:
                label = "TRUE"
            else:
                label = "parent_is_divider_wrong_daughter"
            rows.append({
                "film": film, "embryo": emb, "t": t, "P": int(P), "Q": int(Q),
                "score": float(doc[zz, yy, xx]),
                "score_iso": float(iso[zz, yy, xx]),
                "score_pt": float(prob[zz, yy, xx]),
                "img": float(imd[zz, yy, xx]),
                "label": label, "true": int(label == "TRUE"),
                "gt_src": int(mg) if mg is not None else -1,
            })

        # every pruned detection living in a scored frame, flagged by GT match
        in_need = np.isin(dt, need)
        for i in np.where(in_need)[0]:
            t = int(dt[i])
            prob, doc, iso, imd = cache[t]
            zz, yy, xx = (int(v) for v in dz[i])
            det_rows.append((emb, float(doc[zz, yy, xx]), float(imd[zz, yy, xx]),
                             1 if int(i) in p2g else 0))

        films_cache[film] = {
            "emb": emb, "dt": dt, "dz": dz.astype(np.float32), "edges": edges,
            "proposals": proposals, "pscore": np.asarray(prop_scores, np.float32),
            "gt_grid": z["gt_grid"], "gt_t": z["gt_t"].astype(np.int64),
            "gt_e": gt_e, "n_est": float(z["est_nodes"]),
        }

        n_true = sum(r["true"] for r in rows if r["film"] == film)
        per_film.append({"film": film, "embryo": emb, "gt_divisions": int(rec.divisions),
                         "nodes_pruned": int(len(dt)), "removed": stats["removed"],
                         "proposals": len(proposals), "true_proposals": n_true,
                         "frames_scored": len(need),
                         "gt_div_matched": sum(1 for g in gt_div_src if g in g2p)})
        print(f"  [{n:>2}/{len(sel)}] {film} {emb}  nodes {len(dt):>6}  "
              f"proposals {len(proposals):>4}  true {n_true:>2}  "
              f"frames {len(need):>3}  gtdiv {int(rec.divisions)}  "
              f"({time.time() - t_start:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    pf = pd.DataFrame(per_film)
    df.to_csv(ROOT / "artifacts" / "deepcenter_veto_proposals.csv", index=False)
    print(f"\nwrote artifacts/deepcenter_veto_proposals.csv  ({len(df)} proposals)")

    dd = pd.DataFrame(det_rows, columns=["embryo", "score", "img", "gt_matched"])

    # ---------------- threshold table ----------------
    total_gt_div = int(sel.divisions.sum())
    n_true_all = int(df.true.sum())
    print("\n=== KEY TABLE: DeepCenter score as a veto on division proposals ===")
    print(f"(pool = {len(df)} proposals over {len(sel)} films; "
          f"{n_true_all} are TRUE divisions; {total_gt_div} labelled divisions in subset)")
    hdr = (f"{'thr':>6} {'survive':>8} {'true':>5} {'precision':>10} {'rec_pool':>9} "
           f"{'rec_GT':>7} {'prec_44b6':>10} {'prec_6bba':>10}")
    print(hdr)
    print("-" * len(hdr))
    trows = []
    for thr in THRESHOLDS:
        keep = df[df.score >= thr] if thr > 0 else df
        n, k = len(keep), int(keep.true.sum())
        prec = k / n if n else float("nan")
        rec_pool = k / n_true_all if n_true_all else float("nan")
        gt_hit = keep[keep.true == 1].groupby(["film", "gt_src"]).ngroups
        rec_gt = gt_hit / total_gt_div if total_gt_div else float("nan")
        sub = {}
        for emb in ("44b6", "6bba"):
            ke = keep[keep.embryo == emb]
            sub[emb] = (int(ke.true.sum()) / len(ke)) if len(ke) else float("nan")
        print(f"{thr:>6.2f} {n:>8} {k:>5} {prec:>10.4f} {rec_pool:>9.3f} {rec_gt:>7.3f} "
              f"{sub['44b6']:>10.4f} {sub['6bba']:>10.4f}")
        trows.append({"threshold": thr, "survive": n, "true": k, "precision": prec,
                      "recall_pool": rec_pool, "recall_gt_divisions": rec_gt,
                      "gt_divisions_hit": gt_hit,
                      "precision_44b6": sub["44b6"], "precision_6bba": sub["6bba"],
                      "survive_44b6": int((keep.embryo == "44b6").sum()),
                      "survive_6bba": int((keep.embryo == "6bba").sum())})
    pd.DataFrame(trows).to_csv(ROOT / "artifacts" / "deepcenter_veto_thresholds.csv", index=False)
    print("wrote artifacts/deepcenter_veto_thresholds.csv")

    print("\n--- proposal label census (GT is SPARSE: most parents match nothing) ---")
    print(df.label.value_counts().to_string())

    # The competition metric only charges a division FP when the fork source
    # MATCHED a GT node.  Proposals whose parent matched nothing are invisible to
    # the metric, so this is the pool where precision is actually defined.
    mm = df[df.gt_src >= 0]
    print(f"\n=== SECONDARY TABLE: proposals whose parent P matched a GT node "
          f"(n={len(mm)}, the metric-visible pool) ===")
    print(f"{'thr':>6} {'survive':>8} {'true':>5} {'precision':>10}")
    for thr in THRESHOLDS:
        k2 = mm[mm.score >= thr] if thr > 0 else mm
        p2 = int(k2.true.sum()) / len(k2) if len(k2) else float("nan")
        print(f"{thr:>6.2f} {len(k2):>8} {int(k2.true.sum()):>5} {p2:>10.4f}")

    # ---------------- AUC + distributions ----------------
    from sklearn.metrics import roc_auc_score

    def auc(y, s):
        y = np.asarray(y)
        return float(roc_auc_score(y, np.asarray(s))) if 0 < y.sum() < len(y) else float("nan")

    def iqr(s):
        s = np.asarray(s, float)
        if not len(s):
            return (float("nan"),) * 3
        return tuple(float(v) for v in np.percentile(s, [25, 50, 75]))

    print("\n=== AUC: DeepCenter score, TRUE vs FALSE division proposal ===")
    for name, sub in [("ALL", df)] + [(e, df[df.embryo == e]) for e in ("44b6", "6bba")]:
        for col in ("score", "score_iso", "score_pt"):
            print(f"  {name:>5}  {col:<10} AUC {auc(sub.true, sub[col]):.4f}   "
                  f"(n={len(sub)}, true={int(sub.true.sum())})")

    print("\n=== score distributions (documented window), median [Q1-Q3] ===")
    print(f"{'fold':>6} {'class':>26} {'n':>7} {'median':>8} {'Q1':>8} {'Q3':>8}")
    for name, sub in [("ALL", df)] + [(e, df[df.embryo == e]) for e in ("44b6", "6bba")]:
        for cls, s in [("TRUE division", sub[sub.true == 1].score),
                       ("FALSE candidate", sub[sub.true == 0].score)]:
            q1, med, q3 = iqr(s)
            print(f"{name:>6} {cls:>26} {len(s):>7} {med:>8.4f} {q1:>8.4f} {q3:>8.4f}")
        for lab in ("parent_unmatched", "parent_matched_not_divider",
                    "parent_is_divider_wrong_daughter"):
            s = sub[sub.label == lab].score
            q1, med, q3 = iqr(s)
            print(f"{name:>6} {lab:>26} {len(s):>7} {med:>8.4f} {q1:>8.4f} {q3:>8.4f}")

    # mechanism check: can DeepCenter tell a REAL detection from any other detection?
    print("\n=== mechanism check: DeepCenter on GT-matched vs other detections ===")
    print("(all pruned detections living in the scored frames; GT labels are sparse,")
    print(" so 'other' is a mixture of unlabelled-real and false detections)")
    print(f"{'fold':>6} {'n_det':>9} {'n_gt_matched':>13} {'AUC':>8} "
          f"{'med(GT)':>9} {'med(other)':>11}")
    for name, sub in [("ALL", dd)] + [(e, dd[dd.embryo == e]) for e in ("44b6", "6bba")]:
        if not len(sub):
            continue
        a = auc(sub.gt_matched, sub.score)
        mg = float(np.median(sub[sub.gt_matched == 1].score)) if sub.gt_matched.sum() else float("nan")
        mo = float(np.median(sub[sub.gt_matched == 0].score)) if (sub.gt_matched == 0).any() else float("nan")
        print(f"{name:>6} {len(sub):>9} {int(sub.gt_matched.sum()):>13} {a:>8.4f} "
              f"{mg:>9.4f} {mo:>11.4f}")

    # candidate daughters vs GT-matched detections, directly
    print("\n=== candidate daughters Q vs GT-matched detections (same frames) ===")
    for name in ("ALL", "44b6", "6bba"):
        q = df.score if name == "ALL" else df[df.embryo == name].score
        g = dd[dd.gt_matched == 1] if name == "ALL" else dd[(dd.gt_matched == 1) & (dd.embryo == name)]
        q1, med, q3 = iqr(q)
        g1, gmed, g3 = iqr(g.score)
        print(f"  {name:>5}  candidate Q  n={len(q):>6} median {med:.4f} [{q1:.4f}-{q3:.4f}]   "
              f"GT-matched n={len(g):>5} median {gmed:.4f} [{g1:.4f}-{g3:.4f}]")

    # ---------------- control: is this just brightness? ----------------
    from scipy.stats import mannwhitneyu
    print("\n=== CONTROL: raw normalised intensity at Q, same window ===")
    print(f"{'fold':>6} {'DeepCenter AUC':>15} {'intensity AUC':>14}")
    for name, sub in [("ALL", df)] + [(e, df[df.embryo == e]) for e in ("44b6", "6bba")]:
        print(f"{name:>6} {auc(sub.true, sub.score):>15.4f} {auc(sub.true, sub.img):>14.4f}")
    print("  detection-level (GT-matched vs other):  "
          f"DeepCenter {auc(dd.gt_matched, dd.score):.4f}   "
          f"intensity {auc(dd.gt_matched, dd.img):.4f}")
    print("\n  matched-retention comparison (keep the same COUNT by each score):")
    print(f"{'keep':>7} {'DC true':>9} {'int true':>9}")
    for thr in (0.25, 0.30, 0.40):
        n_keep = int((df.score >= thr).sum())
        dc = int(df.nlargest(n_keep, "score").true.sum())
        it = int(df.nlargest(n_keep, "img").true.sum())
        print(f"{n_keep:>7} {dc:>9} {it:>9}")
    u = mannwhitneyu(df[df.true == 1].score, df[df.true == 0].score, alternative="greater")
    print(f"\n  Mann-Whitney (TRUE > FALSE, ALL): U={u.statistic:.0f}  p={u.pvalue:.2e}")
    n0, k0 = len(df), int(df.true.sum())
    for thr in (0.25, 0.30):
        n1 = int((df.score >= thr).sum())
        k1 = int(df[df.score >= thr].true.sum())
        p_null = (n1 / n0) ** k0 if k1 == k0 else float("nan")
        print(f"  thr {thr:.2f}: retains {n1}/{n0} = {n1/n0:.3f} of the pool, keeps {k1}/{k0} "
              f"true; P(all true survive | score irrelevant) = {p_null:.2e}")

    # ---------------- what the competition metric actually does ----------------
    print("\n=== METRIC EFFECT on the same 40 films (prune6 + fork, veto applied) ===")
    hdr2 = (f"{'variant':>14} {'score':>8} {'d_score':>9} {'adjJ':>8} {'divJ':>7} "
            f"{'TP':>4} {'FP':>4} {'FN':>4} {'forks':>6}")
    print(hdr2)
    print("-" * len(hdr2))
    mrows = []

    def run(keep_fn, label):
        out = []
        nf = 0
        for f, c in films_cache.items():
            sel_p = [pq for pq, s in zip(c["proposals"], c["pscore"]) if keep_fn(s)]
            nf += len(sel_p)
            e = c["edges"] + sel_p
            out.append(MT.score_film(list(range(len(c["dt"]))), e, c["dz"], c["dt"],
                                     c["gt_grid"], c["gt_t"], c["gt_e"], c["n_est"]))
        return MT.aggregate(out), nf, out

    base_agg, _, _ = run(lambda s: False, "no fork")
    print(f"{'no fork':>14} {base_agg['score']:>8.4f} {0.0:>+9.4f} "
          f"{base_agg['adj_J_edge']:>8.4f} {base_agg['div_jaccard']:>7.4f} "
          f"{base_agg['div_tp']:>4} {base_agg['div_fp']:>4} {base_agg['div_fn']:>4} {0:>6}")
    mrows.append({"variant": "no_fork", "threshold": None, "forks": 0, **base_agg})
    for thr in [None] + THRESHOLDS:
        fn = (lambda s: True) if thr is None else (lambda s, t=thr: s >= t)
        a, nf, _ = run(fn, str(thr))
        lbl = "fork (no veto)" if thr is None else f"veto>={thr:.2f}"
        print(f"{lbl:>14} {a['score']:>8.4f} {a['score'] - base_agg['score']:>+9.4f} "
              f"{a['adj_J_edge']:>8.4f} {a['div_jaccard']:>7.4f} "
              f"{a['div_tp']:>4} {a['div_fp']:>4} {a['div_fn']:>4} {nf:>6}")
        mrows.append({"variant": lbl, "threshold": thr, "forks": nf,
                      "d_score": a["score"] - base_agg["score"], **a})
    pd.DataFrame(mrows).to_csv(ROOT / "artifacts" / "deepcenter_veto_metric.csv", index=False)
    print("wrote artifacts/deepcenter_veto_metric.csv")

    # DeepCenter TRAINED on 44b6 and validated on 6bba, so the folds are not
    # interchangeable here: 6bba is the honest held-out read.
    print("\n--- metric effect per embryo (44b6 = DeepCenter TRAIN, 6bba = held out) ---")
    order = list(films_cache)
    embs = [films_cache[f]["emb"] for f in order]
    print(f"{'variant':>14} " + "  ".join(f"{e}: {'score':>7} {'divJ':>6} {'TP/FP/FN':>10}"
                                          for e in ("44b6", "6bba")))
    for thr in [None, 0.15, 0.20, 0.25]:
        fn = (lambda s: True) if thr is None else (lambda s, t=thr: s >= t)
        _, _, rws = run(fn, "")
        line = f"{('fork' if thr is None else f'veto>={thr:.2f}'):>14} "
        for e in ("44b6", "6bba"):
            a = MT.aggregate([r for r, ee in zip(rws, embs) if ee == e])
            line += (f"  {e}: {a['score']:>7.4f} {a['div_jaccard']:>6.4f} "
                     f"{a['div_tp']:>3}/{a['div_fp']:>3}/{a['div_fn']:>3}")
        print(line)

    print("\n=== per-film summary ===")
    print(pf.groupby("embryo").agg(films=("film", "size"), proposals=("proposals", "sum"),
                                   true=("true_proposals", "sum"),
                                   gt_div=("gt_divisions", "sum"),
                                   gt_div_matched=("gt_div_matched", "sum"),
                                   frames=("frames_scored", "sum")))
    print(f"\ntotal wall {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
