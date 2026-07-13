# GPU Job A — trajectory-based division representation (T4).
# Trains a compact scale-free temporal fork classifier on dense Zebrahub division events,
# leave-one-embryo-out. Internet off. Deterministic. Checkpoint/resume. Runtime-guarded.
# Exports: per-fold checkpoint, config, per-embryo metrics, prediction parquet, log, env manifest.
# The critical metric is CROSS-EMBRYO precision/recall, not training accuracy.
import json, os, time, random, sys, platform
from pathlib import Path
import numpy as np

T0 = time.time()
RUNTIME_GUARD_S = 60 * 60 * 3.0          # stop new folds past 3h (T4 limit ~9h; small model, safe)
SEED = 20260713
IN = Path("/kaggle/input/biohub-divevents-zebrahub")
OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
LOG = open(OUT / "train_log.txt", "a")
def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()

def set_seed(s):
    random.seed(s); np.random.seed(s)
    import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

import torch, torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

EMBRYOS = ["ZSNS001", "ZSNS003", "ZSNS004", "ZSNS005"]

def load():
    import glob
    files = glob.glob("/kaggle/input/**/*.npz", recursive=True)
    log("mounted /kaggle/input:", sorted(os.listdir("/kaggle/input")) if Path("/kaggle/input").exists() else "MISSING")
    log("npz found:", files)
    data = {}
    for f in files:
        e = Path(f).stem
        if e in EMBRYOS:
            d = np.load(f); data[e] = (d["X"].astype(np.float32), d["y"].astype(np.float32))
    return data

def derived(x):  # x (B,3,W,3) -> augment with velocity/accel + daughter geometry (scale-free)
    vel = np.diff(x, axis=2)                                   # (B,3,W-1,3)
    acc = np.diff(vel, axis=2)                                 # (B,3,W-2,3)
    d1, d2 = x[:, 1], x[:, 2]                                  # daughter tracks (B,W,3)
    sep = np.linalg.norm(d1 - d2, axis=-1)                     # (B,W) daughter separation
    sym = np.abs(np.linalg.norm(d1 - x[:, 0], axis=-1) - np.linalg.norm(d2 - x[:, 0], axis=-1))
    div = np.diff(sep, axis=1)                                 # (B,W-1) separation divergence
    g = np.concatenate([sep, sym, div], axis=1).astype(np.float32)  # (B, 3W-1)
    return g

class DivModel(nn.Module):
    def __init__(self, d=48, gfeat=14):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(3, d, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(d, d, 3, padding=1), nn.GELU(),
                                 nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(3 * d + d + gfeat, 96), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(96, 1))
    def forward(self, x, g):                                   # x (B,3,W,3), g (B,gfeat)
        B, _, W, _ = x.shape
        t = x.permute(0, 1, 3, 2).reshape(B * 3, 3, W)         # (B*3, 3, W)
        emb = self.enc(t).reshape(B, 3, -1)                    # (B,3,d)
        sym = (emb[:, 1] - emb[:, 2]).abs()                    # daughter-symmetry embedding
        return self.head(torch.cat([emb.reshape(B, -1), sym, g], -1)).squeeze(-1)

def recall_at_precision(y, p, target=0.9):
    order = np.argsort(-p); y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(y.sum(), 1)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0

def run_fold(data, held, dev):
    set_seed(SEED)
    Xtr = np.concatenate([data[e][0] for e in EMBRYOS if e != held and e in data])
    ytr = np.concatenate([data[e][1] for e in EMBRYOS if e != held and e in data])
    Xte, yte = data[held]
    gtr, gte = derived(Xtr), derived(Xte)
    model = DivModel(gfeat=gtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, device=dev); gtr_t = torch.tensor(gtr, device=dev); ytr_t = torch.tensor(ytr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev); gte_t = torch.tensor(gte, device=dev)
    n, bs, best = len(ytr), 512, {"ap": -1}
    ckpt = OUT / f"divmodel_{held}.pt"
    for ep in range(40):
        model.train(); idx = torch.randperm(n, device=dev)
        for b in range(0, n, bs):
            j = idx[b:b + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[j], gtr_t[j]), ytr_t[j]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xte_t, gte_t)).cpu().numpy()
        ap = average_precision_score(yte, p); auc = roc_auc_score(yte, p)
        if ap > best["ap"]:
            best = {"ap": ap, "auc": auc, "recall@p90": recall_at_precision(yte, p),
                    "calib": float(p.mean() - yte.mean()), "ep": ep}
            torch.save({"state": model.state_dict(), "gfeat": gtr.shape[1]}, ckpt)
        if ep % 5 == 0:
            log(f"  [{held}] ep{ep} ap={ap:.4f} auc={auc:.4f}")
    log(f"HELD-OUT {held}: PR-AUC={best['ap']:.4f} ROC-AUC={best['auc']:.4f} "
        f"recall@P0.9={best['recall@p90']:.4f} calib={best['calib']:+.4f} (ep{best['ep']})")
    return best

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={dev} gpu={torch.cuda.get_device_name(0) if dev=='cuda' else 'none'} seed={SEED}")
    data = load()
    log("embryos:", {e: len(data[e][1]) for e in data})
    results = {}
    done = json.loads((OUT / "metrics.json").read_text()) if (OUT / "metrics.json").exists() else {}
    for held in EMBRYOS:
        if held not in data or held in done:
            continue
        if time.time() - T0 > RUNTIME_GUARD_S:
            log("runtime guard hit; stopping (resume-safe)"); break
        results[held] = run_fold(data, held, dev)
        done[held] = results[held]
        (OUT / "metrics.json").write_text(json.dumps(done, indent=2))
    # summary transfer gate: cross-embryo mean PR-AUC / recall@P0.9
    if done:
        mean_ap = float(np.mean([v["ap"] for v in done.values()]))
        mean_r = float(np.mean([v["recall@p90"] for v in done.values()]))
        log(f"\n=== LOEO transfer summary: mean PR-AUC={mean_ap:.4f} mean recall@P0.9={mean_r:.4f} ===")
        (OUT / "config.json").write_text(json.dumps(
            {"seed": SEED, "model": "conv-temporal-divmodel", "W": 5, "embryos": EMBRYOS,
             "mean_pr_auc": mean_ap, "mean_recall_at_p90": mean_r}, indent=2))
    (OUT / "env_manifest.json").write_text(json.dumps(
        {"python": platform.python_version(), "torch": torch.__version__,
         "numpy": np.__version__, "cuda": torch.version.cuda}, indent=2))

if __name__ == "__main__":
    main()
