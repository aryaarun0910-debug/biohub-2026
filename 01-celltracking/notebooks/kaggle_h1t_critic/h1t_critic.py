"""Kaggle training kernel — H1-T compact external fork critic.

Trains ONLY on the external Zebrahub event set (CC BY 4.0, Lange et al., Cell 2024).
No competition data is read here; competition data is used solely for leave-one-embryo-out
evaluation and a single cross-fitted threshold, both of which run locally on the exact
patched scorer.

Self-contained on purpose:
  * reads NPZ, never parquet — the Kaggle image ships a polars whose compiled backend does
    not load;
  * globs /kaggle/input recursively — a previous kernel here ran empty three times because
    it hardcoded the mount path;
  * writes temp files ending in .npz — np.savez_compressed appends .npz otherwise and the
    atomic replace fails after the work is done.

Objective and stopping rule are the deployment ones, not a generic loss: per mother the
top-scoring shortlist pair is admitted above tau, and the model is selected on the resulting
division Jaccard J = TP / (TP + FP + FN) on a held-in embryo window.

Outputs: /kaggle/working/h1t_metrics.json, /kaggle/working/h1t_critic_weights.npz
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

SEED = 0
HIDDEN = (48, 24)
MAX_EPOCHS = 60
PATIENCE = 8
NEG_PER_POS = 40
BATCH = 4096
LR = 1e-3
WD = 1e-4
OUT = "/kaggle/working"


def load():
    hits = sorted(glob.glob("/kaggle/input/**/h1t_external_events.npz", recursive=True))
    if not hits:
        hits = sorted(glob.glob("/kaggle/input/**/*.npz", recursive=True))
    if not hits:
        raise SystemExit("no external event archive found under /kaggle/input")
    print("loading", hits[0], flush=True)
    z = np.load(hits[0], allow_pickle=False)
    return (z["X"].astype(np.float32), z["y"].astype(np.float32), z["embryo"],
            z["group"], [str(s) for s in z["feats"]], [str(s) for s in z["embryos"]],
            z["t_lo"])


def mlp(n_in):
    torch.manual_seed(SEED)
    layers, prev = [], n_in
    for h in HIDDEN:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers)


def division_jaccard(scores, y, group, n_gt, taus):
    """One pair per mother (argmax), admit above tau, J = TP / (n_gt + FP)."""
    order = np.argsort(-scores, kind="stable")
    seen, keep = set(), []
    for i in order:
        g = group[i]
        if g in seen:
            continue
        seen.add(g)
        keep.append(i)
    keep = np.asarray(keep)
    s, yy = scores[keep], y[keep]
    best = (0.0, -1.0, 0, 0)
    for tau in taus:
        sel = s >= tau
        tp = int(yy[sel].sum())
        fp = int(sel.sum() - tp)
        j = tp / (n_gt + fp) if (n_gt + fp) else 0.0
        if j > best[1]:
            best = (float(tau), j, tp, fp)
    return best


def score_all(net, X, mu, sd, dev):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 200_000):
            z = torch.from_numpy((X[i:i + 200_000] - mu) / sd).to(dev)
            out.append(net(z).squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def balance(X, y, seed=SEED):
    rng = np.random.default_rng(seed)
    pi = np.flatnonzero(y == 1)
    ni = np.flatnonzero(y == 0)
    ni = rng.choice(ni, size=min(len(ni), NEG_PER_POS * max(len(pi), 1)), replace=False)
    idx = np.concatenate([pi, ni])
    rng.shuffle(idx)
    return X[idx], y[idx]


def train(Xtr, ytr, Xv, yv, gv, dev):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = mlp(Xtr.shape[1]).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    Z = torch.from_numpy((Xtr - mu) / sd).to(dev)
    Y = torch.from_numpy(ytr).to(dev)
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], dtype=torch.float32).to(dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    ngtv = int(yv.sum())
    best_j, best_state, best_ep, bad, loss = -1.0, None, -1, 0, torch.tensor(0.0)
    for ep in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(len(Z), generator=g).to(dev)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(net(Z[b]).squeeze(-1), Y[b])
            loss.backward()
            opt.step()
        sv = score_all(net, Xv, mu, sd, dev)
        tau, j, tp, fp = division_jaccard(sv, yv, gv, ngtv,
                                          np.quantile(sv, np.linspace(0.5, 0.99999, 200)))
        if j > best_j + 1e-6:
            best_j, best_ep, bad = j, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        print(f"    ep{ep:02d} loss={float(loss):.4f} innerval_divJ={j:.4f} "
              f"(tp{tp}/fp{fp}) best={best_j:.4f}@{best_ep}", flush=True)
        if bad >= PATIENCE:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, mu, sd, best_j, best_ep, n_par


def main() -> None:
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, emb, group, feats, embryos, t_lo = load()
    print(f"device={dev} rows={len(y):,} positives={int(y.sum()):,} feats={len(feats)} "
          f"embryos={embryos}", flush=True)
    res = {"device": dev, "rows": int(len(y)), "positives": int(y.sum()),
           "features": feats, "embryos": embryos, "hidden": list(HIDDEN)}

    # ---- leave-one-embryo-out --------------------------------------------------------
    loeo = {}
    for k, name in enumerate(embryos):
        te = emb == k
        tr_all = ~te
        inner = max(i for i in range(len(embryos)) if i != k)
        lastw = t_lo[(emb == inner)].max()
        va = (emb == inner) & (t_lo == lastw)
        tr = tr_all & ~va
        Xtr, ytr = balance(X[tr], y[tr])
        print(f"  held-out {name}: train {len(ytr):,} ({int(ytr.sum())} pos) "
              f"inner-val {embryos[inner]}@t{lastw} ({int(y[va].sum())} pos)", flush=True)
        net, mu, sd, bj, bep, npar = train(Xtr, ytr, X[va], y[va], group[va], dev)
        s = score_all(net, X[te], mu, sd, dev)
        yt, gt = y[te], group[te]
        ngt = int(yt.sum())
        tau, j, tp, fp = division_jaccard(s, yt, gt, ngt,
                                          np.quantile(s, np.linspace(0.5, 0.999999, 300)))
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc = float(roc_auc_score(yt, s))
        pr = float(average_precision_score(yt, s))
        print(f"   -> {name}: ROC-AUC={roc:.4f} PR-AUC={pr:.4f} prev={yt.mean():.2e} "
              f"divJ={j:.4f} TP{tp}/FP{fp}", flush=True)
        loeo[name] = {"roc_auc": roc, "pr_auc": pr, "prevalence": float(yt.mean()),
                      "div_j": j, "tp": tp, "fp": fp, "tau": tau,
                      "inner_val_div_j": bj, "best_epoch": bep, "n_params": npar}
    res["loeo"] = loeo
    res["loeo_mean_roc_auc"] = float(np.mean([v["roc_auc"] for v in loeo.values()]))
    res["loeo_mean_pr_auc"] = float(np.mean([v["pr_auc"] for v in loeo.values()]))

    # ---- deployment model: every external embryo -------------------------------------
    inner = len(embryos) - 1
    lastw = t_lo[emb == inner].max()
    va = (emb == inner) & (t_lo == lastw)
    Xtr, ytr = balance(X[~va], y[~va])
    print(f"  deploy: train {len(ytr):,} ({int(ytr.sum())} pos)", flush=True)
    net, mu, sd, bj, bep, npar = train(Xtr, ytr, X[va], y[va], group[va], dev)
    res.update({"deploy_inner_val_div_j": bj, "deploy_best_epoch": bep, "n_params": npar,
                "runtime_s": time.time() - t0})

    sd_np = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    tmp = os.path.join(OUT, "h1t_critic_weights.tmp.npz")
    np.savez(tmp, mu=mu, sd=sd, **sd_np)
    os.replace(tmp, os.path.join(OUT, "h1t_critic_weights.npz"))
    with open(os.path.join(OUT, "h1t_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=2,
                     default=float), flush=True)


if __name__ == "__main__":
    main()
