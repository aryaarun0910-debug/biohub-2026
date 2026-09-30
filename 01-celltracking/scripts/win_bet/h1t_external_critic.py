"""H1-T — train a compact fork critic on EXTERNAL Zebrahub events, test cross-family here.

Competition labels cannot support representation learning: 92 positives across two embryo
families. So the representation is learned ENTIRELY on external data (Zebrahub, CC-BY-4.0)
and the competition contributes only (a) leave-one-embryo-out evaluation and (b) at most one
scalar - the admission threshold - which is cross-fitted between families so it is never
chosen and tested on the same embryos. Family identity is never an input and never routes.

Train/test distributions are matched by construction:
  * candidates on both sides come from the SAME frozen proposer (cfg hash 04eeac97500d)
  * features on both sides come from the SAME code path (h1t_zebrahub_events.h1g_features,
    proven byte-identical to the deployed H1-G block)
  * the external graph the proposer sees is a GT-free prediction, so parent-stealing,
    persistence and track-age have deployment-like distributions

Early stopping is on the exact deployment objective on a held-in external embryo, not on a
generic loss: per mother take the top-scoring pair, admit above tau, and score
division Jaccard J = TP / (TP + FP + FN) - the same quantity the competition metric uses.

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\h1t_external_critic.py --loeo
  .venv\\Scripts\\python.exe scripts\\win_bet\\h1t_external_critic.py --deploy
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

SCRATCH = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026\_evidence\agent_runs\agent4")
EVENTS = SCRATCH / "external_events"
COMP = ROOT / "artifacts/kaggle/e0c_cache/fork_candidates/h1g_features"
OUT = SCRATCH / "h1t_results.json"

FEATS = ["flow_midpoint_residual", "rank", "parent_midpoint_um", "pd1_um", "pd2_um",
         "pd_ratio", "sister_um", "cos_daughter_axis", "cos_split_vs_flow", "daughter_angle",
         "persist_d1", "persist_d2", "n_persist", "mother_track_age", "mother_speed_um",
         "vel_consistency", "local_density_t", "local_density_t1", "competing_parents",
         "steal_required", "bdist_um", "resid_gap_to_best", "resid_ratio", "best_alt_gap"]

# GT divisions per family (H0c: 44b6 TP0/FP93/FN26 -> 26 total; 6bba TP4/FP582/FN121 -> 125)
N_GT_DIV = {"44b6": 26, "6bba": 125}
E0C_DIV_J = {"44b6": 0.0000, "6bba": 0.0057}
H0C_ORACLE_DIV_J = {"44b6": 16 / 26, "6bba": 76 / 125}

NEG_PER_POS = 40          # training subsample only; evaluation keeps natural prevalence
MAX_EPOCHS = 60
PATIENCE = 8
SEED = 0


# --------------------------------------------------------------------------------------
def load_external() -> pl.DataFrame:
    files = sorted(EVENTS.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no external events in {EVENTS}; run h1t_zebrahub_events.py --build")
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    return df.with_columns(pl.col("steal_required").cast(pl.Int64))


def load_competition() -> pl.DataFrame:
    lf = pl.scan_parquet(str(COMP / "*" / "*.parquet")).filter(pl.col("metric_visible"))
    return lf.collect().with_columns(pl.col("steal_required").cast(pl.Int64))


def xy(df: pl.DataFrame):
    X = df.select(FEATS).to_numpy().astype(np.float32)
    y = (df["label"] == "positive").to_numpy().astype(np.float32)
    return X, y


def division_jaccard(scores, y, mother_key, n_gt: int, taus):
    """Deployment rule: one pair per mother (argmax score), admit above tau.

    Returns (best_tau, best_J, tp, fp) plus the full sweep.
    """
    order = np.argsort(-scores, kind="stable")
    seen, keep = set(), []
    for i in order:
        m = mother_key[i]
        if m in seen:
            continue
        seen.add(m)
        keep.append(i)
    keep = np.asarray(keep)
    s, yy = scores[keep], y[keep]
    sweep = []
    for tau in taus:
        sel = s >= tau
        tp = int(yy[sel].sum())
        fp = int(sel.sum() - tp)
        j = tp / (n_gt + fp) if (n_gt + fp) else 0.0
        sweep.append((float(tau), j, tp, fp))
    best = max(sweep, key=lambda z: z[1])
    return best, sweep


def auc_pr(scores, y):
    from sklearn.metrics import roc_auc_score, average_precision_score
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    return float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))


def top1(scores, y, mother_key):
    """Among mothers whose shortlist contains the true pair, how often is it ranked first."""
    from collections import defaultdict
    g = defaultdict(list)
    for i, m in enumerate(mother_key):
        g[m].append(i)
    hit = tot = 0
    for m, ii in g.items():
        ii = np.asarray(ii)
        if y[ii].sum() < 1:
            continue
        tot += 1
        hit += int(y[ii[np.argmax(scores[ii])]] == 1)
    return hit / max(tot, 1), tot


# --------------------------------------------------------------------------------------
class Critic:
    """Compact MLP: 24 -> 48 -> 24 -> 1, 2,401 parameters. Deliberately the smallest model
    that can express feature interactions; a linear control is reported alongside."""

    def __init__(self, n_in, hidden=(48, 24), seed=SEED):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
        self.n_params = sum(p.numel() for p in self.net.parameters())
        self.mu = None
        self.sd = None

    def fit_scaler(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6

    def _z(self, X):
        return (X - self.mu) / self.sd

    def score(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            out = []
            Z = self._z(X)
            for i in range(0, len(Z), 200_000):
                out.append(self.net(torch.from_numpy(Z[i:i + 200_000])).squeeze(-1).numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def fit(self, Xtr, ytr, val, log=print):
        """val = (Xv, yv, mother_v, n_gt_v). Early stop on exact division Jaccard."""
        import torch
        import torch.nn as nn
        self.fit_scaler(Xtr)
        Z = torch.from_numpy(self._z(Xtr))
        Y = torch.from_numpy(ytr)
        pos_w = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(SEED)
        best_j, best_state, best_ep, bad = -1.0, None, -1, 0
        loss = torch.tensor(0.0)
        Xv, yv, mv, ngtv = val
        for ep in range(MAX_EPOCHS):
            self.net.train()
            perm = torch.randperm(len(Z), generator=g)
            for i in range(0, len(perm), 4096):
                b = perm[i:i + 4096]
                opt.zero_grad()
                loss = lossf(self.net(Z[b]).squeeze(-1), Y[b])
                loss.backward()
                opt.step()
            sv = self.score(Xv)
            taus = np.quantile(sv, np.linspace(0.5, 0.99999, 200))
            (tau, j, tp, fp), _ = division_jaccard(sv, yv, mv, ngtv, taus)
            if j > best_j + 1e-6:
                best_j, best_ep, bad = j, ep, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
            if ep % 5 == 0 or bad >= PATIENCE:
                log(f"      ep{ep:02d} loss={float(loss):.4f} innerval_divJ={j:.4f} "
                    f"(tp{tp}/fp{fp}) best={best_j:.4f}@{best_ep}")
            if bad >= PATIENCE:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return best_j, best_ep


def balance(X, y, seed=SEED):
    rng = np.random.default_rng(seed)
    pi = np.flatnonzero(y == 1)
    ni = np.flatnonzero(y == 0)
    k = min(len(ni), NEG_PER_POS * max(len(pi), 1))
    ni = rng.choice(ni, size=k, replace=False)
    idx = np.concatenate([pi, ni])
    rng.shuffle(idx)
    return X[idx], y[idx]


# --------------------------------------------------------------------------------------
def eval_competition(scorer, comp: pl.DataFrame, tag: str, res: dict, log=print):
    log(f"\n  ##### competition transfer [{tag}] #####")
    per_fam = {}
    for fam in ("44b6", "6bba"):
        sub = comp.filter(pl.col("family") == fam)
        X, y = xy(sub)
        s = scorer(X)
        mk = (sub["crop"] + ":" + sub["mother"].cast(str)).to_numpy()
        roc, pr = auc_pr(s, y)
        t1, nt1 = top1(s, y, mk)
        taus = np.quantile(s, np.linspace(0.5, 0.999999, 400))
        (tau, j, tp, fp), sweep = division_jaccard(s, y, mk, N_GT_DIV[fam], taus)
        base = float(np.mean(y)) if len(y) else 0.0
        log(f"   {fam}: n={len(y):,} pos={int(y.sum())} prevalence={base:.2e} "
            f"ROC-AUC={roc:.4f} PR-AUC={pr:.4f} (lift {pr / max(base, 1e-12):.0f}x) "
            f"top1={t1:.3f} of {nt1} answerable mothers")
        log(f"        best-case divJ={j:.4f} (TP{tp}/FP{fp}) vs E0c {E0C_DIV_J[fam]:.4f} "
            f"and H0c oracle {H0C_ORACLE_DIV_J[fam]:.4f}")
        per_fam[fam] = {"n": len(y), "pos": int(y.sum()), "prevalence": base,
                        "roc_auc": roc, "pr_auc": pr, "pr_lift": pr / max(base, 1e-12),
                        "top1": t1, "n_answerable": nt1,
                        "best_tau": tau, "best_div_j": j, "tp": tp, "fp": fp,
                        "sweep": sweep, "scores": s, "y": y, "mother_key": mk}
    # ---- cross-fitted threshold: pick tau on one family, apply to the other -----------
    cross = {}
    for fit_fam, test_fam in (("6bba", "44b6"), ("44b6", "6bba")):
        tau = per_fam[fit_fam]["best_tau"]
        p = per_fam[test_fam]
        _, sw = division_jaccard(p["scores"], p["y"], p["mother_key"],
                                 N_GT_DIV[test_fam], [tau])
        tau_, j, tp, fp = sw[0]
        cross[test_fam] = {"tau_from": fit_fam, "tau": tau_, "div_j": j, "tp": tp, "fp": fp,
                           "delta_vs_e0c": j - E0C_DIV_J[test_fam]}
        log(f"   CROSS-FIT tau from {fit_fam} -> {test_fam}: divJ={j:.4f} (TP{tp}/FP{fp}) "
            f"delta vs E0c {j - E0C_DIV_J[test_fam]:+.4f}")
    signs = [per_fam[f]["roc_auc"] > 0.5 for f in ("44b6", "6bba")]
    log(f"   SIGN CONSISTENCY: ROC-AUC>0.5 in both families = {all(signs)} "
        f"({per_fam['44b6']['roc_auc']:.4f} / {per_fam['6bba']['roc_auc']:.4f})")
    res[tag] = {"per_family": {f: {k: v for k, v in per_fam[f].items()
                                   if k not in ("scores", "y", "mother_key", "sweep")}
                               for f in per_fam},
                "cross_fit": cross, "sign_consistent": bool(all(signs))}
    return per_fam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loeo", action="store_true")
    ap.add_argument("--deploy", action="store_true")
    a = ap.parse_args()

    logf = open(SCRATCH / "h1t_critic.log", "a", encoding="utf-8")

    def log(*m):
        s = " ".join(str(x) for x in m)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    ext = load_external()
    ext_rel = ext.filter(pl.col("label").is_in(["positive", "reliable_negative"]))
    embryos = sorted(ext_rel["embryo"].unique().to_list())
    log(f"external: {ext.height:,} rows, reliable {ext_rel.height:,}, "
        f"positives {int((ext_rel['label'] == 'positive').sum()):,}, embryos {embryos}")

    comp = load_competition()
    log(f"competition metric-visible: {comp.height:,} rows, "
        f"positives {int((comp['label'] == 'positive').sum())}, "
        f"reliable_negatives {int((comp['label'] == 'reliable_negative').sum())}")
    comp = comp.filter(pl.col("label").is_in(["positive", "reliable_negative"]))

    res: dict = {"feats": FEATS, "n_external_rows": ext.height, "embryos": embryos}

    # ---- external LOEO -------------------------------------------------------------
    if a.loeo:
        log("\n===== external leave-one-embryo-out =====")
        loeo = {}
        for held in embryos:
            tr = ext_rel.filter(pl.col("embryo") != held)
            te = ext_rel.filter(pl.col("embryo") == held)
            inner = sorted(tr["embryo"].unique().to_list())[-1]
            last_w = tr.filter(pl.col("embryo") == inner)["t_lo"].max()
            va = tr.filter((pl.col("embryo") == inner) & (pl.col("t_lo") == last_w))
            rest = tr.filter(~((pl.col("embryo") == inner) & (pl.col("t_lo") == last_w)))
            if rest.height == 0:            # degenerate: only one training window exists
                rest, va = tr.head(int(0.8 * tr.height)), tr.tail(int(0.2 * tr.height))
            tr = rest
            Xtr, ytr = xy(tr)
            Xtr, ytr = balance(Xtr, ytr)
            Xv, yv = xy(va)
            mv = (va["embryo"] + ":" + va["t_lo"].cast(str) + ":"
                  + va["mother"].cast(str)).to_numpy()
            ngtv = int((va["label"] == "positive").sum())
            c = Critic(len(FEATS))
            log(f"   held-out {held}: train {len(ytr):,} ({int(ytr.sum())} pos) "
                f"inner-val {inner}@t{last_w} ({ngtv} pos) params={c.n_params}")
            bj, bep = c.fit(Xtr, ytr, (Xv, yv, mv, ngtv), log=log)
            Xte, yte = xy(te)
            s = c.score(Xte)
            mk = (te["embryo"] + ":" + te["t_lo"].cast(str) + ":"
                  + te["mother"].cast(str)).to_numpy()
            roc, pr = auc_pr(s, yte)
            t1, nt1 = top1(s, yte, mk)
            ngt = int(yte.sum())
            taus = np.quantile(s, np.linspace(0.5, 0.999999, 300))
            (tau, j, tp, fp), _ = division_jaccard(s, yte, mk, ngt, taus)
            log(f"      -> {held}: ROC-AUC={roc:.4f} PR-AUC={pr:.4f} "
                f"(prev {yte.mean():.2e}) top1={t1:.3f}/{nt1} divJ={j:.4f} TP{tp}/FP{fp}")
            loeo[held] = {"roc_auc": roc, "pr_auc": pr, "prevalence": float(yte.mean()),
                          "top1": t1, "n_answerable": nt1, "div_j": j, "tp": tp, "fp": fp,
                          "inner_val_best_div_j": bj, "best_epoch": bep,
                          "n_params": c.n_params}
        res["external_loeo"] = loeo
        log(f"   external LOEO mean ROC-AUC="
            f"{np.mean([v['roc_auc'] for v in loeo.values()]):.4f} "
            f"PR-AUC={np.mean([v['pr_auc'] for v in loeo.values()]):.4f}")

    # ---- deployment model: all external embryos, then competition transfer ----------
    if a.deploy:
        log("\n===== deployment critic (all external embryos) =====")
        inner = embryos[-1]
        last_w = ext_rel.filter(pl.col("embryo") == inner)["t_lo"].max()
        va = ext_rel.filter((pl.col("embryo") == inner) & (pl.col("t_lo") == last_w))
        tr = ext_rel.filter(~((pl.col("embryo") == inner) & (pl.col("t_lo") == last_w)))
        Xtr, ytr = xy(tr)
        Xtr, ytr = balance(Xtr, ytr)
        Xv, yv = xy(va)
        mv = (va["embryo"] + ":" + va["t_lo"].cast(str) + ":" + va["mother"].cast(str)).to_numpy()
        ngtv = int((va["label"] == "positive").sum())
        c = Critic(len(FEATS))
        log(f"   train {len(ytr):,} ({int(ytr.sum())} pos) inner-val {inner}@t{last_w} "
            f"({ngtv} pos) params={c.n_params}")
        bj, bep = c.fit(Xtr, ytr, (Xv, yv, mv, ngtv), log=log)
        res["deploy_inner_val_div_j"] = bj
        res["deploy_best_epoch"] = bep
        res["n_params"] = c.n_params
        np.savez(SCRATCH / "h1t_critic_weights.npz", mu=c.mu, sd=c.sd,
                 **{k: v.numpy() for k, v in c.net.state_dict().items()})

        eval_competition(c.score, comp, "external_mlp", res, log=log)

        # controls -------------------------------------------------------------------
        eval_competition(lambda X: -X[:, FEATS.index("flow_midpoint_residual")],
                         comp, "control_frozen_residual", res, log=log)
        from sklearn.linear_model import LogisticRegression
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit((Xtr - mu) / sd, ytr)
        eval_competition(lambda X: lr.decision_function((X - mu) / sd),
                         comp, "control_external_linear", res, log=log)
        import lightgbm as lgb
        gb = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                min_child_samples=50, verbose=-1, random_state=SEED)
        gb.fit(Xtr, ytr)
        eval_competition(lambda X: gb.predict_proba(X)[:, 1],
                         comp, "control_external_gbdt", res, log=log)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    OUT.write_text(json.dumps(clean(res), indent=2, default=float))
    log(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
