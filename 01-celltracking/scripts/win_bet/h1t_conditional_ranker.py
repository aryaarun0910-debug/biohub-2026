"""H1-T lane D — is the external critic a usable SECOND factor, P(pair | mother divides)?

The H1-T RED verdict was about the FIRST factor: as a global mother-level detector it is
useless (of the top 100 mothers by score, zero are true dividers, in both families). That
says nothing about the conditional pair ranker inside a factorised

    P(mother divides) x P(candidate pair | mother divides)

architecture. This script measures exactly the second factor and nothing else.

Conditioning set: mothers that genuinely divide AND whose true daughter pair survived into
the frozen top-3 shortlist (cfg hash 04eeac97500d). 16 in 44b6, 76 in 6bba. Within each such
mother the true pair is ranked against that mother's own choice set (<= 3 pairs). Ranks are
CALIBRATION-FREE: no global threshold is applied anywhere, which is the question this lane is
explicitly not asking.

Incumbent = the frozen geometric ranker, i.e. ascending flow-midpoint residual. That ordering
is already materialised as the census `rank` column, so the incumbent is read verbatim from
the frozen artefact rather than recomputed.

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\h1t_conditional_ranker.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from h1t_external_critic import FEATS, NEG_PER_POS, SEED, balance  # noqa: E402
from phaseb_h0c_replay import CFG_HASH  # noqa: E402

AGENT4 = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH\agent_runs\agent4")
LANED = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH\agent_runs\laneD")
COMP = ROOT / "artifacts/kaggle/e0c_cache/fork_candidates/h1g_features"
WEIGHTS = AGENT4 / "h1t_critic_weights.npz"
EVENTS = AGENT4 / "external_events"
OUT = LANED / "laneD_conditional_ranks.json"

FAMS = ("44b6", "6bba")


# ---------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval — correct at the tiny n / near-1 rates this lane lives at."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _binom_tail(n: int, k: int) -> float:
    """P(X <= k) for X ~ Binom(n, 0.5)."""
    return sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n) if n else 1.0


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar. b = A-only wins, c = B-only wins. Paired, same mothers."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * _binom_tail(n, k)
    return float(min(1.0, p))


def paired_boot_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 20000, seed: int = 0):
    """Percentile bootstrap CI for mean(a) - mean(b), resampling MOTHERS (pairs kept)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


# ---------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------
def load_conditioning() -> pl.DataFrame:
    """Every candidate belonging to a mother whose TRUE pair is in the frozen shortlist.

    Two passes so the 14.37M-row census is never materialised: pass 1 pulls the 92 positive
    rows only, pass 2 pushes the resulting (crop, mother) predicate down into the scan.
    """
    src = str(COMP / "*" / "*.parquet")
    pos = (pl.scan_parquet(src).filter(pl.col("label") == "positive")
           .select("crop", "mother").collect())
    crops = pos["crop"].unique().to_list()
    mks = set(zip(pos["crop"].to_list(), pos["mother"].to_list()))
    cond = (pl.scan_parquet(src)
            .filter(pl.col("crop").is_in(crops))
            .collect())
    cond = cond.filter(
        pl.struct(["crop", "mother"]).map_elements(
            lambda r: (r["crop"], r["mother"]) in mks, return_dtype=pl.Boolean))
    cond = cond.with_columns((pl.col("crop") + ":" + pl.col("mother").cast(str)).alias("mk"))
    return cond.with_columns(pl.col("steal_required").cast(pl.Int64))


def load_external() -> pl.DataFrame:
    files = sorted(EVENTS.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no external events in {EVENTS}")
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    return df.with_columns(pl.col("steal_required").cast(pl.Int64))


def mlp_scorer():
    """Frozen H1-T deployment critic — weights loaded, never refit."""
    z = np.load(WEIGHTS)
    mu, sd = z["mu"], z["sd"]
    W = [(z["0.weight"], z["0.bias"]), (z["2.weight"], z["2.bias"]), (z["4.weight"], z["4.bias"])]

    def f(X: np.ndarray) -> np.ndarray:
        h = (X - mu) / sd
        for i, (w, b) in enumerate(W):
            h = h @ w.T + b
            if i < len(W) - 1:
                h = np.maximum(h, 0.0)
        return h[:, 0]

    return f


def external_controls(log):
    """Linear + GBDT controls, refit on EXTERNAL data only, same recipe/seed as H1-T."""
    ext = load_external().filter(pl.col("label").is_in(["positive", "reliable_negative"]))
    embryos = sorted(ext["embryo"].unique().to_list())
    inner = embryos[-1]
    last_w = ext.filter(pl.col("embryo") == inner)["t_lo"].max()
    tr = ext.filter(~((pl.col("embryo") == inner) & (pl.col("t_lo") == last_w)))
    Xtr = tr.select(FEATS).to_numpy().astype(np.float32)
    ytr = (tr["label"] == "positive").to_numpy().astype(np.float32)
    Xtr, ytr = balance(Xtr, ytr)
    log(f"  external control fit: {len(ytr):,} rows ({int(ytr.sum())} pos), "
        f"neg_per_pos={NEG_PER_POS}, seed={SEED}")

    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit((Xtr - mu) / sd, ytr)

    import lightgbm as lgb
    gb = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                            min_child_samples=50, verbose=-1, random_state=SEED)
    gb.fit(Xtr, ytr)
    return (lambda X: lr.decision_function((X - mu) / sd),
            lambda X: gb.predict_proba(X)[:, 1])


# ---------------------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------------------
def true_pair_ranks(sub: pl.DataFrame, scores: np.ndarray | None) -> dict[str, np.ndarray]:
    """Per mother, 1-based rank of the true pair inside that mother's own choice set.

    scores=None -> use the FROZEN census `rank` column verbatim (the incumbent as deployed).
    Otherwise sort by descending score; ties are broken by the frozen rank, so a scorer that
    is uninformative degrades to geometry rather than to chance.
    """
    frozen = sub["rank"].to_numpy()
    y = (sub["label"] == "positive").to_numpy()
    mk = sub["mk"].to_numpy()
    order = np.argsort(mk, kind="stable")
    ranks, sizes, keys = [], [], []
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and mk[order[j]] == mk[order[i]]:
            j += 1
        ii = order[i:j]
        if scores is None:
            key = frozen[ii].astype(np.float64)
        else:
            key = -scores[ii].astype(np.float64) * 1e6 + frozen[ii] * 1e-6
        o = np.argsort(key, kind="stable")
        pos_at = int(np.flatnonzero(y[ii][o])[0])
        ranks.append(pos_at + 1)
        sizes.append(len(ii))
        keys.append(mk[order[i]])
        i = j
    return {"rank": np.asarray(ranks), "size": np.asarray(sizes), "mk": np.asarray(keys)}


def summarise(r: np.ndarray, tag: str) -> dict:
    n = len(r)
    out = {"tag": tag, "n": n, "mean_rank": float(r.mean()) if n else float("nan"),
           "mrr": float((1.0 / r).mean()) if n else float("nan")}
    for k in (1, 2, 3):
        hit = int((r <= k).sum())
        lo, hi = wilson(hit, n)
        out[f"rank_at_{k}"] = {"hit": hit, "n": n, "rate": hit / n if n else float("nan"),
                               "ci95": [lo, hi]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    LANED.mkdir(parents=True, exist_ok=True)
    logf = open(LANED / "laneD.log", "a", encoding="utf-8")

    def log(*m):
        s = " ".join(str(x) for x in m)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"\n===== lane D: H1-T as P(pair | mother divides), frozen cfg {CFG_HASH} =====")
    cond = load_conditioning()
    log(f"conditioning candidates: {cond.height} rows across "
        f"{cond['mk'].n_unique()} dividing mothers with the true pair in the shortlist")

    scorers = {"frozen_geometry": None, "h1t_mlp": mlp_scorer()}
    lin, gbdt = external_controls(log)
    scorers["h1t_linear"] = lin
    scorers["h1t_gbdt"] = gbdt
    # negated residual == the frozen ordering, kept as an explicit sanity control
    scorers["residual_as_score"] = lambda X: -X[:, FEATS.index("flow_midpoint_residual")]

    res: dict = {"config_hash": CFG_HASH, "families": {}}
    for fam in FAMS:
        sub = cond.filter(pl.col("family") == fam)
        X = sub.select(FEATS).to_numpy().astype(np.float32)
        log(f"\n########## {fam} ##########")
        log(f"  {sub['mk'].n_unique()} dividing mothers, {sub.height} candidate pairs")
        per = {}
        for name, sc in scorers.items():
            s = None if sc is None else sc(X)
            per[name] = true_pair_ranks(sub, s)
        sizes = per["frozen_geometry"]["size"]
        nontrivial = sizes >= 2
        log(f"  choice-set sizes: "
            f"{dict(zip(*[x.tolist() for x in np.unique(sizes, return_counts=True)]))} "
            f"-> {int(nontrivial.sum())} mothers have a real choice")

        fam_out = {"n_mothers": int(len(sizes)),
                   "n_nontrivial": int(nontrivial.sum()),
                   "choice_set_sizes": {int(k): int(v) for k, v in
                                        zip(*np.unique(sizes, return_counts=True))},
                   "all": {}, "nontrivial": {}, "paired_vs_geometry": {}}
        for name, d in per.items():
            fam_out["all"][name] = summarise(d["rank"], name)
            fam_out["nontrivial"][name] = summarise(d["rank"][nontrivial], name)
            s = fam_out["all"][name]
            lo, hi = s["rank_at_1"]["ci95"]
            log(f"    {name:20s} rank@1 {s['rank_at_1']['hit']:3d}/{s['n']:3d} "
                f"= {s['rank_at_1']['rate']:.3f} [{lo:.3f},{hi:.3f}]  "
                f"rank@2 {s['rank_at_2']['hit']:3d}/{s['n']:3d}  "
                f"rank@3 {s['rank_at_3']['hit']:3d}/{s['n']:3d}  "
                f"MRR {s['mrr']:.4f}  mean_rank {s['mean_rank']:.3f}")

        # paired comparisons against the incumbent, on identical mothers
        g = per["frozen_geometry"]["rank"]
        for name in ("h1t_mlp", "h1t_linear", "h1t_gbdt"):
            h = per[name]["rank"]
            a1 = (h <= 1).astype(float)
            g1 = (g <= 1).astype(float)
            b = int(((a1 == 1) & (g1 == 0)).sum())
            c = int(((a1 == 0) & (g1 == 1)).sum())
            p = mcnemar_exact(b, c)
            lo, hi = paired_boot_ci(a1, g1)
            fam_out["paired_vs_geometry"][name] = {
                "rank1_delta": float(a1.mean() - g1.mean()),
                "boot_ci95": [lo, hi],
                "mcnemar_b_critic_only": b, "mcnemar_c_geometry_only": c,
                "mcnemar_p_two_sided": p,
                "mean_rank_delta": float(h.mean() - g.mean()),
                "mrr_delta": float((1.0 / h).mean() - (1.0 / g).mean()),
            }
            log(f"    PAIRED {name} vs frozen_geometry: rank@1 delta "
                f"{a1.mean() - g1.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  "
                f"McNemar b={b} c={c} p={p:.4f}  "
                f"mean_rank {h.mean() - g.mean():+.3f}  MRR {(1/h).mean() - (1/g).mean():+.4f}")

        # per-mother disagreement detail, so a 2-3 event swing can be inspected not guessed
        dis = []
        for i in range(len(g)):
            if per["h1t_mlp"]["rank"][i] != g[i]:
                dis.append({"mk": str(per["h1t_mlp"]["mk"][i]),
                            "size": int(sizes[i]),
                            "geometry_rank": int(g[i]),
                            "h1t_mlp_rank": int(per["h1t_mlp"]["rank"][i])})
        fam_out["mlp_vs_geometry_disagreements"] = dis
        log(f"    mothers where H1-T MLP and geometry disagree on the true pair's rank: "
            f"{len(dis)}")
        for d in dis:
            log(f"      {d['mk']:34s} size={d['size']} geom_rank={d['geometry_rank']} "
                f"h1t_rank={d['h1t_mlp_rank']}")
        res["families"][fam] = fam_out

    # pooled (supplementary; families are the reportable unit)
    pooled = {}
    for name in scorers:
        hits = sum(res["families"][f]["all"][name]["rank_at_1"]["hit"] for f in FAMS)
        n = sum(res["families"][f]["all"][name]["n"] for f in FAMS)
        lo, hi = wilson(hits, n)
        pooled[name] = {"rank_at_1": {"hit": hits, "n": n, "rate": hits / n, "ci95": [lo, hi]}}
        log(f"  POOLED {name:20s} rank@1 {hits}/{n} = {hits / n:.3f} [{lo:.3f},{hi:.3f}]")
    res["pooled"] = pooled

    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
