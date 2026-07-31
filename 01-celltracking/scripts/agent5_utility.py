"""Agent 5 / Subtrack B — EXPECTED POOLED UTILITY instead of AUC.

For every candidate mother/daughter-pair action this estimates

    E[dPooled] =   expected division-TP gain
                 - expected division-FP cost
                 - expected ordinary-edge loss from parent stealing
                 - expected edge-FP cost
    act iff  E[dPooled] > 0

and then measures whether ordering by that utility beats ordering by probability.

--------------------------------------------------------------------------------------
EXACT POOLED EVALUATOR (proved by scripts/agent5_ledger.py --verify, err 5.7e-14)
--------------------------------------------------------------------------------------
    pooled = NUM / DEN + 0.1 * DTP / (DTP + DFP + DFN)
    NUM = SUM_i tp_i * (1 - 0.1 * r_i)      DEN = SUM_i (tp_i + fp_i + fn_i)
so the marginal value of one counter at the current operating point is

    dS/dNUM = 1/DEN            dS/dDEN = -NUM/DEN**2
    dS/dDTP = 0.1/D_div        (a recovered division moves FN->TP, D_div constant)
    dS/dDFP = -0.1*DTP/D_div**2

Those four partials ARE the utility weights.  Nothing here is fitted to a Jaccard
approximation; the arithmetic is the organiser's scorer.

--------------------------------------------------------------------------------------
DEPLOYABILITY RULES OBSERVED
--------------------------------------------------------------------------------------
* Cross-fit is ACROSS EMBRYO FAMILIES (fold 0 = 44b6, fold 1 = 6bba).  Every reported
  number is out-of-family.  A within-fold split would be the mistake seven earlier
  methods made.
* `metric_visible` (mother matched an annotated GT node) is NOT predicted from features.
  Annotation coverage is a property of the label set, not of the embryo, so learning it
  would be a metric exploit and would not transfer to the private split.  Visibility
  enters utility only through a single global constant `pi_vis`, estimated on the
  training fold.
* Features are geometry/topology only (frozen H1-G set), all deployment-observable.
* The candidate surface is the immutable frozen H0c shortlist (cfg 04eeac97500d).
* NO-REFILTER, like `phaseb_pooled_breakeven.py`.  Deltas are a LOWER BOUND; the live
  wrapper filter (`phaseb_h0d_livefilter.py`) is strictly more favourable.

--------------------------------------------------------------------------------------
OUTPUTS
--------------------------------------------------------------------------------------
utility_calibration.json / .csv    predicted-utility decile vs REALISED pooled delta
cumulative_gain.csv                exact pooled score after greedy conflict-free
                                   admission of the top-n candidates, for
                                   ordering in {probability, utility, residual}
stopping.json                      cross-fit-selected stopping point and the held-out
                                   pooled gain it delivers

USAGE
  .venv\\Scripts\\python.exe scripts\\agent5_utility.py --ledger <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

CACHE = ROOT / "artifacts/kaggle/e0c_cache"
FEATS = CACHE / "fork_candidates" / "h1g_features"
DEFAULT_LEDGER = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026"
                      r"_RESEARCH\agent_runs\diversity\ledger")

# frozen H1-G geometry features; deployment-observable, no annotation-coverage leakage
FEATURES = [
    "flow_midpoint_residual", "rank", "parent_midpoint_um", "pd1_um", "pd2_um", "pd_ratio",
    "sister_um", "cos_daughter_axis", "cos_split_vs_flow", "daughter_angle",
    "persist_d1", "persist_d2", "n_persist", "mother_track_age", "mother_speed_um",
    "vel_consistency", "local_density_t", "local_density_t1", "competing_parents",
    "steal_required", "bdist_um", "resid_gap_to_best", "resid_ratio", "best_alt_gap",
]


def pooled(num, den, dtp, dfp, dfn):
    """Exact pooled composite from running counters. Scalar or numpy-vectorised."""
    d = np.asarray(dtp, float) + np.asarray(dfp, float) + np.asarray(dfn, float)
    den = np.asarray(den, float)
    edge = np.where(den > 0, np.asarray(num, float) / np.where(den > 0, den, 1.0), np.nan)
    div = np.where(d > 0, 0.1 * np.asarray(dtp, float) / np.where(d > 0, d, 1.0), 0.0)
    r = edge + div
    return float(r) if np.ndim(r) == 0 else r


CAND_COLS = ["crop", "fold", "mother", "d1", "d2", "rank", "label", "metric_visible",
             "e_rem_tp", "d_tp", "d_fp", "div_dtp", "div_dfp", "div_dfn"]
KEY = ["crop", "mother", "d1", "d2"]


def load_ledger(ledger: Path, unlabeled_frac: float, seed: int):
    """Memory-lean load. `cand_id` (a 14M-element string column) is never materialised;
    the join key is the integer tuple (crop, mother, d1, d2), which is unique by
    construction of the frozen census."""
    crops = [json.loads(p.read_text()) for p in sorted((ledger / "crops").glob("*/*.json"))]
    parts = []
    for p in sorted((ledger / "cand").glob("*/*.parquet")):
        c = pl.read_parquet(p, columns=CAND_COLS)
        if unlabeled_frac < 1.0:
            keep = (pl.col("label") != "unlabeled") | pl.col("metric_visible") \
                   | (pl.col("d_tp") != 0) | (pl.col("d_fp") != 0)
            c = c.filter(keep | (pl.int_range(pl.len()).hash(seed) % 1_000_000
                                 < int(unlabeled_frac * 1_000_000)))
        parts.append(c.with_columns([pl.col("crop").cast(pl.Categorical),
                                     pl.col("mother").cast(pl.Int32),
                                     pl.col("d1").cast(pl.Int32),
                                     pl.col("d2").cast(pl.Int32)]))
    return crops, pl.concat(parts)


def anchors(crops):
    """Global suppress-all anchors (NUM, DEN, DTP, DFP, DFN) over the ledger's crops."""
    num = sum(c["supp"]["edge_tp"] * (1 - 0.1 * c["supp"]["total_node_ratio"]) for c in crops)
    den = sum(c["supp"]["edge_tp"] + c["supp"]["edge_fp"] + c["supp"]["edge_fn"]
              for c in crops)
    dtp = sum(c["supp"]["division_tp"] for c in crops)
    dfp = sum(c["supp"]["division_fp"] for c in crops)
    dfn = sum(c["supp"]["division_fn"] for c in crops)
    return float(num), float(den), int(dtp), int(dfp), int(dfn)


def fit_fold(train: pl.DataFrame, test: pl.DataFrame, seed: int):
    """Cross-family fit. Returns (p_hat, e_dtp_hat, e_dfp_hat) for `test`."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    # --- p(true daughter pair | geometry): scorer-reliable labels only ---------------
    rel = train.filter(pl.col("label").is_in(["positive", "reliable_negative"]))
    ytr = (rel["label"] == "positive").to_numpy().astype(int)
    Xtr = rel.select(FEATURES).to_numpy()
    npos = int(ytr.sum())
    if npos < 2 or ytr.sum() == len(ytr):
        p_hat = np.full(test.height, npos / max(len(ytr), 1), dtype=float)
        clf = None
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, class_weight="balanced", random_state=seed)
        clf.fit(Xtr, ytr)
        p_hat = clf.predict_proba(test.select(FEATURES).to_numpy())[:, 1]
        # scale back to the true prior (class_weight="balanced" inflates it)
        pi = npos / len(ytr)
        p_hat = p_hat * pi / (p_hat * pi + (1 - p_hat) * (1 - pi) + 1e-30)

    # --- expected EDGE-side counter deltas: the parent-stealing cost ------------------
    # Realised on the training fold from the exact ledger; regressed on the same
    # deployment-observable geometry.  This is what makes utility != probability.
    Xall = train.select(FEATURES).to_numpy()
    out = []
    for col in ("d_tp", "d_fp"):
        reg = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=100,
            l2_regularization=1.0, random_state=seed)
        reg.fit(Xall, train[col].to_numpy().astype(float))
        out.append(reg.predict(test.select(FEATURES).to_numpy()))
    return p_hat, out[0], out[1], clf


def state_utility(p, e_dtp, e_dfp, wn, pi_vis, N, D, T, F, Fn):
    """Expected pooled delta of ONE action, evaluated by FINITE DIFFERENCE at the current
    operating point (N, D, T, F, Fn) -- not by a derivative at the empty point.

    This matters: at the suppress-all anchor DTP == 0, so the first-order division-FP cost
    is identically zero and a linearised utility would admit unlimited false forks.  The
    finite-difference form carries the real k/(D_div) saturation.
    """
    ddiv = T + F + Fn
    s0 = 0.1 * (T / ddiv if ddiv > 0 else 0.0)
    g_tp = 0.1 * ((T + 1) / ddiv if ddiv > 0 else 0.0) - s0          # FN -> TP
    c_fp = 0.1 * (T / (ddiv + 1) if ddiv + 1 > 0 else 0.0) - s0      # extra FP
    edge = e_dtp * wn / D - e_dfp * N / D ** 2
    return edge + p * g_tp + (1.0 - p) * pi_vis * c_fp


def greedy_exact(df: pl.DataFrame, order_col: str, num, den, dtp, dfp, dfn, wnode_by_crop,
                 max_n=None, descending=True, adaptive=False, pi_vis=0.0,
                 rerank_every=25, shortlist=200_000):
    """Admit candidates in `order_col` order, skipping conflicts, accumulating the EXACT
    realised counter deltas from the ledger. Returns the cumulative pooled trace.

    `adaptive=True` re-scores the shortlist with `state_utility` at the CURRENT operating
    point every `rerank_every` admissions (confidence-adaptive, never a fixed global
    threshold).
    """
    d = df.sort(order_col, descending=descending)
    if adaptive and d.height > shortlist:
        d = d.head(shortlist)
    crop = d["crop"].to_list()
    mo, a1, a2 = d["mother"].to_numpy(), d["d1"].to_numpy(), d["d2"].to_numpy()
    dtp_e, dfp_e = d["d_tp"].to_numpy(), d["d_fp"].to_numpy()
    vdtp, vdfp, vdfn = (d["div_dtp"].to_numpy(), d["div_dfp"].to_numpy(),
                        d["div_dfn"].to_numpy())
    wn_arr = np.array([wnode_by_crop[c] for c in crop])
    if adaptive:
        p_arr, edtp_arr, edfp_arr = (d["p_hat"].to_numpy(), d["e_dtp"].to_numpy(),
                                     d["e_dfp"].to_numpy())
    base = pooled(num, den, dtp, dfp, dfn)
    used_m, used_d = set(), set()
    N, D, T, F, Fn = num, den, dtp, dfp, dfn
    trace, n_admit = [], 0
    lim = max_n if max_n is not None else len(crop)
    order = np.arange(len(crop))
    ptr, since = 0, 0
    while n_admit < lim and ptr < len(order):
        if adaptive and since >= rerank_every:
            u = state_utility(p_arr, edtp_arr, edfp_arr, wn_arr, pi_vis, N, D, T, F, Fn)
            u[order[:ptr]] = -np.inf                      # never revisit
            order = np.argsort(-u, kind="stable")
            ptr, since = 0, 0
        i = int(order[ptr]); ptr += 1
        m, x, y = int(mo[i]), int(a1[i]), int(a2[i])
        km, kx, ky = (crop[i], m), (crop[i], x), (crop[i], y)
        if km in used_m or kx in used_d or ky in used_d or km in used_d or kx in used_m \
           or ky in used_m:
            continue
        used_m.add(km); used_d.add(kx); used_d.add(ky)
        N += float(dtp_e[i]) * wn_arr[i]
        D += float(dfp_e[i])
        T += int(vdtp[i]); F += int(vdfp[i]); Fn += int(vdfn[i])
        n_admit += 1; since += 1
        trace.append((n_admit, pooled(N, D, T, F, Fn) - base, T, F, i))
    return base, trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--max-admit", type=int, default=4000)
    ap.add_argument("--anticipate", type=int, default=-1,
                    help="anticipated recovered divisions k_t for the finite-difference "
                         "utility; -1 = training-fold positive count")
    ap.add_argument("--rerank-every", type=int, default=25)
    ap.add_argument("--shortlist", type=int, default=200_000)
    ap.add_argument("--unlabeled-frac", type=float, default=1.0,
                    help="keep this fraction of INERT unlabeled candidates (rows with no "
                         "label, no visibility and zero counter deltas contribute exactly "
                         "zero to the pooled score). Use <1 only if memory forces it.")
    a = ap.parse_args()
    out_dir = a.ledger.parent
    crops, cand = load_ledger(a.ledger, a.unlabeled_frac, a.seed)
    have = sorted({c["crop"] for c in crops})
    print(f"ledger: {len(crops)} crop(s), {cand.height:,} candidates, "
          f"{int((cand['label']=='positive').sum())} positives, "
          f"{int(cand['metric_visible'].sum()):,} metric-visible")

    NUM, DEN, DTP, DFP, DFN = anchors(crops)
    base = pooled(NUM, DEN, DTP, DFP, DFN)
    print(f"\nSUPPRESS-ALL ANCHOR over these crops: NUM={NUM:.3f} DEN={DEN:.0f} "
          f"div=({DTP},{DFP},{DFN})  pooled={base:.6f}")
    print(f"  partials:  dS/dTPedge={1/DEN:+.3e}  dS/dFPedge={-NUM/DEN**2:+.3e}  "
          f"dS/dTPdiv={0.1/max(DTP+DFP+DFN,1):+.3e}  "
          f"dS/dFPdiv={-0.1*DTP/max(DTP+DFP+DFN,1)**2:+.3e}")
    wnode = {c["crop"]: c["w_node"] for c in crops}

    # ---- join the frozen geometry features ------------------------------------------
    fe = pl.concat([
        pl.read_parquet(FEATS / str(c["fold"]) / f"{c['crop']}.parquet",
                        columns=KEY + FEATURES)
        .with_columns([pl.col("crop").cast(pl.Categorical),
                       pl.col("mother").cast(pl.Int32), pl.col("d1").cast(pl.Int32),
                       pl.col("d2").cast(pl.Int32)]
                      + [pl.col(f).cast(pl.Float32) for f in FEATURES])
        for c in crops])
    df = cand.join(fe, on=KEY, how="inner")
    print(f"joined features: {df.height:,} rows ({df.height/max(cand.height,1):.3%} of ledger)")

    folds = sorted(df["fold"].unique().to_list())
    if len(folds) < 2:
        raise SystemExit("cross-family cross-fit needs both folds in the ledger")

    parts = []
    for te in folds:
        tr = df.filter(pl.col("fold") != te)
        ts = df.filter(pl.col("fold") == te)
        p, edtp, edfp, clf = fit_fold(tr, ts, a.seed)
        pi_vis = float(tr["metric_visible"].mean())          # training-fold constant only
        wn = np.array([wnode[c] for c in ts["crop"].to_list()])
        # anticipated operating point: k_t recovered divisions already admitted. Selected
        # on the TRAINING fold only (expected yield of an equal-size admission there).
        k_t = a.anticipate if a.anticipate >= 0 else int((tr["label"] == "positive").sum())
        Ta, Fa, Fna = DTP + k_t, DFP, max(DFN - k_t, 0)
        util = state_utility(p, edtp, edfp, wn, pi_vis, NUM, DEN, Ta, Fa, Fna)
        # realised solo pooled delta of this single action: the EXACT closed form applied
        # to the ledger's true counters, evaluated at the SAME anticipated point
        realised = (pooled(NUM + ts["d_tp"].to_numpy() * wn, DEN + ts["d_fp"].to_numpy(),
                           Ta + ts["div_dtp"].to_numpy(), Fa + ts["div_dfp"].to_numpy(),
                           Fna + ts["div_dfn"].to_numpy())
                    - pooled(NUM, DEN, Ta, Fa, Fna))
        parts.append(ts.with_columns([
            pl.Series("p_hat", p), pl.Series("u_hat", util),
            pl.Series("e_dtp", edtp), pl.Series("e_dfp", edfp),
            pl.Series("realised_solo", realised), pl.lit(pi_vis).alias("pi_vis"),
        ]))
        print(f"  fold {te} (held out, trained on {[f for f in folds if f!=te]}): "
              f"train pos={int((tr['label']=='positive').sum())} "
              f"pi_vis={pi_vis:.5f} model={'GBM' if clf else 'prior'}")
    S = pl.concat(parts)

    # ---- utility calibration curve ---------------------------------------------------
    S = S.with_columns((pl.col("u_hat").rank("ordinal", descending=True)
                        / pl.len()).alias("_q"))
    bins = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5, 1.0]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = S.filter((pl.col("_q") > lo) & (pl.col("_q") <= hi))
        if not b.height:
            continue
        rows.append({
            "top_frac_lo": lo, "top_frac_hi": hi, "n": b.height,
            "pred_utility_mean": float(b["u_hat"].mean()),
            "realised_utility_mean": float(b["realised_solo"].mean()),
            "pred_p_mean": float(b["p_hat"].mean()),
            "positives": int((b["label"] == "positive").sum()),
            "visible": int(b["metric_visible"].sum()),
            "prec_among_visible": float((b["label"] == "positive").sum()
                                        / max(int(b["metric_visible"].sum()), 1)),
            "frac_destroys_true_edge": float((b["e_rem_tp"] > 0).mean()),
        })
    cal = pl.DataFrame(rows)
    print("\n=== EXPECTED-UTILITY CALIBRATION (out-of-family) ===")
    print(cal.select("top_frac_lo", "top_frac_hi", "n", "pred_utility_mean",
                     "realised_utility_mean", "positives", "visible",
                     "prec_among_visible", "frac_destroys_true_edge"))
    cal.write_csv(out_dir / "utility_calibration.csv")

    # ---- cumulative pooled gain: utility order vs probability order -------------------
    print("\n=== CUMULATIVE POOLED GAIN (exact realised deltas, greedy conflict-free) ===")
    pv = float(S["pi_vis"].mean())
    curves = {}
    for name, col, desc, adap in (("utility", "u_hat", True, False),
                                  ("utility_adaptive", "u_hat", True, True),
                                  ("probability", "p_hat", True, False),
                                  ("residual", "flow_midpoint_residual", False, False)):
        b, tr = greedy_exact(S, col, NUM, DEN, DTP, DFP, DFN, wnode,
                             max_n=a.max_admit, descending=desc, adaptive=adap,
                             pi_vis=pv, rerank_every=a.rerank_every,
                             shortlist=a.shortlist)
        curves[name] = tr
        if tr:
            best = max(tr, key=lambda z: z[1])
            print(f"  {name:<12} peak {best[1]:+.6f} pooled at n={best[0]:,} "
                  f"(divTP={best[2]}, divFP={best[3]});  at n=100 "
                  f"{tr[min(99,len(tr)-1)][1]:+.6f};  final {tr[-1][1]:+.6f}")
    n_max = max(len(v) for v in curves.values())
    grid = sorted({int(x) for x in np.unique(np.geomspace(1, max(n_max, 2), 60).astype(int))})
    cg = pl.DataFrame({"n": grid, **{
        k: [(v[min(n, len(v)) - 1][1] if v else float("nan")) for n in grid]
        for k, v in curves.items()}})
    cg.write_csv(out_dir / "cumulative_gain.csv")

    # ---- cross-fit stopping point -----------------------------------------------------
    stop = {}
    for te in folds:
        tr_side = S.filter(pl.col("fold") != te)
        te_side = S.filter(pl.col("fold") == te)
        _, tr_tr = greedy_exact(tr_side, "u_hat", NUM, DEN, DTP, DFP, DFN, wnode,
                                max_n=a.max_admit)
        if not tr_tr:
            continue
        # stopping rule selected ONLY on the training fold: last n with positive marginal
        n_star = max(tr_tr, key=lambda z: z[1])[0]
        _, te_tr = greedy_exact(te_side, "u_hat", NUM, DEN, DTP, DFP, DFN, wnode,
                                max_n=a.max_admit)
        held = te_tr[min(n_star, len(te_tr)) - 1][1] if te_tr else float("nan")
        _, te_p = greedy_exact(te_side, "p_hat", NUM, DEN, DTP, DFP, DFN, wnode,
                               max_n=a.max_admit)
        held_p = te_p[min(n_star, len(te_p)) - 1][1] if te_p else float("nan")
        stop[str(te)] = {"n_star_from_train": n_star, "heldout_pooled_utility_order": held,
                         "heldout_pooled_probability_order": held_p,
                         "utility_minus_probability": held - held_p}
        print(f"  cross-fit stop: train folds {[f for f in folds if f!=te]} -> "
              f"n*={n_star:,};  held-out fold {te} pooled "
              f"utility {held:+.6f} vs probability {held_p:+.6f} "
              f"({held-held_p:+.6f})")

    (out_dir / "stopping.json").write_text(json.dumps({
        "crops": have, "anchor": {"NUM": NUM, "DEN": DEN, "DTP": DTP, "DFP": DFP,
                                  "DFN": DFN, "pooled": base},
        "partials": {"dS_dTPedge": 1 / DEN, "dS_dFPedge": -NUM / DEN ** 2,
                     "dS_dTPdiv": 0.1 / max(DTP + DFP + DFN, 1),
                     "dS_dFPdiv": -0.1 * DTP / max(DTP + DFP + DFN, 1) ** 2},
        "stopping": stop}, indent=2, default=float))
    print(f"\nwrote {out_dir}\\utility_calibration.csv, cumulative_gain.csv, stopping.json")


if __name__ == "__main__":
    main()
