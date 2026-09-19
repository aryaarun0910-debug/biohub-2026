"""Can we REMOVE wrong edges? The one direction this pipeline has never tried.

scripts/110 (error budget) found false-positive edges are the single largest
error class -- 184 of 371 errors on the validator films, oracle ceiling +0.03000
-- and that NOTHING in the chain targets them. gap_close, gap2 and safe_div all
ADD edges. Every submission in the ledger pushes the same direction.

The prediction .geff carries a probability per ILP-selected edge. So a pruning
stage is possible: drop edges below a threshold.

DIAGNOSTIC FIRST, because a stage that cannot work should not be built. For every
predicted edge whose BOTH endpoints match a GT node, the edge is decidably TP or
FP. If edge probability does not separate those two populations, pruning is dead
on arrival and no threshold will save it. Only if it separates does the sweep
mean anything.

Then the sweep, under the both-sets rule: a threshold must gain on BOTH the 4
scored films and the 8 validator films, and must not gain by dropping J.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from concurrent.futures import ProcessPoolExecutor

import numpy as np

exec((ROOT / "scripts/91_other_stages.py").read_text().split("\nALL = []")[0])

TEST_PRED = ROOT / ("artifacts/s05_output/tracking_repo/predictions/unknown/"
                    "unet_transformer/split_0")
SCORED = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
SETS = {"scored": {s: (load_pred(TEST_PRED / f"{s}.geff"), load_gt(s)) for s in SCORED},
        "validator": DATA}


def classify(P, GT):
    """TP/FP exactly as metric2.score counts them (metric2.py lines 51-57).

    NOTE: an edge is FP when EITHER endpoint matches a GT node that ought to
    have an edge in that direction -- typically a labelled cell linked to an
    UNLABELLED node instead of its true successor. Requiring both endpoints to
    match (the obvious first guess) finds only 5 of 184 and makes pruning look
    untestable. Use the metric's own definition or the diagnostic is meaningless.
    """
    p2g, _ = M2.match(P["t"], P["zyx"], GT["t"], GT["zyx"])
    gt_set = {(int(a), int(b)) for a, b in GT["edges"]}
    gt_out, gt_in = set(), set()
    for a, b in GT["edges"]:
        gt_out.add(int(a)); gt_in.add(int(b))
    pr = np.asarray(P["prob"], float)
    tp, fp = [], []
    for k, (s, d) in enumerate(P["edges"]):
        ms, mt = p2g.get(int(s)), p2g.get(int(d))
        if ms is not None and mt is not None and (ms, mt) in gt_set:
            tp.append(pr[k])
        elif (mt is not None and mt in gt_in) or (ms is not None and ms in gt_out):
            fp.append(pr[k])
    return np.array(tp), np.array(fp)


def auc(pos, neg):
    """P(a random TP scores above a random FP). 0.5 = no signal."""
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def prune(G, P, thr):
    if thr <= 0:
        return G
    pr = {(int(a), int(b)): p for (a, b), p in zip(P["edges"], P["prob"])}
    return dict(G, edges=[e for e in G["edges"]
                          if pr.get((int(e[0]), int(e[1])), 1.0) >= thr])


def chain(G, P, thr):
    G = prune(G, P, thr)
    G, _ = gap_close(G, max_um=5.0, reuse_um=3.2, allow_synth=True)
    G, _ = gap2(G, max_total=10.2, max_step=4.4)
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    G = dict(G, edges=safe_div(Q)[0])
    G, _ = prune_isolated(G)
    G, _ = short_track(G, 6, True)
    G, _ = linefit(G, w=0.6, window=2)
    return G


def job(a):
    setname, thr = a
    rows = []
    for stem, (P, GT) in SETS[setname].items():
        G = chain(G_of(P), P, thr)
        rows.append(M2.score(G["t"], G["zyx"], G["edges"],
                             GT["t"], GT["zyx"], GT["edges"], GT["n_est"]))
    r = M2.aggregate(rows)
    r["ratio"] = sum(x["n_pred"] for x in rows) / sum(x["n_est"] for x in rows)
    return setname, thr, r


THRS = (0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def main():
    print(__doc__.split("Then the sweep")[0].rstrip())
    print("=" * 96)
    print("--- DIAGNOSTIC: does edge probability separate TP from FP? ---")
    print(f"{'set':<11}{'film':<18}{'nTP':>7}{'nFP':>6}{'TP med':>9}{'FP med':>9}{'AUC':>8}")
    ALL = {}
    for sn, data in SETS.items():
        atp, afp = [], []
        for stem, (P, GT) in data.items():
            tp, fp = classify(P, GT)
            atp.append(tp); afp.append(fp)
            print(f"{sn:<11}{stem:<18}{len(tp):>7}{len(fp):>6}"
                  f"{np.median(tp) if len(tp) else float('nan'):>9.4f}"
                  f"{np.median(fp) if len(fp) else float('nan'):>9.4f}"
                  f"{auc(tp, fp):>8.3f}")
        tp, fp = np.concatenate(atp), np.concatenate(afp)
        ALL[sn] = (tp, fp)
        print(f"{sn:<11}{'POOLED':<18}{len(tp):>7}{len(fp):>6}"
              f"{np.median(tp):>9.4f}{np.median(fp):>9.4f}{auc(tp, fp):>8.3f}")
    a_val = auc(*ALL["validator"])
    print(f"\npooled AUC -- validator {a_val:.3f}, scored {auc(*ALL['scored']):.3f}")
    if a_val < 0.60:
        print("=> WEAK SEPARATION. Pruning by probability is near chance; expect the")
        print("   sweep to remove TP and FP in similar proportion and gain nothing.")
    else:
        print("=> REAL SEPARATION. A threshold should remove FP faster than TP.")

    # WHY a good discriminator still loses: the base rate. Removing an FP gains
    # 1 (fp-1). Removing a TP costs ~2 (tp-1 AND fn+1, since that GT edge becomes
    # unmatched). With ~28 TP per FP, a threshold that catches half the FPs
    # sweeps up far more TPs. Quantify it before the empirical sweep.
    tp, fp = ALL["validator"]
    print("\n--- WHY: the base rate, on the validator films ---")
    print(f"    {len(tp)} TP against {len(fp)} FP -- {len(tp)/len(fp):.0f}:1")
    print(f"{'thr':>6}{'TP cut':>9}{'FP cut':>8}{'TP per FP':>11}{'net J units':>13}  (need > 0)")
    for t in (0.55, 0.65, 0.75, 0.83, 0.90):
        ct, cf = int((tp < t).sum()), int((fp < t).sum())
        net = cf - 2 * ct          # +1 per FP removed, -2 per TP removed
        print(f"{t:>6.2f}{ct:>9}{cf:>8}{(ct/max(cf,1)):>11.1f}{net:>13}")
    print("    even at the FP median (0.83) the cut is dominated by TP loss.")

    print("\n--- SWEEP: prune edges below threshold, both tiers, both-sets rule ---")
    with ProcessPoolExecutor(max_workers=14) as ex:
        out = list(ex.map(job, [(s, t) for s in SETS for t in THRS], chunksize=1))
    R = {(s, t): r for s, t, r in out}
    b = {s: R[s, 0.0] for s in SETS}
    print(f"{'thr':>6} | {'scored dprox':>13}{'dJ':>10}{'dratio':>9} | "
          f"{'val dprox':>11}{'dJ':>10}{'dratio':>9}   verdict")
    for t in THRS[1:]:
        s, v = R["scored", t], R["validator", t]
        dps, djs, drs = (s["proxy"]-b["scored"]["proxy"], s["J"]-b["scored"]["J"],
                         s["ratio"]-b["scored"]["ratio"])
        dpv, djv, drv = (v["proxy"]-b["validator"]["proxy"], v["J"]-b["validator"]["J"],
                         v["ratio"]-b["validator"]["ratio"])
        if dps > 1e-4 and dpv > 1e-4 and djs > 0 and djv > 0:
            verdict = "*** CANDIDATE: proxy and J up on BOTH ***"
        elif dps > 1e-4 and dpv > 1e-4:
            verdict = "proxy up on both but J falls -> node-count exploit"
        else:
            verdict = "fails the both-sets rule"
        print(f"{t:>6.2f} | {dps:>+13.5f}{djs:>+10.5f}{drs:>+9.5f} | "
              f"{dpv:>+11.5f}{djv:>+10.5f}{drv:>+9.5f}   {verdict}")


if __name__ == "__main__":
    main()
