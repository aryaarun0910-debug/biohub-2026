"""Does OUTPUT_MIN_TRACK_LEN 6 -> 9 survive the both-sets rule?

On the scored films it is +0.00772 with J RISING (+0.00293) -- which is NOT the
node-count exploit signature and is why it looked like a candidate. On the
validator films scripts/100 had it at +0.00041 with J FALLING (-0.00324), which
IS the signature.

So the two sets disagree about the MECHANISM, not just the size. Settle it before
spending a GPU slot: report proxy, J and ratio for both sets on the s05 chain.

A candidate must (a) gain proxy on BOTH sets and (b) not do it by dropping J on
either. Section 5 records that this stage's multiplier gain is monotone to L=40,
so the prior is bad.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from concurrent.futures import ProcessPoolExecutor
exec((ROOT / "scripts/91_other_stages.py").read_text().split("\nALL = []")[0])

TEST_PRED = ROOT / ("artifacts/s05_output/tracking_repo/predictions/unknown/"
                    "unet_transformer/split_0")
SCORED = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
SETS = {"scored": {s: (load_pred(TEST_PRED / f"{s}.geff"), load_gt(s)) for s in SCORED},
        "validator": DATA}


def chain(G, P, L):
    G, _ = gap_close(G, max_um=5.0, reuse_um=3.2, allow_synth=True)
    G, _ = gap2(G, max_total=10.2, max_step=4.4)
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    G = dict(G, edges=safe_div(Q)[0])
    G, _ = prune_isolated(G)
    G, _ = short_track(G, L, True)
    G, _ = linefit(G, w=0.8, window=2)
    return G


def job(a):
    setname, L = a
    rows = []
    for stem, (P, GT) in SETS[setname].items():
        G = chain(G_of(P), P, L)
        rows.append(M2.score(G["t"], G["zyx"], G["edges"],
                             GT["t"], GT["zyx"], GT["edges"], GT["n_est"]))
    r = M2.aggregate(rows)
    r["ratio"] = sum(x["n_pred"] for x in rows) / sum(x["n_est"] for x in rows)
    return setname, L, r


def main():
    LS = (4, 6, 7, 8, 9, 10, 12)
    with ProcessPoolExecutor(max_workers=14) as ex:
        out = list(ex.map(job, [(s, L) for s in SETS for L in LS], chunksize=1))
    R = {(s, L): r for s, L, r in out}

    print(__doc__.strip())
    print("=" * 92)
    print(f"{'L':>4} | {'scored dproxy':>14}{'dJ':>10}{'dratio':>9} | "
          f"{'val dproxy':>12}{'dJ':>10}{'dratio':>9}   verdict")
    for L in LS:
        s, v = R["scored", L], R["validator", L]
        bs, bv = R["scored", 6], R["validator", 6]
        dps, djs, drs = s["proxy"]-bs["proxy"], s["J"]-bs["J"], s["ratio"]-bs["ratio"]
        dpv, djv, drv = v["proxy"]-bv["proxy"], v["J"]-bv["J"], v["ratio"]-bv["ratio"]
        if L == 6:
            verdict = "deployed"
        elif dps > 0 and dpv > 0 and djs > 0 and djv > 0:
            verdict = "*** CANDIDATE: proxy and J up on BOTH ***"
        elif dps > 0 and dpv > 0:
            verdict = "proxy up on both, but J FALLS on " + (
                "validator" if djv <= 0 else "scored") + " -> node-count exploit"
        else:
            verdict = "fails the both-sets rule"
        print(f"{L:>4} | {dps:>+14.5f}{djs:>+10.5f}{drs:>+9.5f} | "
              f"{dpv:>+12.5f}{djv:>+10.5f}{drv:>+9.5f}   {verdict}")


if __name__ == "__main__":
    main()
