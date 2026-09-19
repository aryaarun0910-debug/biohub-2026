"""Pick a linefit weight that is positive on BOTH film sets, on the s05 chain.

s09 shipped w=0.3 on the strength of the 8 validator films (+0.00487) and it is
-0.00086 on the 4 scored films. scripts/106 shows the scored films peak at w=0.6.
A weight is only worth shipping if it is positive on BOTH -- the whole lesson of
scripts/102-103 is that either set alone can mislead.

Chain is s05 exactly: relink off, gap2 BEFORE safe_div, everything else deployed.
linefit moves coordinates only, so `ratio` must be identical across every cell;
the script asserts that rather than trusting it.
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
TDATA = {s: (load_pred(TEST_PRED / f"{s}.geff"), load_gt(s)) for s in SCORED}
SETS = {"scored": TDATA, "validator": DATA}


def chain(G, P, w, win):
    G, _ = gap_close(G, max_um=5.0, reuse_um=3.2, allow_synth=True)
    G, _ = gap2(G, max_total=10.2, max_step=4.4)
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    G = dict(G, edges=safe_div(Q)[0])
    G, _ = prune_isolated(G)
    G, _ = short_track(G, 6, True)
    if w > 0:
        G, _ = linefit(G, w=w, window=win)
    return G


def job(a):
    setname, w, win = a
    rows = []
    for stem, (P, GT) in SETS[setname].items():
        G = chain(G_of(P), P, w, win)
        rows.append(M2.score(G["t"], G["zyx"], G["edges"],
                             GT["t"], GT["zyx"], GT["edges"], GT["n_est"]))
    r = M2.aggregate(rows)
    r["ratio"] = sum(x["n_pred"] for x in rows) / sum(x["n_est"] for x in rows)
    return setname, w, win, r


WS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
JOBS = [(s, w, 2) for s in SETS for w in WS]


def main():
    print(__doc__.strip())
    print("=" * 84)
    with ProcessPoolExecutor(max_workers=16) as ex:
        out = list(ex.map(job, JOBS, chunksize=1))
    R = {(s, w): r for s, w, _, r in out}
    for s in SETS:
        rr = [R[s, w]["ratio"] for w in WS]
        assert max(rr) - min(rr) < 1e-12, f"{s}: linefit moved node count! {rr}"
    print("ratio identical across all weights on both sets: PASS "
          "(linefit moves coordinates only)\n")
    b = {s: R[s, 0.8]["proxy"] for s in SETS}
    print(f"deployed w=0.8 -> scored {b['scored']:.5f}   validator {b['validator']:.5f}")
    print(f"\n{'w':>5}{'scored d':>12}{'validator d':>14}   verdict")
    best = None
    for w in WS:
        ds = R["scored", w]["proxy"] - b["scored"]
        dv = R["validator", w]["proxy"] - b["validator"]
        ok = ds > 0 and dv > 0
        mark = "positive on BOTH" if ok else ("scored only" if ds > 0 else
                                             "validator only" if dv > 0 else "")
        star = ""
        if ok and (best is None or min(ds, dv) > best[1]):
            best = (w, min(ds, dv)); star = ""
        print(f"{w:>5.1f}{ds:>+12.5f}{dv:>+14.5f}   {mark}")
    print(f"\nSAFEST PICK = the weight maximising the WORST of the two sets:")
    for w in WS:
        ds = R["scored", w]["proxy"] - b["scored"]
        dv = R["validator", w]["proxy"] - b["validator"]
        if best and w == best[0]:
            print(f"  w={w:g}: scored {ds:+.5f}, validator {dv:+.5f}, "
                  f"worst case {min(ds, dv):+.5f}")
    print(f"\n(for contrast, s09 shipped w=0.3: scored "
          f"{R['scored', 0.3]['proxy'] - b['scored']:+.5f}, validator "
          f"{R['validator', 0.3]['proxy'] - b['validator']:+.5f})")


if __name__ == "__main__":
    main()
