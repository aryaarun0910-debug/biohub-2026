"""Re-test everything on the 32-film tier that s11 generated.

s05, s08 and s10 were all selected on 12 deployed-quality graphs: 8 validator
films + 4 scored. Twelve films is what killed s06 and s09. s11 raised
BIOHUB_VALIDATOR_N_PER_TYPE 4 -> 16 and emitted 32 validator graphs:

    original 8   12 divisions,  5,751 labelled edges
    added   24   34 divisions, 13,581 labelled edges
    total   32   46 divisions, 19,332 labelled edges

The original 8 are a SUBSET, so every earlier validator number stays directly
comparable and this is a superset, not a replacement. The added films are harder
(kernel base_proxy 0.9397 on 32 vs 0.9715 on 8), which is the point: a change
that only looks good on the easy 8 becomes visible as such.

Two questions, both about submissions that are ON THE BOARD RIGHT NOW:
  1. does s08's reorder still gain? Its whole validator-tier gain was ONE
     division out of 12; there are now 46.
  2. does s10's linefit 0.6 still sit on the 0.4-0.6 plateau?

Caveat unchanged from scripts/103: single-parent repair, the short-track rescue
and the DeepCenter vetoes are not ported, so ABSOLUTE proxy is not the kernel's.
They are identical across arms, so A/B deltas are what is claimed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from concurrent.futures import ProcessPoolExecutor

import numpy as np

exec((ROOT / "scripts/91_other_stages.py").read_text().split("\nALL = []")[0])
_r = (ROOT / "scripts/25_relink_control.py").read_text()
exec(_r.split("def motion_relink", 1)[1].join(["def motion_relink", ""])
     .split("\nstems =")[0].split("\nDATA =")[0].split("\nif __name__")[0])

P32 = ROOT / ("artifacts/s11_output/tracking_repo/predictions/unknown/"
              "unet_transformer_val/split_0")
OLD8 = ["44b6_12dfb391", "44b6_267148e4", "44b6_2a2eff9f", "44b6_341df25f",
        "6bba_062c8d37", "6bba_07e24132", "6bba_085bf656", "6bba_09961292"]
STEMS32 = sorted(p.stem for p in P32.glob("*.geff"))
D32 = {s: (load_pred(P32 / f"{s}.geff"), load_gt(s)) for s in STEMS32}
SETS = {"32 films": D32, "the original 8": {s: D32[s] for s in OLD8}}


def SD(G, P):
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    return dict(G, edges=safe_div(Q)[0])


def chain(G, P, relink, gap2_after, lf_w):
    if relink:
        G = dict(G, edges=motion_relink(dict(P, edges=G["edges"])))
    G, _ = gap_close(G, max_um=5.0, reuse_um=3.2, allow_synth=True)
    if not gap2_after:
        G, _ = gap2(G, max_total=10.2, max_step=4.4)
    G = SD(G, P)
    if gap2_after:
        G, _ = gap2(G, max_total=10.2, max_step=4.4)
    G, _ = prune_isolated(G)
    G, _ = short_track(G, 6, True)
    if lf_w > 0:
        G, _ = linefit(G, w=lf_w, window=2)
    return G


ARMS = [("BASE relink ON,  gap2 before, lf0.8", True, False, 0.8),
        ("s05  relink OFF, gap2 before, lf0.8", False, False, 0.8),
        ("s08  relink OFF, gap2 AFTER,  lf0.8", False, True, 0.8),
        ("s10  relink OFF, gap2 before, lf0.6", False, False, 0.6)]


def job(a):
    setname, label, rl, g2a, lfw = a
    rows = []
    for stem, (P, GT) in SETS[setname].items():
        G = chain(G_of(P), P, rl, g2a, lfw)
        rows.append(M2.score(G["t"], G["zyx"], G["edges"],
                             GT["t"], GT["zyx"], GT["edges"], GT["n_est"]))
    r = M2.aggregate(rows)
    r["ratio"] = sum(x["n_pred"] for x in rows) / sum(x["n_est"] for x in rows)
    return setname, label, r


def lfjob(a):
    setname, w = a
    rows = []
    for stem, (P, GT) in SETS[setname].items():
        G = chain(G_of(P), P, False, False, w)
        rows.append(M2.score(G["t"], G["zyx"], G["edges"],
                             GT["t"], GT["zyx"], GT["edges"], GT["n_est"]))
    return setname, w, M2.aggregate(rows)


def main():
    print(__doc__.split("Caveat unchanged")[0].rstrip())
    print("=" * 104)
    jobs = [(sn, l, a, b, c) for sn in SETS for l, a, b, c in ARMS]
    with ProcessPoolExecutor(max_workers=14) as ex:
        out = list(ex.map(job, jobs, chunksize=1))
    R = {(sn, l): r for sn, l, r in out}

    for sn in SETS:
        print(f"\n### {sn}  (n={len(SETS[sn])})")
        print(f"{'arm':<38}{'proxy':>10}{'vs BASE':>10}{'J':>10}{'divJ':>8}{'TP/FP/FN':>11}")
        base = R[sn, ARMS[0][0]]
        for l, *_ in ARMS:
            r = R[sn, l]
            d = "" if l == ARMS[0][0] else f"{r['proxy']-base['proxy']:+.5f}"
            led = f"{r['dtp']}/{r['dfp']}/{r['dfn']}"
            print(f"{l:<38}{r['proxy']:>10.5f}{d:>10}{r['J']:>10.5f}"
                  f"{r['divJ']:>8.4f}{led:>11}")

    print("\n" + "=" * 104)
    print("DOES THE 8-FILM CONCLUSION SURVIVE AT 32 FILMS?")
    print(f"{'change':<20}{'on 8 films':>13}{'on 32 films':>14}  verdict")
    for name, a, b in (("s05 over BASE", ARMS[1][0], ARMS[0][0]),
                       ("s08 over s05", ARMS[2][0], ARMS[1][0]),
                       ("s10 over s05", ARMS[3][0], ARMS[1][0])):
        d8 = R["the original 8", a]["proxy"] - R["the original 8", b]["proxy"]
        d32 = R["32 films", a]["proxy"] - R["32 films", b]["proxy"]
        if d32 > 1e-4 and d8 > 1e-4:
            v = "HOLDS"
        elif d32 <= 1e-4 < d8:
            v = "*** DOES NOT SURVIVE -- was an 8-film artefact ***"
        elif d32 > 1e-4 >= d8:
            v = "appears only at 32 films"
        else:
            v = "negative on both"
        print(f"{name:<20}{d8:>+13.5f}{d32:>+14.5f}  {v}")

    print("\n--- linefit weight, on the s05 chain, 8 vs 32 films ---")
    WS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    with ProcessPoolExecutor(max_workers=14) as ex:
        lo = list(ex.map(lfjob, [(sn, w) for sn in SETS for w in WS], chunksize=1))
    L = {(sn, w): r for sn, w, r in lo}
    print(f"{'w':>5}{'8-film d':>12}{'32-film d':>12}   note")
    best = None
    for w in WS:
        d8 = L["the original 8", w]["proxy"] - L["the original 8", 0.8]["proxy"]
        d32 = L["32 films", w]["proxy"] - L["32 films", 0.8]["proxy"]
        if best is None or d32 > best[1]:
            best = (w, d32)
        tag = "  <- s10 ships this" if w == 0.6 else ("  <- s09 shipped this" if w == 0.3 else "")
        print(f"{w:>5.1f}{d8:>+12.5f}{d32:>+12.5f}{tag}")
    print(f"\n32-film optimum: w={best[0]:g} ({best[1]:+.5f}); s10 ships 0.6 "
          f"({L['32 films', 0.6]['proxy'] - L['32 films', 0.8]['proxy']:+.5f})")


if __name__ == "__main__":
    main()
