"""Does removing motion relink (s05, +0.0224 on the validator) survive on the
FOUR FILMS THE LEADERBOARD ACTUALLY SCORES?

scripts/102 showed s08 is +0.00000 and s09 is -0.00086 on the scored films,
against +0.00748 and +0.00487 on the validator films. Both of those sit on the
no-relink base, so neither tested the decision that base rests on.

s05 is already SUBMITTED. This is the measurement that should have preceded it.

Same substrate as 102: ILP graphs from
artifacts/s05_output/.../unet_transformer/split_0, verified byte-identical to
the relink-ON run, i.e. pure ILP output. Ground truth from data/.../train/.
motion_relink() is the deployed-stage reproduction from scripts/25_relink_control.py.

Caveat unchanged: single-parent repair, the short-track rescue and the
DeepCenter vetoes are not ported, so ABSOLUTE proxy is not the board's. They are
identical across arms, so the A/B delta is what is claimed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_DOC = __doc__
exec((ROOT / "scripts/91_other_stages.py").read_text().split("\nALL = []")[0])
_r = (ROOT / "scripts/25_relink_control.py").read_text()
exec(_r.split("def motion_relink", 1)[1].join(["def motion_relink", ""])
     .split("\nstems =")[0].split("\nDATA =")[0].split("\nif __name__")[0])

TEST_PRED = ROOT / ("artifacts/s05_output/tracking_repo/predictions/unknown/"
                    "unet_transformer/split_0")
SCORED = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]


def SD(G, P):
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    e, a = safe_div(Q)
    return dict(G, edges=e), len(a)


def chain(G, P, relink, gap2_after, lf_w):
    if relink:
        G = dict(G, edges=motion_relink(dict(P, edges=G["edges"])))
    G, _ = gap_close(G, max_um=5.0, reuse_um=3.2, allow_synth=True)
    if not gap2_after:
        G, _ = gap2(G, max_total=10.2, max_step=4.4)
    G, _ = SD(G, P)
    if gap2_after:
        G, _ = gap2(G, max_total=10.2, max_step=4.4)
    G, _ = prune_isolated(G)
    G, _ = short_track(G, 6, True)
    if lf_w > 0:
        G, _ = linefit(G, w=lf_w, window=2)
    return G


ARMS = [("BASE  relink ON,  gap2 before, lf 0.8", True,  False, 0.8),
        ("s05   relink OFF, gap2 before, lf 0.8", False, False, 0.8),
        ("s08   relink OFF, gap2 AFTER,  lf 0.8", False, True,  0.8),
        ("s09   relink OFF, gap2 AFTER,  lf 0.3", False, True,  0.3)]

print(_DOC.split("Caveat unchanged")[0].rstrip())
print("=" * 100)

SETS = {"SCORED (the 4 leaderboard films)":
            {s: (load_pred(TEST_PRED / f"{s}.geff"), load_gt(s)) for s in SCORED},
        "VALIDATOR (the 8 train hold-outs)": DATA}

res = {}
for setname, data in SETS.items():
    print(f"\n### {setname}")
    print(f"{'arm':<40}{'proxy':>10}{'vs BASE':>10}{'J':>10}{'divJ':>8}{'TP/FP/FN':>10}")
    base = None
    for label, rl, g2a, lfw in ARMS:
        rows = []
        for stem, (P, GT) in data.items():
            G = chain(G_of(P), P, rl, g2a, lfw)
            rows.append(M2.score(G["t"], G["zyx"], G["edges"],
                                 GT["t"], GT["zyx"], GT["edges"], GT["n_est"]))
        r = M2.aggregate(rows)
        res[setname, label] = r
        if base is None:
            base = r
        d = f"{r['proxy'] - base['proxy']:+.5f}" if label != ARMS[0][0] else ""
        led = f"{r['dtp']}/{r['dfp']}/{r['dfn']}"
        print(f"{label:<40}{r['proxy']:>10.5f}{d:>10}{r['J']:>10.5f}{r['divJ']:>8.4f}{led:>10}")

print("\n" + "=" * 100)
print("EVERY CHANGE, MEASURED WHERE IT IS SCORED")
print(f"{'change':<22}{'validator':>12}{'SCORED':>12}  verdict")
S, V = "SCORED (the 4 leaderboard films)", "VALIDATOR (the 8 train hold-outs)"
for name, a, b in (("s05 over BASE", ARMS[1][0], ARMS[0][0]),
                   ("s08 over s05",  ARMS[2][0], ARMS[1][0]),
                   ("s09 over s08",  ARMS[3][0], ARMS[2][0]),
                   ("s09 over BASE", ARMS[3][0], ARMS[0][0])):
    dv = res[V, a]["proxy"] - res[V, b]["proxy"]
    ds = res[S, a]["proxy"] - res[S, b]["proxy"]
    # 1e-6 is far below anything reportable. Use a threshold with meaning:
    # one division event on the SCORED films is 0.1/(0+6+3) = 0.0111, and the
    # smallest board-visible step is 0.001. Anything under 1e-4 is "neutral".
    EPS = 1e-4
    if ds > EPS and dv > EPS:
        verdict = "holds on both"
    elif abs(ds) <= EPS < dv:
        verdict = "NEUTRAL where it is scored (all of it was validator-only)"
    elif ds < -EPS < dv:
        verdict = "*** SIGN FLIP ***"
    elif ds > EPS >= dv:
        verdict = "gains only where it is scored"
    else:
        verdict = "negative on both"
    print(f"{name:<22}{dv:>+12.5f}{ds:>+12.5f}  {verdict}")

print("\n--- per-film, BASE vs s05, on the SCORED films ---")
print(f"{'film':<18}{'divs':>6}{'BASE':>10}{'s05':>10}{'delta':>10}{'GT edges':>10}")
for stem in SCORED:
    P, GT = SETS[S][stem]
    o = {}
    for label, rl, g2a, lfw in (ARMS[0], ARMS[1]):
        G = chain(G_of(P), P, rl, g2a, lfw)
        o[label] = M2.score(G["t"], G["zyx"], G["edges"],
                            GT["t"], GT["zyx"], GT["edges"], GT["n_est"])
    a, b = o[ARMS[0][0]], o[ARMS[1][0]]
    pa = a["adj"] + 0.1 * a.get("divJ", 0.0)
    pb = b["adj"] + 0.1 * b.get("divJ", 0.0)
    print(f"{stem:<18}{a['dtp'] + a['dfn']:>6}{pa:>10.5f}{pb:>10.5f}"
          f"{pb - pa:>+10.5f}{a['weight']:>10}")
