"""Score s05 / s08 / s09 on the FOUR FILMS THE LEADERBOARD ACTUALLY SCORES.

A discussion post argues the public stack's detectors were trained on all 199
train videos, so any local hold-out is in-sample and local measurements cannot
be trusted. The manifest claim is TRUE and verified:

    weights/biohub-temporal-unet3d-seed314159-v1/.../split_manifest.json
        method = unet_transformer_alltrain_seed314159_v1
        train  = all 199 videos;  test = 40, a strict SUBSET of train

But the same manifest also contains the four films the submission is scored on,
so the leaderboard films are in-sample too. In-sample-ness is therefore SYMMETRIC
and cannot by itself explain a local-vs-LB sign flip. What actually differs
between our 8 validator films and the 4 scored films is composition:

    validator  8 films, 4x 44b6 + 4x 6bba, 12 GT divisions, weighted ~evenly
    scored     4 films, 2x 44b6 + 2x 6bba, THREE GT divisions (all in
               6bba_05db0fb1), dominated by one 70k-node film

So rather than argue, score the real films. We hold their ILP graphs:
    artifacts/s05_output/tracking_repo/predictions/unknown/unet_transformer/split_0/
verified byte-identical to the relink-ON run's graphs, i.e. pure ILP output,
independent of post-processing config. Ground truth is in data/.../train/.

Caveat kept in view: this chain omits single-parent repair, the short-track
rescue and the DeepCenter vetoes, so ABSOLUTE proxy here is not the board's.
Only the A/B deltas are claimed, and those stages are identical across arms.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_DOC = __doc__
_src = (ROOT / "scripts/91_other_stages.py").read_text()
exec(_src.split("\nALL = []")[0])

TEST_PRED = ROOT / ("artifacts/s05_output/tracking_repo/predictions/unknown/"
                    "unet_transformer/split_0")
SCORED = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]

GC_UM, GC_REUSE = 5.0, 3.2
G2_TOTAL, G2_STEP = 10.2, 4.4
MIN_TRACK_LEN, KEEP_FORKS = 6, True


def SD(G, P):
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    e, a = safe_div(Q)
    return dict(G, edges=e), len(a)


def chain(G, P, gap2_after, lf_w):
    st = {}
    G, d = gap_close(G, max_um=GC_UM, reuse_um=GC_REUSE, allow_synth=True); st.update(d)
    if not gap2_after:
        G, d = gap2(G, max_total=G2_TOTAL, max_step=G2_STEP); st.update(d)
    G, _ = SD(G, P)
    if gap2_after:
        G, d = gap2(G, max_total=G2_TOTAL, max_step=G2_STEP); st.update(d)
    G, d = prune_isolated(G); st.update(d)
    G, d = short_track(G, MIN_TRACK_LEN, KEEP_FORKS); st.update(d)
    if lf_w > 0:
        G, d = linefit(G, w=lf_w, window=2); st.update(d)
    return G, st


ARMS = [("s05  gap2 before, linefit 0.8", False, 0.8),
        ("s08  gap2 AFTER,  linefit 0.8", True, 0.8),
        ("s09  gap2 AFTER,  linefit 0.3", True, 0.3)]

print(_DOC.split("Caveat kept")[0].rstrip())
print("=" * 104)

SETS = {"SCORED (the 4 leaderboard films)":
            {s: (load_pred(TEST_PRED / f"{s}.geff"), load_gt(s)) for s in SCORED},
        "VALIDATOR (the 8 train hold-outs)": DATA}

results = {}
for setname, data in SETS.items():
    print(f"\n### {setname}")
    print(f"{'arm':<32}{'proxy':>10}{'dproxy':>10}{'J':>10}{'divJ':>8}"
          f"{'TP/FP/FN':>10}{'ratio':>8}")
    base = None
    for label, g2a, lfw in ARMS:
        rows = []
        for stem, (P, GT) in data.items():
            G, _ = chain(G_of(P), P, g2a, lfw)
            rows.append(M2.score(G["t"], G["zyx"], G["edges"],
                                 GT["t"], GT["zyx"], GT["edges"], GT["n_est"]))
        r = M2.aggregate(rows)
        r["ratio"] = sum(x["n_pred"] for x in rows) / sum(x["n_est"] for x in rows)
        results[setname, label] = r
        d = "" if base is None else f"{r['proxy'] - base['proxy']:+.5f}"
        if base is None:
            base = r
        led = f"{r['dtp']}/{r['dfp']}/{r['dfn']}"
        print(f"{label:<32}{r['proxy']:>10.5f}{d:>10}{r['J']:>10.5f}"
              f"{r['divJ']:>8.4f}{led:>10}{r['ratio']:>8.4f}")

print("\n" + "=" * 104)
print("DOES THE LOCAL RESULT SURVIVE ON THE FILMS THAT ARE ACTUALLY SCORED?")
print(f"{'change':<34}{'validator':>12}{'SCORED films':>14}  verdict")
for arm, prev in (("s08  gap2 AFTER,  linefit 0.8", "s05  gap2 before, linefit 0.8"),
                  ("s09  gap2 AFTER,  linefit 0.3", "s08  gap2 AFTER,  linefit 0.8")):
    v = (results["VALIDATOR (the 8 train hold-outs)", arm]["proxy"]
         - results["VALIDATOR (the 8 train hold-outs)", prev]["proxy"])
    s = (results["SCORED (the 4 leaderboard films)", arm]["proxy"]
         - results["SCORED (the 4 leaderboard films)", prev]["proxy"])
    name = arm.split()[0] + " over " + prev.split()[0]
    if s > 0 and v > 0:
        verdict = "HOLDS"
    elif s <= 0 < v:
        verdict = "*** SIGN FLIP -- local gain does NOT survive ***"
    elif s > 0 >= v:
        verdict = "gains only on the scored films"
    else:
        verdict = "negative on both"
    print(f"{name:<34}{v:>+12.5f}{s:>+14.5f}  {verdict}")

print("\n--- per-film on the SCORED films (s09 vs s05) ---")
print(f"{'film':<18}{'divs':>6}{'s05 proxy':>11}{'s09 proxy':>11}{'delta':>10}{'weight':>9}")
for stem in SCORED:
    P, GT = SETS["SCORED (the 4 leaderboard films)"][stem]
    out = {}
    for label, g2a, lfw in ARMS:
        G, _ = chain(G_of(P), P, g2a, lfw)
        out[label] = M2.score(G["t"], G["zyx"], G["edges"],
                              GT["t"], GT["zyx"], GT["edges"], GT["n_est"])
    a = out[ARMS[0][0]]; c = out[ARMS[2][0]]
    pa = a["adj"] + 0.1 * a.get("divJ", 0.0)
    pc = c["adj"] + 0.1 * c.get("divJ", 0.0)
    ndiv = a["dtp"] + a["dfn"]
    print(f"{stem:<18}{ndiv:>6}{pa:>11.5f}{pc:>11.5f}{pc - pa:>+10.5f}{a['weight']:>9}")
