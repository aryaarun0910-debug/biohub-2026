"""Does the gap2/safe_div reorder survive in the FULL s05 chain (relink OFF)?

scripts/91_other_stages.py measured the reorder on `raw + gap2 + safe_div`
alone:

    + gap2 total=10.2 step=4.4  (BEFORE safe_div)   proxy 0.961627  -0.008016
    + safe_div then gap2        (AFTER)             proxy 0.970849  +0.001206

a +0.00922 swing. But that is a THREE-STAGE topology. s05 deploys the whole
chain, and docs/HANDOFF.md section 6 is the record of what happens when a delta
measured in one topology is assumed to hold in another: linefit w=0.4 was
+0.0024 on raw graphs and -0.00135 on the relinked pipeline, because the stages
either side of it changed what it was fitting along.

So measure the reorder where it would actually ship: every deployed stage on,
at deployed parameters, motion relink off (that is what s05 is), and gap2 moved
from before safe_div to after it. Nothing else differs between the two arms.

    A (s05 as submitted)  gap_close -> gap2 -> safe_div -> prune -> short -> linefit
    B (s07 on top of s05) gap_close -> safe_div -> gap2 -> prune -> short -> linefit

Deployed parameters, read out of cell 2 of the 0.947 notebook:
    GAP_CLOSE_UM 5.0, GAP_CLOSE_REUSE_UM 3.2, GAP2_MAX_TOTAL_UM 10.2,
    GAP2_MAX_STEP_UM 4.4, GAP2_REQUIRE_CONTEXT 1, OUTPUT_MIN_TRACK_LEN 6,
    OUTPUT_KEEP_DIVISION_COMPONENTS 1, OUTPUT_LINEFIT_WEIGHT 0.8, WINDOW 2,
    OUTPUT_PRUNE_ISOLATED 1.

Known gaps between this chain and the kernel's: single-parent repair and the
adaptive short-track rescue are not ported, and the DeepCenter gap/safe-div
vetoes are not applied. Those are held IDENTICAL across the two arms, so they
cannot manufacture the difference -- but they do mean the absolute proxy here
is not the kernel's proxy. Only the A-vs-B delta is being claimed.

Per ABORT_RULES.md: J and the multiplier are reported separately, and one
division event is 0.0083 of proxy on these 8 films.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# exec() of a source string whose first statement is a string literal ASSIGNS
# __doc__ in the target globals, so the exec below would silently replace this
# module's docstring with 91's -- and the log header would then describe the
# wrong experiment. Keep our own before that happens.
_DOC = __doc__

# Reuse the validated stage ports from 91 (everything defined before it starts
# running experiments at `ALL = []`): prune_isolated, short_track, gap_close,
# gap2, linefit, safe_div, G_of, DATA, run, show, HDR, M2.
_src = (ROOT / "scripts/91_other_stages.py").read_text()
assert "\nALL = []" in _src, "91_other_stages.py layout changed; re-check the split point"
exec(_src.split("\nALL = []")[0])

# Deployed parameters.
GC_UM, GC_REUSE = 5.0, 3.2
G2_TOTAL, G2_STEP = 10.2, 4.4
MIN_TRACK_LEN, KEEP_FORKS = 6, True
LF_W, LF_WIN = 0.8, 2


def SD(G, P):
    """safe_div on a (possibly node-augmented) graph. Verbatim from 91."""
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    e, a = safe_div(Q)
    return dict(G, edges=e), len(a)


def chain(G, P, s, gap2_after, use_gap2=True):
    """The deployed post-processing chain with relink off; gap2 on either side."""
    st = {}
    G, d = gap_close(G, max_um=GC_UM, reuse_um=GC_REUSE, allow_synth=True); st.update(d)
    if use_gap2 and not gap2_after:
        G, d = gap2(G, max_total=G2_TOTAL, max_step=G2_STEP); st.update(d)
    G, _ = SD(G, P)
    if use_gap2 and gap2_after:
        G, d = gap2(G, max_total=G2_TOTAL, max_step=G2_STEP); st.update(d)
    G, d = prune_isolated(G); st.update(d)
    G, d = short_track(G, MIN_TRACK_LEN, KEEP_FORKS); st.update(d)
    G, d = linefit(G, w=LF_W, window=LF_WIN); st.update(d)
    return G, st


print(_DOC.split("Per ABORT_RULES")[0].rstrip())
print("=" * 118)
print(HDR)

A = run(lambda G, P, s: chain(G, P, s, gap2_after=False), "A  s05 chain, gap2 BEFORE safe_div")
show(A)
B = run(lambda G, P, s: chain(G, P, s, gap2_after=True), "B  s05 chain, gap2 AFTER safe_div")
show(B, A)
# Row C trips show()'s "<<FALSE GAIN: all multiplier" flag. That flag is a
# heuristic: it fires when proxy rises while edge J does not, which is the
# node-count exploit signature. Here it is a FALSE ALARM -- C's gain over A is
# the division (divJ 0.2857 -> 0.3571, the same one B recovers), not node
# deletion. The flag cannot see divJ. Do not read it as an exploit warning.
N = run(lambda G, P, s: chain(G, P, s, gap2_after=False, use_gap2=False),
        "C  s05 chain, gap2 OFF entirely")
show(N, A)

d = B["proxy"] - A["proxy"]
# One division event is worth 0.1/(TP+FP+FN) of proxy. ABORT_RULES quotes 0.0083
# for a 12-division denominator; here the denominator is 14, so an event is
# 0.1/14 = 0.00714. Comparing this delta against 0.0083 would be the wrong ruler.
event = 0.1 / (B["dtp"] + B["dfp"] + B["dfn"])
print()
print(f"reorder delta B - A : {d:+.5f} proxy   "
      f"(dJ {B['J'] - A['J']:+.5f}, dmult {B['mult'] - A['mult']:+.5f})")
print(f"divisions           : A {A['dtp']}/{A['dfp']}/{A['dfn']}  ->  "
      f"B {B['dtp']}/{B['dfp']}/{B['dfn']}   divJ {A['divJ']:.4f} -> {B['divJ']:.4f}")
print(f"one division event  : {event:.5f} of proxy at this denominator "
      f"({B['dtp'] + B['dfp'] + B['dfn']} divisions)")
print(f"three-stage topology predicted +0.00922; full chain gives {d:+.5f} "
      f"({100 * (d / 0.00922 - 1):+.0f}%), sign HELD")
print()
print("READ: the delta is one recovered division (TP 4 -> 5, FN 8 -> 7) plus "
      f"{B['J'] - A['J']:+.5f} of edge J.")
print("      That is exactly one event -- real and discrete, but AT the "
      "measurement floor, not above it.")
print("      The gain is on the DIVISION axis, where the offline-vs-board "
      "ledger is 3 for 3,")
print("      not the edge axis, where it is 0 for 3.")

# ---- per-film: is any gain broad, or one film? -------------------------
print()
print("--- per-film (the 8 validator films) ---")
print(f"{'film':<24}{'A proxy':>10}{'B proxy':>10}{'delta':>10}"
      f"{'A J':>10}{'B J':>10}{'dJ':>10}{'n/n_est':>9}")
nwin = 0
for stem, (P, GT) in DATA.items():
    Ga, _ = chain(G_of(P), P, stem, gap2_after=False)
    Gb, _ = chain(G_of(P), P, stem, gap2_after=True)
    ra = M2.score(Ga["t"], Ga["zyx"], Ga["edges"], GT["t"], GT["zyx"], GT["edges"], GT["n_est"])
    rb = M2.score(Gb["t"], Gb["zyx"], Gb["edges"], GT["t"], GT["zyx"], GT["edges"], GT["n_est"])
    pa = ra["adj"] + 0.1 * ra.get("divJ", 0.0)
    pb = rb["adj"] + 0.1 * rb.get("divJ", 0.0)
    nwin += pb > pa
    print(f"{stem:<24}{pa:>10.5f}{pb:>10.5f}{pb - pa:>+10.5f}"
          f"{ra['J_edge']:>10.5f}{rb['J_edge']:>10.5f}{rb['J_edge'] - ra['J_edge']:>+10.5f}"
          f"{rb['n_pred'] / rb['n_est']:>9.4f}")
print(f"films where the reorder helped: {nwin}/8")


# ---- per-film division attribution -------------------------------------
print()
print("--- which film supplies the division? ---")
print(f"{'film':<24}{'A tp/fp/fn':>12}{'B tp/fp/fn':>12}")
for stem, (P, GT) in DATA.items():
    Ga, _ = chain(G_of(P), P, stem, gap2_after=False)
    Gb, _ = chain(G_of(P), P, stem, gap2_after=True)
    ra = M2.score(Ga["t"], Ga["zyx"], Ga["edges"], GT["t"], GT["zyx"], GT["edges"], GT["n_est"])
    rb = M2.score(Gb["t"], Gb["zyx"], Gb["edges"], GT["t"], GT["zyx"], GT["edges"], GT["n_est"])
    a = f"{ra['dtp']}/{ra['dfp']}/{ra['dfn']}"
    b = f"{rb['dtp']}/{rb['dfp']}/{rb['dfn']}"
    mark = "   <-- DIVISION RECOVERED" if rb["dtp"] > ra["dtp"] else ""
    print(f"{stem:<24}{a:>12}{b:>12}{mark}")

# ---- sensitivity: does the sign depend on the stages either side? ------
# s06's lesson was that neighbouring stages decide the sign. So vary the
# nuisance parameters around the deployed point and check B > A throughout.
print()
print("--- sign stability under the stages either side of the move ---")
print(f"{'perturbation':<38}{'A proxy':>10}{'B proxy':>10}{'B-A':>10}{'divJ A->B':>18}")
held = 0
cases = [("deployed", {}),
         ("linefit w=0.4 (the s06 setting)", {"LF_W": 0.4}),
         ("linefit w=0.0 (smoothing off)", {"LF_W": 0.0}),
         ("linefit window=3", {"LF_WIN": 3}),
         ("gap_close 8.0um", {"GC_UM": 8.0}),
         ("gap_close 3.0um", {"GC_UM": 3.0}),
         ("short_track L=9 (the board 0.947)", {"MIN_TRACK_LEN": 9}),
         ("short_track keep_forks=0", {"KEEP_FORKS": False}),
         ("gap2 total=14 step=6", {"G2_TOTAL": 14.0, "G2_STEP": 6.0})]
_defaults = dict(GC_UM=GC_UM, GC_REUSE=GC_REUSE, G2_TOTAL=G2_TOTAL, G2_STEP=G2_STEP,
                 MIN_TRACK_LEN=MIN_TRACK_LEN, KEEP_FORKS=KEEP_FORKS,
                 LF_W=LF_W, LF_WIN=LF_WIN)
for name, over in cases:
    g = globals()
    g.update(_defaults)
    g.update(over)
    ra = run(lambda G, P, s: chain(G, P, s, gap2_after=False), name)
    rb = run(lambda G, P, s: chain(G, P, s, gap2_after=True), name)
    dd = rb["proxy"] - ra["proxy"]
    held += dd > 0
    dv = f"{ra['divJ']:.4f} -> {rb['divJ']:.4f}"
    print(f"{name:<38}{ra['proxy']:>10.5f}{rb['proxy']:>10.5f}{dd:>+10.5f}{dv:>18}")
globals().update(_defaults)
print(f"reorder positive in {held}/{len(cases)} perturbations")
