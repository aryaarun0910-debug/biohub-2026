"""Re-test linefit smoothing on the s08 chain (relink OFF, gap2 AFTER safe_div).

docs/HANDOFF.md section 6: linefit w=0.4 measured +0.0024 on raw ILP graphs and
-0.00135 on the RELINKED pipeline (s06), because linefit fits along unique
predecessor/successor chains and relink changes that topology. The follow-on it
names is to re-test it on the no-relink base, where it was positive.

s08 is now the right base: relink off AND gap2 after safe_div. This sweeps
(weight, window) over the FULL s08 chain at deployed parameters, on the same 8
validator films.

Deployed is w=0.8, window=2. The notebook never sets BIOHUB_OUTPUT_LINEFIT_WEIGHT
at all -- 0.8 is the os.environ.get default -- so a variant ADDS a line, exactly
as s06 did.

ABORT_RULES: J and multiplier reported separately. linefit moves node
coordinates only, never topology or node count, so mult MUST stay flat -- any
mult movement here is a bug, and the script asserts it. The (w, window) surface
wobbles ~0.002 between neighbouring cells on 8 films, so this reports a
PLATEAU and the per-film spread, not an argmax to be trusted on its own.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_DOC = __doc__
_src = (ROOT / "scripts/91_other_stages.py").read_text()
exec(_src.split("\nALL = []")[0])

GC_UM, GC_REUSE = 5.0, 3.2
G2_TOTAL, G2_STEP = 10.2, 4.4
MIN_TRACK_LEN, KEEP_FORKS = 6, True


def SD(G, P):
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    e, a = safe_div(Q)
    return dict(G, edges=e), len(a)


def s08_chain(G, P, s, lf_w, lf_win):
    """The s08 chain: relink off, gap2 AFTER safe_div, linefit parameterised."""
    st = {}
    G, d = gap_close(G, max_um=GC_UM, reuse_um=GC_REUSE, allow_synth=True); st.update(d)
    G, _ = SD(G, P)
    G, d = gap2(G, max_total=G2_TOTAL, max_step=G2_STEP); st.update(d)
    G, d = prune_isolated(G); st.update(d)
    G, d = short_track(G, MIN_TRACK_LEN, KEEP_FORKS); st.update(d)
    if lf_w > 0:
        G, d = linefit(G, w=lf_w, window=lf_win); st.update(d)
    return G, st


print(_DOC.split("ABORT_RULES")[0].rstrip())
print("=" * 108)
print(HDR)

DEPLOYED = run(lambda G, P, s: s08_chain(G, P, s, 0.8, 2), "s08 DEPLOYED linefit w=0.8 win=2")
show(DEPLOYED)
print()

grid, best = [], None
for win in (2, 3, 4):
    for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        r = run(lambda G, P, s, a=w, b=win: s08_chain(G, P, s, a, b),
                f"  linefit w={w:g} win={win}")
        show(r, DEPLOYED)
        grid.append((w, win, r))
        if best is None or r["proxy"] > best[2]["proxy"]:
            best = (w, win, r)
    print()

# linefit moves coordinates only -- topology and node count are untouched -- so
# the NODE-COUNT RATIO must be exactly constant. Assert on that, not on `mult`.
#
# The aggregate `mult` is NOT constant here, and that is not a bug: metric2
# aggregates with weight = tp + fp + fn (the edge-confusion union, line 91),
# which depends on the PREDICTIONS. Moving coordinates changes edge matching,
# which changes each film's weight in np.average, so the weighted mean of an
# otherwise-constant per-film multiplier drifts by ~1e-5. That is re-weighting,
# not node deletion. ABORT_RULES' node-count exploit shows up as a moving
# `ratio`; a 1e-5 wobble in aggregate `mult` does not.
bad = [(w, win) for w, win, r in grid
       if abs(r["ratio"] - DEPLOYED["ratio"]) > 1e-12]
print(f"node-count ratio identical across all {len(grid)} cells: "
      f"{'PASS' if not bad else 'FAIL ' + str(bad)}  (n_pred/n_est = {DEPLOYED['ratio']:.4f})")
drift = max(abs(r["mult"] - DEPLOYED["mult"]) for _, _, r in grid)
print(f"aggregate mult drifts at most {drift:.2e} -- re-weighting by "
      f"weight=tp+fp+fn, not node deletion")
divs = {(r["dtp"], r["dfp"], r["dfn"]) for _, _, r in grid}
print(f"division ledger across all cells: {divs} "
      f"({'unchanged -- linefit is a pure edge-axis lever here' if len(divs) == 1 else 'MOVED'})")

bw, bwin, br = best
print(f"\nargmax: w={bw:g} win={bwin}  proxy {br['proxy']:.5f} "
      f"({br['proxy'] - DEPLOYED['proxy']:+.5f} vs deployed)")

# A plateau is worth more than an argmax: report every cell within 0.002 of the
# top, which is the known wobble of this surface on 8 films.
plat = sorted([(w, win, r["proxy"]) for w, win, r in grid
               if r["proxy"] > br["proxy"] - 0.002], key=lambda x: -x[2])
print(f"cells within the 0.002 wobble of the top ({len(plat)}):")
for w, win, p in plat:
    print(f"    w={w:g} win={win}  {p:.5f}")
ws = sorted({w for w, _, _ in plat}); wins = sorted({win for _, win, _ in plat})
print(f"    -> weight plateau {ws}, window plateau {wins}")

# per-film for the plateau centre
print(f"\n--- per-film: deployed w=0.8/win=2  vs  w={bw:g}/win={bwin} ---")
print(f"{'film':<24}{'dep J':>10}{'new J':>10}{'dJ':>10}")
nwin = 0
for stem, (P, GT) in DATA.items():
    Ga, _ = s08_chain(G_of(P), P, stem, 0.8, 2)
    Gb, _ = s08_chain(G_of(P), P, stem, bw, bwin)
    ra = M2.score(Ga["t"], Ga["zyx"], Ga["edges"], GT["t"], GT["zyx"], GT["edges"], GT["n_est"])
    rb = M2.score(Gb["t"], Gb["zyx"], Gb["edges"], GT["t"], GT["zyx"], GT["edges"], GT["n_est"])
    nwin += rb["J_edge"] > ra["J_edge"]
    print(f"{stem:<24}{ra['J_edge']:>10.5f}{rb['J_edge']:>10.5f}"
          f"{rb['J_edge'] - ra['J_edge']:>+10.5f}")
print(f"films improved: {nwin}/8")
