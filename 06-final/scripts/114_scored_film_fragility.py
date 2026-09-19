"""How fragile is the 4-film measurement that every decision now rests on?

The test directory holds exactly FOUR films, so the scored tier is not a proxy
for the target -- it IS the target. But it is ~2,273 GT edges, and we have never
asked how much of s10's +0.00471 is signal.

This counts, per arm pair, how many individual GROUND-TRUTH EDGES change status
(matched vs not). If a delta rests on a handful of edges out of thousands, it is
fragile regardless of how many decimal places the proxy has.

Then a paired bootstrap over GT edges for a confidence interval on the flip
count. Resampling edges (not films) is the only option -- with n=4, a film-level
bootstrap has no resolution at all.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

exec((ROOT / "scripts/91_other_stages.py").read_text().split("\nALL = []")[0])
_r = (ROOT / "scripts/25_relink_control.py").read_text()
exec(_r.split("def motion_relink", 1)[1].join(["def motion_relink", ""])
     .split("\nstems =")[0].split("\nDATA =")[0].split("\nif __name__")[0])

TEST_PRED = ROOT / ("artifacts/s05_output/tracking_repo/predictions/unknown/"
                    "unet_transformer/split_0")
SCORED = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
TD = {s: (load_pred(TEST_PRED / f"{s}.geff"), load_gt(s)) for s in SCORED}


def SD(G, P):
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    return dict(G, edges=safe_div(Q)[0])


def chain(G, P, gap2_after, lf_w, relink=False):
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


def hits(G, GT):
    """Boolean per GT edge: was it matched?"""
    p2g, _ = M2.match(G["t"], G["zyx"], GT["t"], GT["zyx"])
    got = set()
    gt_set = {(int(a), int(b)) for a, b in GT["edges"]}
    for s, d in G["edges"]:
        ms, mt = p2g.get(int(s)), p2g.get(int(d))
        if ms is not None and mt is not None and (ms, mt) in gt_set:
            got.add((ms, mt))
    return np.array([(int(a), int(b)) in got for a, b in GT["edges"]])


P32 = ROOT / ("artifacts/s11_output/tracking_repo/predictions/unknown/"
              "unet_transformer_val/split_0")
V32 = {p.stem: (load_pred(p), load_gt(p.stem)) for p in sorted(P32.glob("*.geff"))}
TIERS = {"SCORED (4 films)": TD, "VALIDATOR-32": V32}

ARMS = {"BASE": (False, 0.8, True), "s05": (False, 0.8, False),
        "s08": (True, 0.8, False), "s10": (False, 0.6, False)}
H = {}
for tier, data in TIERS.items():
    for stem, (P, GT) in data.items():
        for name, (g2a, lfw, rl) in ARMS.items():
            H[tier, stem, name] = hits(chain(G_of(P), P, g2a, lfw, rl), GT)

print(__doc__.split("Then a paired bootstrap")[0].rstrip())
print("=" * 86)
rng = np.random.default_rng(0)
for tier, data in TIERS.items():
  tot = sum(len(H[tier, s, "s05"]) for s in data)
  print(f"\n########## {tier}: {len(data)} films, {tot} GT edges ##########")
  for a, b in (("BASE", "s05"), ("s05", "s08"), ("s05", "s10")):
    ha = np.concatenate([H[tier, s, a] for s in data])
    hb = np.concatenate([H[tier, s, b] for s in data])
    gain = int((~ha & hb).sum())
    loss = int((ha & ~hb).sum())
    net = gain - loss
    print(f"--- {b} vs {a} ---")
    print(f"  GT edges gained {gain}, lost {loss}, NET {net:+d}  "
          f"out of {len(ha)} ({100*abs(net)/max(len(ha),1):.3f}% of the set)")
    d = hb.astype(int) - ha.astype(int)
    boot = np.array([d[rng.integers(0, len(d), len(d))].sum() for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  paired bootstrap over GT edges, 4000 draws: net {net:+d} "
          f"95% CI [{lo:+.0f}, {hi:+.0f}]")
    print(f"  => {'CI EXCLUDES ZERO -- direction resolved' if lo > 0 or hi < 0 else 'CI spans zero -- direction NOT resolved by this sample'}")
