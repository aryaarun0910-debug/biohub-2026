"""How much of the no-relink win rests on one film?

The entire s05 case is +0.0295 measured on 8 films and 12 divisions. If it is
driven by one film it is not a finding, it is an anecdote. Leave-one-film-out
plus a per-film breakdown, reporting the EDGE and DIVISION halves separately --
the edge half rests on ~181k edges and should be stable; the division half
rests on 12 events and should not be trusted to the same precision.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.optimize import linear_sum_assignment
from biohub import metric2 as M2
exec(open("scripts/24_keep_ilp_edges.py").read().split("stems = sorted")[0].split('"""', 2)[2])
exec(open("scripts/25_relink_control.py").read().split("PRED = Path")[0].split("SCALE = M2.SCALE")[1])

PRED = Path("artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0")
stems = sorted(p.stem for p in PRED.glob("*.geff"))
DATA = {s: (load_pred(PRED / f"{s}.geff"), load_gt(s)) for s in stems}

R = {}
for s, (P, G) in DATA.items():
    out = {}
    for name in ("raw", "relinked"):
        e = P["edges"] if name == "raw" else motion_relink(P)
        Q = dict(P); Q["edges"] = e
        e2, _ = safe_div(Q)
        out[name] = M2.score(P["t"], P["zyx"], e2, G["t"], G["zyx"], G["edges"], G["n_est"])
    R[s] = out

print(f"{'film':<18}{'raw proxy':>11}{'relink proxy':>14}{'delta':>9}"
      f"{'d_adj':>9}{'raw div':>10}{'relink div':>12}")
for s in stems:
    a, b = R[s]["raw"], R[s]["relinked"]
    pa = a["adj"] + 0.1 * (a["dtp"] / max(a["dtp"] + a["dfp"] + a["dfn"], 1))
    pb = b["adj"] + 0.1 * (b["dtp"] / max(b["dtp"] + b["dfp"] + b["dfn"], 1))
    print(f"{s:<18}{pa:>11.5f}{pb:>14.5f}{pa-pb:>+9.5f}{a['adj']-b['adj']:>+9.5f}"
          f"{f'{a[chr(100)+chr(116)+chr(112)]}/{a[chr(100)+chr(102)+chr(112)]}/{a[chr(100)+chr(102)+chr(110)]}':>10}"
          f"{f'{b[chr(100)+chr(116)+chr(112)]}/{b[chr(100)+chr(102)+chr(112)]}/{b[chr(100)+chr(102)+chr(110)]}':>12}")

full_r = M2.aggregate([R[s]["raw"] for s in stems])
full_l = M2.aggregate([R[s]["relinked"] for s in stems])
print(f"\nALL 8   raw {full_r['proxy']:.5f}   relinked {full_l['proxy']:.5f}   "
      f"delta {full_r['proxy']-full_l['proxy']:+.5f}")

print(f"\nleave-one-film-out:")
print(f"{'held out':<18}{'delta proxy':>13}{'delta adj':>12}{'delta 0.1*divJ':>16}")
deltas = []
for s in stems:
    keep = [x for x in stems if x != s]
    r = M2.aggregate([R[x]["raw"] for x in keep])
    l = M2.aggregate([R[x]["relinked"] for x in keep])
    d = r["proxy"] - l["proxy"]; deltas.append(d)
    print(f"{s:<18}{d:>+13.5f}{r['adj']-l['adj']:>+12.5f}"
          f"{0.1*(r['divJ']-l['divJ']):>+16.5f}")
deltas = np.array(deltas)
print(f"\njackknife range {deltas.min():+.5f} to {deltas.max():+.5f}  "
      f"(full-sample {full_r['proxy']-full_l['proxy']:+.5f})")
print(f"EDGE half alone (181k edges, well resolved): {full_r['adj']-full_l['adj']:+.5f}")
print(f"DIVISION half alone (12 events, coarse):     {0.1*(full_r['divJ']-full_l['divJ']):+.5f}")
print(f"\nsign agreement across all 8 leave-one-out folds: "
      f"{(deltas > 0).sum()}/8 positive")
