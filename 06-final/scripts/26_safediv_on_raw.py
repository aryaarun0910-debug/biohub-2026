"""Re-price the safe_div gates on the RAW ILP graph.

Every earlier gate measurement was made on a relinked graph (deployed) or on my
weak local rebuild. The raw ILP graph has a different, much cleaner candidate
population -- 5 TP / 2 FP versus 3 TP / 1 FP -- so the gates need re-pricing
there rather than assuming the old numbers carry over.

DISCIPLINE: 12 ground-truth divisions across 8 films. One event is ~0.0083 of
proxy. Anything finer is not a measurement, and tuning hard on 12 events is how
you overfit. TP/FP/FN are printed beside every divJ, and both folds separately.
"""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from biohub import metric2 as M2
exec(open("scripts/24_keep_ilp_edges.py").read().split("stems = sorted")[0].split('"""', 2)[2])

PRED = Path("artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0")
stems = sorted(p.stem for p in PRED.glob("*.geff"))
DATA = {s: (load_pred(PRED / f"{s}.geff"), load_gt(s)) for s in stems}
FOLD = {s: s.split("_")[0] for s in stems}


def ev(**kw):
    rows, per, na = [], {}, 0
    for s, (P, G) in DATA.items():
        e, a = safe_div(P, **kw); na += len(a)
        r = M2.score(P["t"], P["zyx"], e, G["t"], G["zyx"], G["edges"], G["n_est"])
        rows.append(r); per.setdefault(FOLD[s], []).append(r)
    return M2.aggregate(rows), {k: M2.aggregate(v) for k, v in per.items()}, na


DEPLOY = dict(parent_max=9.0, sister_max=14.0, child_max=10.0, tau=0.6,
              diverge=2.25, frame_cap=0.0076, glob_cap=0.00375)
base, bper, bn = ev(**DEPLOY)
print(f"deployed gates on the raw graph:  proxy {base['proxy']:.5f}  adj {base['adj']:.5f}"
      f"  divJ {base['divJ']:.4f}  ({base['dtp']}/{base['dfp']}/{base['dfn']})  added {bn}")
print(f"  per fold: " + "  ".join(f"{k} {v['proxy']:.4f} ({v['dtp']}/{v['dfp']}/{v['dfn']})"
                                  for k, v in bper.items()))
print(f"\n{'one change from deployed':<30}{'proxy':>9}{'d':>9}{'adj':>9}{'divJ':>8}"
      f"{'TP':>4}{'FP':>4}{'FN':>4}{'added':>7}")
GRID = {
    "parent_max": [7.0, 9.0, 11.0, 13.0, 16.0],
    "sister_max": [10.0, 12.0, 14.0, 16.0, 20.0],
    "child_max":  [6.0, 8.0, 10.0, 14.0],
    "tau":        [0.3, 0.45, 0.6, 0.9, 1.35],
    "diverge":    [0.0, 1.0, 2.25, 3.5, 5.0],
    "frame_cap":  [0.002, 0.004, 0.0076, 0.015],
    "glob_cap":   [0.001, 0.0015, 0.00375, 0.008],
}
best_single = {}
for k, vals in GRID.items():
    for v in vals:
        if v == DEPLOY[k]:
            continue
        kw = dict(DEPLOY); kw[k] = v
        a, _, na = ev(**kw)
        d = a["proxy"] - base["proxy"]
        mark = "  <-" if d > 0.0083 else ""
        print(f"{f'{k} = {v}':<30}{a['proxy']:>9.5f}{d:>+9.5f}{a['adj']:>9.5f}"
              f"{a['divJ']:>8.4f}{a['dtp']:>4}{a['dfp']:>4}{a['dfn']:>4}{na:>7}{mark}")
        if d > best_single.get(k, (0, None))[0]:
            best_single[k] = (d, v)
print()
winners = {k: v for k, (d, v) in best_single.items() if d > 0.0083}
print(f"changes worth more than one division event ({0.0083:.4f}): {winners or 'none'}")
if winners:
    kw = dict(DEPLOY); kw.update(winners)
    a, per, na = ev(**kw)
    print(f"\nall of them together: proxy {a['proxy']:.5f} ({a['proxy']-base['proxy']:+.5f})"
          f"  adj {a['adj']:.5f}  divJ {a['divJ']:.4f} ({a['dtp']}/{a['dfp']}/{a['dfn']})"
          f"  added {na}")
    print(f"  per fold: " + "  ".join(f"{k} {v['proxy']:.4f} ({v['dtp']}/{v['dfp']}/{v['dfn']})"
                                      for k, v in per.items()))
    print("\n  CAUTION: this is 12 events. A combination tuned on 12 events is a "
          "candidate to TEST, not a result.")
