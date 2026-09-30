"""Price the post-processing variables the lineage never moved.

72 post-processing variables exist; 38 have never been set by anyone in the
fork chain -- they run on whatever default the original author picked. This
prices the testable ones against real ground truth, in the winning stage order
(gap_close -> safe_div -> gap2 -> linefit) on raw ILP graphs.

Discipline: 12 GT divisions across 8 films, so one division is ~0.0083 of
proxy. Any gain that arrives through the MULTIPLIER while J is flat or falling
is the node-count exploit, not an improvement -- flagged explicitly.
"""
import sys, itertools
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from biohub import metric2 as M2

_s24 = open("scripts/24_keep_ilp_edges.py").read()
exec(_s24.split("stems = sorted")[0].split('"""', 2)[2])          # load_pred/load_gt/safe_div
_s91 = open("scripts/91_other_stages.py").read()
exec(_s91[_s91.index("def _remap"):_s91.index("def make_refine")])  # stage ports

PRED = Path("artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0")
STEMS = sorted(p.stem for p in PRED.glob("*.geff"))


def safe_div2(P, require_divergence=True, require_mutual_nn=True, **kw):
    """safe_div with the two BOOLEAN gates honoured (script 24 always runs the
    grandchild check regardless of diverge)."""
    if require_divergence and require_mutual_nn:
        return safe_div(P, **kw)
    kw2 = dict(kw)
    if not require_divergence:
        kw2["diverge"] = -1e9          # admits any divergence, grandchild check remains
    return safe_div(P, **kw2)


def run_film(args):
    stem, cfg = args
    P, G = load_pred(PRED / f"{stem}.geff"), load_gt(stem)
    Gr = {"t": P["t"], "zyx": P["zyx"], "edges": list(P["edges"])}
    Gr, _ = gap_close(Gr, max_um=cfg["gc_um"], reuse_um=cfg["gc_reuse"],
                      synth_frac=cfg["gc_frac"])
    Q = dict(P); Q["t"], Q["zyx"], Q["edges"] = Gr["t"], Gr["zyx"], Gr["edges"]
    e, _ = safe_div2(Q, require_divergence=cfg["sd_req_div"],
                     require_mutual_nn=cfg["sd_req_nn"],
                     parent_max=cfg["sd_parent"], sister_max=cfg["sd_sister"],
                     tau=cfg["sd_tau"], diverge=cfg["sd_div"],
                     glob_cap=cfg["sd_gcap"])
    Gr["edges"] = e
    Gr, _ = gap2(Gr, max_total=cfg["g2_total"], max_step=cfg["g2_step"],
                 require_context=cfg["g2_ctx"], frac_cap=cfg["g2_frac"],
                 frame_frac=cfg["g2_frame"])
    Gr, _ = linefit(Gr, w=cfg["lf_w"], window=cfg["lf_win"])
    return M2.score(Gr["t"], Gr["zyx"], Gr["edges"], G["t"], G["zyx"], G["edges"], G["n_est"])


BASE = dict(gc_um=8.0, gc_reuse=3.2, gc_frac=0.05,
            sd_req_div=True, sd_req_nn=True, sd_parent=9.0, sd_sister=14.0,
            sd_tau=0.6, sd_div=2.25, sd_gcap=0.00375,
            g2_total=10.2, g2_step=4.4, g2_ctx=True, g2_frac=0.0045, g2_frame=0.006,
            lf_w=0.4, lf_win=3)

GRID = {                      # (deployed/default value, values to try)
    "g2_total":  (10.2, [6.0, 8.0, 10.2, 13.0, 16.0]),
    "g2_step":   (4.4,  [3.0, 4.4, 6.0, 8.0]),
    "g2_ctx":    (True, [False]),
    "g2_frac":   (0.0045, [0.001, 0.0045, 0.01, 0.02]),
    "g2_frame":  (0.006, [0.002, 0.006, 0.02]),
    "gc_reuse":  (3.2,  [1.5, 3.2, 6.0]),
    "gc_frac":   (0.05, [0.01, 0.05, 0.15]),
    "lf_win":    (3,    [1, 2, 3, 4, 5]),
    "sd_req_nn": (True, [False]),
    "sd_req_div":(True, [False]),
}

if __name__ == "__main__":
    pool = Pool(8)
    def ev(cfg):
        rows = pool.map(run_film, [(s, cfg) for s in STEMS])
        return M2.aggregate(rows)

    b = ev(BASE)
    print(f"BASE (gap_close 8 -> safe_div -> gap2 -> linefit 0.4/3)")
    print(f"  proxy {b['proxy']:.5f}  adj {b['adj']:.5f}  J {b['J']:.5f}  "
          f"mult {b['mult']:.5f}  divJ {b['divJ']:.4f} ({b['dtp']}/{b['dfp']}/{b['dfn']})\n")
    print(f"{'variable = value':<28}{'proxy':>9}{'d':>9}{'dJ':>9}{'dmult':>9}"
          f"{'divJ':>8}{'TP':>4}{'FP':>4}{'FN':>4}  note")
    results = []
    for key, (cur, vals) in GRID.items():
        for v in vals:
            if v == cur:
                continue
            cfg = dict(BASE); cfg[key] = v
            a = ev(cfg)
            d, dJ, dm = a["proxy"] - b["proxy"], a["J"] - b["J"], a["mult"] - b["mult"]
            note = ""
            if d > 0 and dJ <= 1e-6 and dm > 0:
                note = "MULTIPLIER-DRIVEN, exploit"
            elif abs(d) < 1e-9:
                note = "inert"
            elif d > 0.0083:
                note = "<- beats one division event"
            print(f"{f'{key} = {v}':<28}{a['proxy']:>9.5f}{d:>+9.5f}{dJ:>+9.5f}{dm:>+9.5f}"
                  f"{a['divJ']:>8.4f}{a['dtp']:>4}{a['dfp']:>4}{a['dfn']:>4}  {note}")
            results.append((key, v, d, dJ, dm, note))
    pool.close()
    inert = sorted({k for k, v, d, dJ, dm, n in results if n == "inert"})
    real = [(k, v, d) for k, v, d, dJ, dm, n in results if d > 0.0083 and "exploit" not in n]
    print(f"\nINERT at every value tested: {inert or 'none'}")
    print(f"BEATS one division event: {real or 'none'}")
