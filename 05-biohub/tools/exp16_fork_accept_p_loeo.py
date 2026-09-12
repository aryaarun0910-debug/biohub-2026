#!/usr/bin/env python3
"""EXP-16 -- tune fork_accept_p LEAVE-ONE-EMBRYO-OUT, with the model refitted per fold.

EXP-15 shipped a fork-acceptance model trained on BOTH embryos and picked its operating point
from a pooled sweep. Both halves of that leak. The honest protocol refits everything per fold:

    train the model on embryo A  ->  pick the threshold on embryo A  ->  report on embryo B

and takes the MIN of the two directions. Anything else is choosing on the data you then quote.
"""
import sys, os, json, gc
from pathlib import Path
import numpy as np, tracksdata as td

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from biohub.contracts import Config, SCALE
from biohub.detect import detect_oracle
from biohub.edges import score_edges
from biohub.refine import refine
from biohub.repair import repair
from tracking_cellmot.metrics import summarise
from geff import GeffMetadata
from _eval_common import load_gt, score_one

GT = Path("data/train_geff")
GRID = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]

files = sorted(GT.glob("*.geff"))
# Caching the 199 tracksdata ground-truth graphs OOM-killed a 16GB box. Only the small numpy
# arrays are cached; the graph itself is reloaded per evaluation, which costs I/O, not RAM.
CACHE = {}
for p in files:
    gt = load_gt(p); n = gt.node_attrs()
    CACHE[p] = (np.array([r["t"] for r in n.iter_rows(named=True)]),
                np.array([[r["z"], r["y"], r["x"]] for r in n.iter_rows(named=True)], float),
                (GeffMetadata.read(p).extra or {}).get("estimated_number_of_nodes"))
    del gt, n

A = [p for p in files if p.stem.startswith("44b6")]
B = [p for p in files if p.stem.startswith("6bba")]


def run(subset, cfg):
    import biohub.resolve as R
    rows = []
    for p in subset:
        t, zyx, est = CACHE[p]
        g = detect_oracle(t.copy(), zyx.copy(), p.stem)
        for st in (refine, score_edges, R.resolve, repair):
            g = st(g, cfg)
        rows.append(score_one(g, load_gt(p), est=est))
    return summarise(rows)


results = {}
for train_emb, test_emb, test_set in (("44b6", "6bba", B), ("6bba", "44b6", A)):
    os.environ["BIOHUB_FORK_MODEL"] = f"data/fork_models/trained_on_{train_emb}.json"
    import biohub.resolve as R
    R.reset_model_cache()
    print(f"\n=== model trained on {train_emb}  ->  reported on {test_emb} ===", flush=True)
    print(f"  {'p':>6} | {'TRAIN '+train_emb:>22} | {'HELD-OUT '+test_emb:>22}")
    train_set = A if train_emb == "44b6" else B
    for p_ in GRID:
        cfg = Config(fork_accept_p=p_)
        str_, ste = run(train_set, cfg), run(test_set, cfg)
        results[(train_emb, p_)] = (str_["score"], ste["score"], ste["division_jaccard"])
        gc.collect()
        print(f"  {p_:>6.2f} | score {str_['score']:.4f} divJ {str_['division_jaccard']:.4f} "
              f"| score {ste['score']:.4f} divJ {ste['division_jaccard']:.4f}", flush=True)

print("\n=== LEAVE-ONE-EMBRYO-OUT RESULT ===")
mins = []
for train_emb, test_emb in (("44b6", "6bba"), ("6bba", "44b6")):
    best_p = max(GRID, key=lambda p_: results[(train_emb, p_)][0])   # chosen on TRAIN only
    held = results[(train_emb, best_p)][1]
    print(f"  picked p={best_p:.2f} on {train_emb}  ->  HELD-OUT {test_emb} score {held:.4f} "
          f"(divJ {results[(train_emb, best_p)][2]:.4f})")
    mins.append((held, best_p))
print(f"\n  MIN across the two directions: {min(m[0] for m in mins):.4f}")
print(f"  thresholds chosen independently: {[f'{m[1]:.2f}' for m in mins]}")
ship = float(np.mean([m[1] for m in mins]))
print(f"  -> a defensible shipping value is their mean: {ship:.2f}")
