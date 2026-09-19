"""Where is the remaining error, and what is each fix WORTH at its ceiling?

The question this answers: is a better DETECTOR (an SSL pretrain, a new model,
anything upstream of linking) worth building in the time left?

Source is the kernel's own validator output for s05 -- not our ports --
artifacts/s05_output/validator_results.csv (per-film edge_tp/fp/fn) and
ppsweep_results.csv, which classifies every missed GT edge into:

    edges_lost_to_detection   the node was never detected      -> DETECTOR
    edges_fragmented          nodes detected, link not made    -> LINKER
    wrong_association_edges   linked to the wrong node         -> LINKER

plus false-positive edges, which are links asserted between two correctly
matched nodes that do not exist in the ground truth -> LINKER.

For each error class the ceiling is computed by ORACLE ABLATION: set that class
to zero, hold every other class fixed, recompute micro edge Jaccard. That is the
most any fix of that kind could ever be worth, before any implementation loss.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    vr = list(csv.DictReader(open(ROOT / "artifacts/s05_output/validator_results.csv")))
    base = [r for r in vr if r["config"] == "base"]
    tp = sum(int(r["edge_tp"]) for r in base)
    fp = sum(int(r["edge_fp"]) for r in base)
    fn = sum(int(r["edge_fn"]) for r in base)

    sw = [r for r in csv.DictReader(open(ROOT / "artifacts/s05_output/ppsweep_results.csv"))
          if r["config"] == "base"][0]
    det = int(sw["edges_lost_to_detection"])
    frag = int(sw["edges_fragmented"])
    wrong = int(sw["wrong_association_edges"])
    rec = int(sw["edges_recovered"])

    print(__doc__.split("For each error class")[0].rstrip())
    print("=" * 88)
    print(f"s05, 8 validator films, kernel's own numbers")
    print(f"  edge TP {tp}   FP {fp}   FN {fn}")
    print(f"  FN breakdown: detection {det}, fragmented {frag}, wrong-assoc {wrong}"
          f"  (sum {det + frag + wrong}, reported FN {fn})")
    if det + frag + wrong != fn:
        print(f"  NOTE: breakdown sums to {det+frag+wrong}, FN is {fn}; "
              f"the classifier and the metric disagree by {abs(det+frag+wrong-fn)}. "
              f"Ceilings below use the BREAKDOWN, so treat them as +/-{abs(det+frag+wrong-fn)} edges.")
    J = tp / (tp + fp + fn)
    print(f"  micro edge Jaccard = {tp}/({tp}+{fp}+{fn}) = {J:.5f}\n")

    print(f"{'oracle fix (that class -> 0)':<34}{'edges':>7}{'new J':>10}{'ceiling':>10}  owner")
    rows = [
        ("perfect DETECTION", det, "DETECTOR  <- what SSL would target"),
        ("zero FRAGMENTATION", frag, "LINKER"),
        ("zero WRONG-ASSOCIATION", wrong, "LINKER"),
        ("zero FALSE-POSITIVE edges", fp, "LINKER"),
    ]
    out = []
    for name, n, owner in rows:
        if name.startswith("zero FALSE"):
            Jn = tp / (tp + (fp - n) + fn)
        else:
            Jn = (tp + n) / ((tp + n) + fp + (fn - n))
        out.append((name, n, Jn - J, owner))
        print(f"{name:<34}{n:>7}{Jn:>10.5f}{Jn - J:>+10.5f}  {owner}")

    det_ceiling = [d for nm, _, d, _ in out if "DETECTION" in nm][0]
    link_ceiling = sum(d for nm, _, d, _ in out if "DETECTION" not in nm)
    print(f"\n  DETECTOR ceiling (all of it) : {det_ceiling:+.5f}")
    print(f"  LINKER  ceiling (all of it)  : {link_ceiling:+.5f}"
          f"   = {link_ceiling / det_ceiling:.1f}x the detector")
    print(f"  errors owned by the linker   : "
          f"{(frag + wrong + fp)}/{(frag + wrong + fp + det)} "
          f"({100 * (frag + wrong + fp) / (frag + wrong + fp + det):.0f}%)")

    print(f"\n  for scale: gap from the 0.947 board to 7th place (0.964) is +0.017,")
    print(f"             and these are ORACLE ceilings -- a real fix captures a fraction.")


if __name__ == "__main__":
    main()
