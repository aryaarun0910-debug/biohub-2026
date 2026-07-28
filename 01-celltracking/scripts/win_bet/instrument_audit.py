"""INSTRUMENT AUDIT — does any OOF statistic rank our public submissions correctly?

Availability tiers are enforced, not estimated:
  TIER-A  post-patch (scorer 075fc5f) AND deployment parity proven -> qualifies
  TIER-B  exact all-199 OOF, same edge-volume weighting, but PRE-patch and/or parity unproven
  TIER-U  no exact OOF -> excluded entirely, never estimated

Rank statistics are computed on TIER-A+B purely as a *degraded* diagnostic and are
labelled as such. With n this small they cannot calibrate anything.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field


@dataclass
class Sub:
    sub_id: str
    public: float
    family: str
    parent: str | None
    j44: float | None      # 44b6 fold composite (held-out)
    j6b: float | None      # 6bba fold composite (held-out)
    tier: str
    parity: str
    note: str = ""
    slice_worst: str = "unavailable"

    # 44b6 = 19.8k annotated edges, 6bba = 109k  (JOURNAL: 128,883 total)
    W44: float = field(default=19.8, repr=False)
    W6B: float = field(default=109.0, repr=False)

    @property
    def min_fold(self):
        if self.j44 is None or self.j6b is None:
            return None
        return min(self.j44, self.j6b)

    @property
    def edge_weighted(self):
        if self.j44 is None or self.j6b is None:
            return None
        return (self.j44 * self.W44 + self.j6b * self.W6B) / (self.W44 + self.W6B)

    @property
    def arith(self):
        if self.j44 is None or self.j6b is None:
            return None
        return (self.j44 + self.j6b) / 2

    @property
    def harm(self):
        if self.j44 is None or self.j6b is None:
            return None
        return 2 / (1 / self.j44 + 1 / self.j6b)


SUBS = [
    Sub("54290725", 0.815, "classical V3 (DoG)", None, 0.632, 0.756, "B", "unproven",
        "public recorded as 0.807 on 2026-07-03; API now reports 0.815 (post-patch rescore)"),
    Sub("54301967", 0.741, "classical op_bright", "54290725", 0.684, 0.751, "B", "unproven",
        "kernel aryaarun07/biohub-op-bright never parity-checked vs local 199-crop config"),
    Sub("54601594", 0.865, "Trackastra direct + pruning", "54534923", 0.6948, 0.6044, "B", "unproven",
        "OOF computed 2026-07-12, PRE-patch; division term changed by 075fc5f"),
    Sub("54534923", 0.889, "E0c learned wrapper", None, 0.7595, 0.6490, "A", "PROVEN",
        "15/15 diagnostics exact vs run_stats.csv on the 4 saved movies; rescored post-patch"),
    Sub("54588144", 0.889, "Trackastra hint in relinker", "54534923", None, None, "U", "n/a",
        "no OOF artifact exists (HANDOFF records '-' for both folds)"),
    Sub("54854143", 0.908, "clean v122 coupled det/ILP", None, None, None, "U", "n/a",
        "no OOF yet - this is exactly what the C0/C1 experiment must produce"),
]


def spearman(xs, ys):
    n = len(xs)
    rx = {v: i for i, v in enumerate(sorted(xs))}
    ry = {v: i for i, v in enumerate(sorted(ys))}
    d2 = sum((rx[a] - ry[b]) ** 2 for a, b in zip(xs, ys))
    return 1 - 6 * d2 / (n * (n * n - 1))


def kendall(xs, ys):
    c = d = 0
    for (a1, b1), (a2, b2) in itertools.combinations(zip(xs, ys), 2):
        s = (a1 - a2) * (b1 - b2)
        c += s > 0
        d += s < 0
    return (c - d) / (c + d) if (c + d) else float("nan")


STATS = [("min_fold", lambda s: s.min_fold),
         ("edge_weighted", lambda s: s.edge_weighted),
         ("arith_mean", lambda s: s.arith),
         ("harm_mean", lambda s: s.harm),
         ("44b6_only", lambda s: s.j44),
         ("6bba_only", lambda s: s.j6b)]


def main() -> None:
    print("=" * 108)
    print("TABLE 1 - submissions and OOF availability")
    print("=" * 108)
    hdr = (f"{'sub':<10}{'public':>7}  {'family':<28}{'44b6':>8}{'6bba':>8}"
           f"{'minfold':>9}{'edgeWt':>8}{'tier':>6}  {'parity':<9}")
    print(hdr); print("-" * 108)
    for s in SUBS:
        f = lambda v: f"{v:>8.4f}" if v is not None else f"{'--':>8}"
        mf = f"{s.min_fold:>9.4f}" if s.min_fold is not None else f"{'--':>9}"
        ew = f"{s.edge_weighted:>8.4f}" if s.edge_weighted is not None else f"{'--':>8}"
        print(f"{s.sub_id:<10}{s.public:>7.3f}  {s.family:<28}{f(s.j44)}{f(s.j6b)}"
              f"{mf}{ew}{s.tier:>6}  {s.parity:<9}")
    print("\nnotes:")
    for s in SUBS:
        if s.note:
            print(f"  {s.sub_id}: {s.note}")

    qual = [s for s in SUBS if s.tier == "A"]
    degraded = [s for s in SUBS if s.tier in ("A", "B")]
    print("\n" + "=" * 108)
    print(f"QUALIFYING (tier A, post-patch + parity proven): n = {len(qual)}")
    print("=" * 108)
    if len(qual) < 3:
        print("  -> Rank correlation is NOT COMPUTABLE on qualifying data.")
        print("     The audit as specified cannot be completed. This is the primary result.")

    print("\n" + "=" * 108)
    print(f"DEGRADED DIAGNOSTIC ONLY (tier A+B, n = {len(degraded)}) - NOT a calibration")
    print("=" * 108)
    pub = [s.public for s in degraded]
    for name, fn in STATS:
        vals = [fn(s) for s in degraded]
        if any(v is None for v in vals):
            continue
        print(f"  {name:<14} Spearman={spearman(pub, vals):+.3f}   Kendall={kendall(pub, vals):+.3f}")

    print("\n  leave-one-out sensitivity (min_fold):")
    for drop in degraded:
        keep = [s for s in degraded if s is not drop]
        p = [s.public for s in keep]; v = [s.min_fold for s in keep]
        print(f"    drop {drop.sub_id} ({drop.family[:26]:<26}) -> Spearman={spearman(p, v):+.3f}")

    print("\n  pairwise child-vs-parent sign concordance:")
    by_id = {s.sub_id: s for s in SUBS}
    for s in SUBS:
        if not s.parent or s.min_fold is None:
            continue
        p = by_id[s.parent]
        if p.min_fold is None:
            continue
        dpub = s.public - p.public
        dmf = s.min_fold - p.min_fold
        ok = "AGREE" if (dpub > 0) == (dmf > 0) else "**INVERTED**"
        print(f"    {s.sub_id} vs parent {p.parent_str() if hasattr(p,'parent_str') else p.sub_id}: "
              f"d_public={dpub:+.3f}  d_minfold={dmf:+.4f}  -> {ok}")


if __name__ == "__main__":
    main()
