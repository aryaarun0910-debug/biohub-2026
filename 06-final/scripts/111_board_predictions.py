"""Pre-registered board expectations for s05, s10, s08 -- written before results.

Baseline: the unmodified notebook scored 0.947 on the board (submission
56202098, a verbatim repro), and 653 teams sit on exactly that number.

Three independent estimates of what removing motion relink is worth, and the
reason they are NOT equally trustworthy:

 A. IN-KERNEL VALIDATOR, 8 validator films, FULL deployed chain.
    0.9491 -> 0.9715 = +0.0224.
    Right chain (every stage, incl. the three our harness omits), WRONG films.

 B. OUR CHAIN on the 4 SCORED films. 0.89061 -> 0.91768 = +0.02707.
    Right films, WRONG chain (missing single-parent repair, short-track rescue,
    DeepCenter vetoes).

 C. REBUILD-199, direction only: helps 176/199 films, weighted +0.03764.
    Wrong chain AND a weaker linker; carries direction, not magnitude.

That A and B agree at ~+0.022..+0.027 despite differing in BOTH chain and films
is the main reason to expect a real gain. The main reason to discount it: our
chain is 0.0564 below the board in absolute terms (0.89061 vs 0.947), and the
stages accounting for that gap -- single-parent repair especially -- REPAIR THE
SAME DAMAGE relink causes. They are partially redundant with removing it, so the
board gain should be at or below A, not above it.
"""
BASE = 0.947
PRIZE_7TH = 0.964
LEADER = 0.973

print(__doc__.rstrip())
print("=" * 94)
print(f"board baseline (unmodified) {BASE}   7th/last prize {PRIZE_7TH}   leader {LEADER}")
print(f"board resolution is 0.001 -- anything finer is invisible\n")

rows = [
    ("s05  remove motion relink", "0.960 - 0.970", "0.969",
     "A +0.0224 (right chain, wrong films); B +0.02707 (right films, wrong chain)"),
    ("s10  s05 + linefit 0.6", "s05 +0.002 .. +0.005", "s05 + 0.003",
     "scored films +0.00471; in-kernel +0.00207. Pure edge axis."),
    ("s08  s05 + gap2 reorder", "EQUAL to s05 (+/-0.001)", "= s05",
     "scored films +0.00000 -- those 4 have no recoverable division (0/6/3)"),
]
print(f"{'submission':<28}{'band':<24}{'central':<14} basis")
for a, b, c, d in rows:
    print(f"{a:<28}{b:<24}{c:<14} {d}")

print(f"""
WHAT EACH OUTCOME MEANS
-----------------------
s05
  >= {PRIZE_7TH}      prize territory. The lane is real and s10 is the better pick.
  > {BASE}       no-relink confirmed. Everything built today stands.
  = {BASE}       the +0.0224 in-kernel gain did NOT transfer at all. That would
                 refute the in-kernel validator as a board predictor -- the
                 instrument that rejected s01 -- and is the single most damaging
                 outcome, worse than a small loss.
  < {BASE}       lane dead. s08, s10, s11 all void; fall back to the 0.947 base.

s10  (the both-sets rule on trial)
  > s05          rule validated; it is the selection criterion from here.
  = s05          +0.00471 on 4 films was below board resolution. Uninformative,
                 not a refutation.
  < s05          rule refuted. Three tiers agreed on this change and were wrong,
                 which would mean 12 films cannot decide an edge-axis question
                 at all -- and s11's 32 films become the whole strategy.

s08  (INSTRUMENT TEST -- the important one)
  = s05          scored-film measurement VALIDATED. It predicted +0.00000 while
                 the validator films said +0.00803, and it was right. Every
                 decision now routes through that instrument, so this matters
                 more than s08's own score.
  != s05         instrument REFUTED. If a change worth +0.00803 on validator
                 films and 0.00000 on scored films moves the board, then the
                 scored films are not the target either, and the basis for
                 cancelling s09 and shipping s10 collapses.

THE ASYMMETRY WORTH NAMING
--------------------------
s08 is predicted to be WORTH NOTHING and is the most informative submission of
the three. s05 carries the score; s08 carries the epistemics.
""")
