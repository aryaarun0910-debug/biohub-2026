---
tags:
  - rule
  - key
---

# The Both-Sets Rule

> **A change ships only if it is positive on BOTH the scored films and the
> validator films — and does not gain by dropping `J` on either.**

Written after [[s09]], which shipped on the validator films alone (+0.00487) and
is **−0.00086** where it counts.

**It has already earned its keep twice:**

- **[[s10]] passes.** Linefit weight on the s05 chain: 0.2 and 0.3 are
  validator-only, 0.7 is scored-only, **0.4/0.5/0.6 are positive on both**, and
  0.6 maximises the worst case. A third tier later agreed — see
  [[Relink Across 199]] and the 0.4–0.6 plateau.
- **`OUTPUT_MIN_TRACK_LEN` fails.** See [[MIN_TRACK_LEN Is Dead]]. It gains proxy
  on both sets at L=7..10 but `J` **falls on the validator set every time**. The
  two sets disagree about the *mechanism*, which the rule catches and a proxy
  comparison would not.

**The second clause matters as much as the first.** Proxy can rise on both sets
while the gain is the [[Node Count Exploit]] on one of them. Require `J` up, or
`ratio` unchanged, not just proxy up.

Related: [[Evidence Tiers]], [[Operating Rules]]
