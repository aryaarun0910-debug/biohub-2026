---
tags:
  - stage
---

# Motion Relink

**The one large lever, and it was doing harm.** Removed in [[s05]].

Rewires edges using a two-pass distance gate: a tight pass at 6 µm, then a
relaxed pass at 10 µm.

**The mechanism of the defect:** the tight gate *forecloses* rather than defers.
A source matched to a nearer wrong target enters `used_i`, and the relaxed pass
can never revisit it. Pooling 5,540 correctly-linked GT edges, the break rate
jumps **171×** with a clean knee exactly at the 6 µm gate — 0.17% below, 28.6%
above.

Worth **+0.0224** on Kaggle's validator to delete (+0.0295 predicted locally).
An oracle per-film adaptive skip beats deletion by only 0.0014, so there is
nothing better to build than removal.

With relink off, five of the eight in-kernel sweep candidates become no-ops —
see [[In-Kernel Sweep Is Inert]].

Related: [[Ordering Bug Class]], [[Learned Bonus Is Crippled]]
