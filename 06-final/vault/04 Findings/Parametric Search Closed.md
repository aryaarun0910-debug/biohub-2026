---
tags:
  - finding
  - key
---

# The Parametric Search — Closed On One Set, Reopened On Another

⚠ **This note previously said "nothing survives, do not re-tune". That was
wrong, and the reason is instructive.**

The sweeps that produced it — [[Script 100 Reprice On s09]] (21 settings) and
[[Script 101 Safediv Gates]] (7 gates) — both ran on the **VALIDATOR** films. See
[[Evidence Tiers]]. Re-running the same 35 configs on the **SCORED** films
(`scripts/106_reprice_on_scored.py`, 19 s on 16 cores) found a candidate the
validator films structurally could not see: `OUTPUT_LINEFIT_WEIGHT` 0.6,
**+0.00471 with `ratio` exactly unchanged**. That became [[s10]].

**What is actually closed:**
- Every knob, on the **validator** films — nothing survives there.
- Every knob, on the **scored** films — except linefit, now shipped as [[s10]].
- `OUTPUT_MIN_TRACK_LEN`, killed by the [[Both-Sets Rule]] — see [[MIN_TRACK_LEN Is Dead]].

**What this episode teaches**, beyond the knobs: a negative result inherits the
scope of the set it was measured on. "Nothing left to tune" meant "nothing left
to tune *on the 8 validator films*", and the wider claim was never licensed.

Two durable facts still stand:
- A **third** confirmation of the [[Transfer Lesson]] (`GAP_CLOSE_UM` 5→8:
  +0.00089 raw, −0.00015 on the s09 chain).
- [[Inert Variables]] re-confirmed after three topology changes.
- **[[Safe Division]] earns its keep on EDGES** even with zero division credit on
  the scored films: switching it off takes the ledger 0/6/3 → 0/0/3 and still
  costs **−0.00179**.

What remains is [[Upstream Knobs]], untestable locally, and better measurement —
see [[s11]].
