# Gate A — recall-ceiling re-audit at true nuclear density

**Audited:** 2026-08-07 · Raw work: `..._RESEARCH\agent_runs\v7_diagnostic_20260807\gateA\`

# VERDICT: **PROCEED**

Of the 15,296 missing GT nodes, **at most 3,410 (22.3%) are optically unresolvable** at the
measured density and the measured anisotropic PSF. Revised deployable ceiling **+0.062 (random
recovery order) to +0.069 (coherent)** at perfect precision. It stays above **+0.020 for any
added-node precision p ≥ 0.55**, and fails only at p ≤ 0.50.

**Confidence: HIGH on direction, MEDIUM on magnitude.**

---

## The density objection was half right — and it was the wrong half

The crowding measurement stands: **23–34% of the true nuclear population IS merged** with its
nearest neighbour under the measured PSF. That is physically real.

But merging almost never costs a **scored** node, because **`MAX_DISTANCE = 7.0 µm` exceeds the
entire merge scale** (50% resolvability boundary: 6bba lateral 7.0 µm, axial 11.0 µm). A merged
pair sites its single maximum *between* the two sources, which is still inside the scorer's
radius **97.7–98.3%** of the time. The metric is more forgiving than the optics.

## Merging is falsified as the cause of the misses

- **Real heatmap, `SOURCE-EXACT`, 3,079 GT centres:** **85.94%** already have a local maximum
  within 7 µm, but only **58.27%** have an **accepted** one. Of 1,285 unmatched GT, **66.30%
  still carry an unaccepted maximum inside the scorer radius.**
- **Crowding stratification runs backwards.** Accepted-within-7 µm rises **15.06% → 89.19%**
  across crowding quartiles, monotonically, a 5.9× spread — the *more* crowded, the *better*.
- **Between families:** 44b6 is 3.4× denser, 25% tighter-packed and 49% more axially overlapped
  — and misses **9.7× fewer** nodes. The optical model predicting 342 unrecoverable 44b6 nodes
  is falsified by 44b6's *entire* miss count of 276.

Within 6bba there **is** a genuine crowding–miss association (ρ = −0.45 to −0.49, CIs clear of
zero) — but those maxima are present in the heatmap and merely **unaccepted**. **That is
calibration, not optics.**

---

## Three corrections that re-price everything downstream

### 1. The 1.0000 recall ceiling was a degenerate instrument

`scripts/target_extractor_ceiling.py` max-combines **unit-amplitude** kernels and tests
`vol == pooled`. Every centre is therefore 1.0 = the global max, so **it is a peak by
construction**. Demonstrated on adversarial input using the repo's own functions: **27 nuclei
inside a single pool window still returns recall 1.0000.** The instrument could only ever fail
on exact voxel collisions, already measured at **0 of 133,318**.

`D1_DESIGN` §6's *conclusion* survives — the target semantics are correct for a *target*
question — but **the 1.0000 headline may no longer be quoted as evidence about detectability.**

### 2. `28,732 edges at stake` is a DEGREE SUM, not an edge count

Exact recomputation: **E1 = 5,922**, **E2 = 11,405**, and `E1 + 2·E2 = 28,732` **bit-exactly**.
The unique at-stake edge count is **17,327**.

> **The +0.136531 oracle in `CYCLE3_LANE_O_AND_D0.md` is inflated by 1.658× and is RETRACTED.**

**+0.10332 is unchanged and independently corroborated** (17,327 against 17,444 — 0.67% apart).

### 3. "+0.020 needs ≈2,961 recoveries" is wrong by 1.27–2.56×

That figure — mine, carried in correction C4 and in the cycle briefing — assumed each recovered
node converts independently. It does not: **65.8% of at-stake edge mass has BOTH endpoints
missing**, so an edge is only recovered when *both* of its nodes are.

At 2,961 recoveries the exact net is **+0.0078 (random order) / +0.0158 (coherent)** — neither
reaches +0.020 **even at perfect precision**.

> **Corrected requirement: ≈5,400–7,600 nodes at 70% precision.**

Recovery *order* matters, which is itself actionable: coherent recovery (both endpoints of the
same edge) is worth ~2× random recovery at equal node count.

---

## The caveat that outweighs the optical one (C7)

**44b6 holds only 2.42% of at-stake edge mass. Its entire detection oracle is +0.0025.**

If the private embryo behaves like 44b6 rather than 6bba, **the detection programme cannot reach
+0.020 at any recall or precision whatsoever.** This is irreducible risk, not a missing
experiment — no measurement on our two embryos can retire it.

---

## Method notes

A DoG image surrogate built for this audit was **rejected on its own specificity control**
(47.6–59.2% own-max on GT the real detector actually found — a ~2× merge over-call). It is
retained in the lane report precisely because it failed; the verdict rests on the real heatmap,
not on the surrogate.
