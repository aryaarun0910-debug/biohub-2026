# Current handoff

**Updated:** 2026-07-31
**Branch:** `master`
**Best public score:** **0.914** (P0-B)
**Authoritative OOF objective:** exact **pooled** composite (see §2 — this changed this cycle)

## Read in order

1. This file
2. `reports/NEXT_DECISION.md` — active queue
3. `reports/submissions/CANDIDATE_LEDGER.md` — every submission, hash, audit, score
4. `reports/RESEARCH_SYNTHESIS.md` — external evidence, one living doc
5. `reports/ENVIRONMENT_TRAPS.md` — **15 defects that have each cost real time**
6. `reports/EXPERIMENT_LEDGER.md` — closed methods

## 1. Deployment state

| system | public | note |
|---|---:|---|
| **P0-B** clean 0.913 base + reverse-time (w=0.20) | **0.914** | current base |
| P0-A exact clean 0.913 reproduction | 0.913 | reproduced the public notebook exactly |
| v122 | 0.908 | previous best; best *pooled OOF* arm (0.69909) |
| E0c | 0.889 | scientific anchor only; pooled OOF 0.66539 |
| P0-CR = v122 + reverse-time + volume guard | 0.906 | **causal probe, scored 2026-08-01** |

`P0-B − P0-A = +0.001` = **exactly one unit of LB resolution**, so it cannot separate a real
gain from rounding. Do not re-litigate with another global blend weight.

**UPDATED 2026-08-01 — reverse-time is BASE-DEPENDENT.** The same source-locked mechanism at the
same `w = 0.20` gives **+0.001 on the clean 0.913 base** and **−0.002 on v122** (0.908 → 0.906,
P0-CR). The signs differ, so it is **not a general mechanism** and must not be ported onto any
other substrate without re-measuring. The pre-registered structural read called this before any
score existed: divisions moved **−9** on the clean base and **+10** on v122, and the arm whose
divisions rose is the one that lost score. **Measure the division-count direction before
spending a slot.**

Milestones from 0.914: **0.920** needs +0.006 · **0.925** +0.011 · **0.935** +0.021 · **0.942** +0.028.

**Submissions: 8 used. Slot 3 of this cycle's 3 is HELD** — nothing deployable clears a credible margin.

## 2. THE OBJECTIVE CHANGED — read this before scoring anything

The leaderboard **pools** all samples with edge-volume weighting. It does **not** average families.
44b6 carries only **14.94%** of edge mass.

- **PRIMARY:** exact pooled composite — one combined `summarise()` over all crops.
- **DIAGNOSTICS (always report):** both family scores, per-family delta, crop-block bootstrap,
  rank-reversal risk, worst-regime behaviour.
- **Min-fold is a robustness constraint, NOT the optimisation target.**

Proof: `scripts/verify_pooled_objective.py` (parity to 2.2e-16; hand-reconstruction from raw
totals; family averaging diverges by up to +0.039). Locked by `tests/test_pooled_objective.py`.

The old **bilateral-delta gate rejected every better pooled arm** — v122 (+0.0337 pooled), C
(+0.0315), Bp (+0.0262) were all closed for losing on a 15%-mass family.

**The pooled objective is closed-form** (verified 2.27e-13 at corpus scale):
`pooled = Σ tp_i(1−0.1 r_i) / Σ (tp_i+fp_i+fn_i) + 0.1·DTP/(DTP+DFP+DFN)`.
Use it — 14.4M scorer calls become vector arithmetic (`scripts/agent5_ledger.py`).

## 3. Reporting discipline — this cycle's hardest lesson

Three headline numbers I reported shrank on the corpus:

| quantity | reported | honest | ratio |
|---|---:|---:|---:|
| H1-M pooled | ~+0.0023 | **+0.00007 / −0.00131** | ~30× |
| node budget | +0.00822 | **+0.00157** | 5.2× |
| FN association share | 63.5% | **43.3%** | 1.5× |

**RULE: no smoke or in-family probe may be quoted as a headline. Corpus numbers only, and name
the basis explicitly every time** — in-family CV vs cross-family LOFO vs GT oracle.

## 4. What is deployable on P0-B — **NOTHING. Both mechanisms died 2026-07-31.**

The previous version of this section listed node budget (+0.00157) and the ssl × geometry veto
(+0.00141) as "deployable and corpus-verified", and projected ~+0.003 combined. **Both were
measured on the actual 0.914 base and both failed.** Do not rebuild either from the old numbers.

| mechanism | old claim | measured | verdict |
|---|---:|---:|---|
| node budget (keep_frac 0.975) | +0.00157 | **−0.0000103** on P0-B; **−0.00088** corpus on arm D | **NO-GO** |
| ssl × geometry veto | +0.00141 | **+0.00104** exact corpus (E0c); min-fold **+0.000097** | **NO-GO / mis-scoped** |

**Node budget.** Like-for-like, same four movies, same frozen keep_frac, exact patched scorer:
E0c arm A **+0.006339**, P0-B **−0.0000103**. Offline parity is exact — the stage at keep_frac 1.0
reproduces the live P0-B artifact byte-identically (`4c285cae0c220a11`).
*Mechanism (corrected 2026-08-01 — it is NOT the node ratio):* the per-node count cost is
`0.1·tp_i/N_est_i`, and `N_est_i` is **GT metadata**, so it is *exactly invariant* to whether a
substrate over- or under-predicts. The ratio enters only via `w_i = 1 − 0.1·r_i`, shifting the
decision threshold by **1.1%** across the entire ±0.16 range, and per-crop ratios are **mixed-sign
on both substrates**. What actually flips the sign is the **`d_tp/d_fp` composition of the deleted
components**: E0c deletions carry **−5/+8** (`q_net −1.67`, so deleting is correct → +0.006339)
while P0-B's carry **+5/0** (`q_net +1.00`, so deleting is wrong → −0.0000103). P0-B already runs
`filter_short_track_components`, so its weakest surviving components are **correct tracks** — on the
highest-edge-weight crop the stage destroys 5 counted edge TPs and removes 0 counted edge FPs, while
on E0c the identical operation *gains* TPs (992→994) because the junk it deleted was stealing
bipartite matches. **General rule:** retain iff `q = d_tp/(d_tp+d_fp) > (Jbar + 0.1·n·ρ/a)/(w + Jbar)`,
floor **0.3994** — judge an edit by the edge quality of what it touches, never by the aggregate node ratio.
*Integrity flag:* ~**116%** of the original +0.00157 was the **count multiplier**, ~−16% edge
quality — a count effect, not a precision effect, while P0-B's own report cell declares
`"metric_hack_used": false`. Treat any revival as a metric-artifact question, not a tracking gain.

**ssl × geometry veto.** Two independent problems.

*(a) It is not a bolt-on filter.* The mechanism is the *admission gate of the H0c
division-reconstruction cascade*; deploying it means running the whole cascade at test time. The one
bolt-on reading — vetoing P0-B's existing forks — has ceiling **exactly zero**: P0-B's divisions are
TP 0 / FP 8 / FN 3, so divJ is 0 before *and* after a perfect veto.

*(b) The headline was overstated and is now measured exactly.* `h4_ssl_gate_replay.py` was run on
the full 199-crop corpus for the first time on 2026-07-31 (it had never run before — see trap 15).
On the canonical pooled objective, with the baseline arm reproducing the published anchors exactly
(44b6 0.759549, 6bba 0.648965, pooled 0.665404):

| | pooled | 44b6 | 6bba |
|---|---:|---:|---:|
| Δ composite | **+0.001042** | +0.003153 | **+0.000097** |
| P(Δ > 0) | 0.890 | 0.933 | **0.520** |
| P(Δ > +0.005) | 0.0001 | 0.207 | 0.0001 |

Divisions TP 4→9, FP 675→202, FN 147→142; edge cost −0.00102 pooled. **Min-fold is +0.000097 with
P(6bba gain) = 0.52 — a coin flip** — against a +0.005 bilateral gate, and the whole pooled effect
is carried by 44b6. The old +0.00141 was the optimistic of two arithmetic projections spanning 43×
(+0.001413 vs +0.000033); the exact scorer says **+0.001042**, so the headline was ~26% high.
Measured on the **E0c** substrate, not P0-B.

Basis, stated explicitly: four visible placeholder movies, in-sample on the public set. Sparse
annotation under-samples the edge cost while the multiplier is fully realised, so **the proxy is
biased in node budget's favour — and it still fails.** No keep_frac sweep was run.

**Consequence: there is currently no deployable mechanism between 0.914 and 0.920.** Everything
larger remains a GT oracle. The division track is the least-bad remaining route, but see §8:
its substrate advantage is **44b6-only and inverts on the 85% family**, and its selector is unbuilt.

## 5. Oracles — real ceilings, not deployable

- **H0c cascade `+0.06012` pooled** — suppress all forks, reconstruct GT-selected ones. Live-filter
  variant `+0.0641/+0.0646` per-family. Substrate-dependent: E0c reach 20/26 + 93/125; v122 only
  15/26 + 68/125; clean903 20/26 + 96/125.
- **Hybrid substrate** — v122 tracking + clean903 auxiliary division nodes lifts 6bba reachable
  divisions **68 → 101** for ~50 aux nodes. Count-multiplier objection dead (aux cost ~1.5e-6/node;
  the wrapper deletes more than we add). Reachability sets are **not nested**.

## 6. Closed, with mechanisms

- **Exploit is score-NEGATIVE** (−0.0027/−0.0007). 0.950-advertising notebooks score 0.881–0.911.
- **Detector diversity empty** — threshold variants strictly nested (0 new nodes); union of all five
  gains +0/+1 GT nodes; publicly falsified (Spotiflow 0.360, StarDist 0.505).
- **Appearance × appearance stacking closed** — all appearance scorers concentrate FPs on the *same*
  mothers (7–232× independence), measured lift **0.00**. Only appearance × geometry works, as a
  broad veto (keeps best 44%), and it is spent.
- **Zebrahub dead on lineage** — 5.7–11.7 terminations per division; oracle over all 30
  anchor×stride pairs still wrong-signed. Pre-flight filter for any future corpus:
  **require `term/div ≲ 1` and `sep(+1) ≥ 0.8 × 10.57 µm` before downloading a voxel.**
- **CTC licence-blocked**; **H1-T closed structurally** (b = 0 in 44b6 — its correct set is a strict
  subset of geometry's, so no blend can pass).

## 7. Method findings to carry forward

- **SMD audits drastically understate multivariate separability** — synthetic patches passed the
  same per-feature audit H1-M used while being 98.9% separable. Any "domain shift is tiny"
  conclusion resting on per-feature SMD is suspect.
- **FN attribution:** 56.7% never detected · **43.3% detected then discarded** by our own pipeline.
- **Break-even:** detection needs **40.6%** precision; a division action needs **10.15%** — and that
  10.15% is at *full recall*; held to ~37 of 92 divisions it stays ~10%, where H1-M delivers 2.25%.
- **E0c's published baseline contains thousands of out-of-volume coordinates**, invisible because
  the writer emits `max(0, int(round(v)))`. Volume guard exists but **defaults OFF** — enabling it
  shifts the baseline and is a decision to re-measure E0c. **Unresolved count dispute: 7,349 vs my
  recount of 14,319 (1,658 integral).**

## 8. Immediate next action

**SUPERSEDED 2026-08-01 — see the retraction below. Fold 0 said 22/26; fold 1 says 66/125.**

The previous text here said the LOEO fold-0 kernel "died after 16 of 71 crops, almost certainly on
the `/kaggle/working` size limit". **That was wrong on both cause and extent.** The log shows
`Found 71 prediction graphs` — every crop predicted — with no disk error and no OOM. It failed at
the injected export/audit cell on `44b6_a2bb48bb: out-degree > 2`, i.e. **2 nodes out of 1,900,633
(0.00011%)** carrying out-degree 3. `/kaggle/working` survives a FAILED kernel, so the whole
measurement was recovered **with zero GPU** and no shards.

| quantity (71 crops, fold 0 / 44b6) | value |
|---|---:|
| adj_edge_jaccard | 0.89859 |
| node_recall | **0.98457** |
| division_jaccard | 0.01587 (TP 2 / FP 100 / FN 24) |
| **reachable GT divisions** | **22 / 26** |

**RETRACTED 2026-08-01 BY FOLD 1. The 22/26 was 44b6 only — 14.94% of edge mass — and the
advantage INVERTS on the 85% family.**

| substrate | 44b6 | 6bba | **POOLED** |
|---|---|---|---|
| E0c | 20/26 = 0.769 | 93/125 = 0.744 | **113/151 = 0.748** |
| clean903 | 20/26 = 0.769 | 96/125 = 0.768 | **116/151 = 0.768** |
| v122 | 15/26 = 0.577 | 68/125 = 0.544 | **83/151 = 0.550** |
| **P0-strict** | **22/26 = 0.846** | **66/125 = 0.528** | **88/151 = 0.583** |

Fold 1 (128 crops, our own `split_1`, manifest audit PASS): adj_edge_jaccard **0.7025**, node_recall
**0.8547**, divJ 0.0016 (TP 1 / FP 491 / FN 124), composite 0.7026, **reach 66/125**.

**P0-strict is the BEST substrate on 44b6 and the WORST on 6bba** — below even v122 — and node
recall falls **0.9846 → 0.8547** across the family boundary. Pooled reach **58.3%** against E0c's
74.8% and clean903's 76.8%.

**What this retracts:** "the 0.914 platform and the division track multiply rather than compete",
and "the ranker, not the substrate, is now the first loss". Both were read off fold 0 alone. The
H0c **+0.073877** figure is a **44b6 GT-oracle** number and must never be pooled or extrapolated.
Honest position: **on the deployment substrate the division track has LESS pooled headroom than on
E0c, not more.**

**Process lesson.** The 44b6-only bound was stated every single time the number was quoted, and it
still produced a wrong strategic read for a whole cycle. **Naming a caveat is not acting on one.**
Fold 1 should have been ranked above the H0c replay, the cost pricing and the cascade design — all
of which were built on a 15%-mass result. When a result rests on a minority of the objective's mass,
the next action is to measure the majority, not to build on the minority.

## 9. Guardrails

- No public-score exploitation, negative-time nodes, out-of-volume nodes, synthetic hubs.
- No routing on family/crop identity; no tuning on the four visible placeholder movies.
- External data needs URL, licence, checksum, provenance. Zebrahub is CC BY 4.0; **CTC is blocked**.
- Public notebook code: competition rule **3.6.b** deems shared competition notebooks OSI-licensed
  by operation of the rules (Winner License MIT, Data CC0) — the absent API metadata is irrelevant.
- **Submissions come from NOTEBOOKS only.** A CSV upload fails `CreateSubmission` with
  `FAILED_PRECONDITION`, shown as a bare 400. Every candidate must be a completed kernel.
- Stage explicit paths when committing; agents run concurrently and `git add -A` sweeps unreviewed code.

## 10. Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q          # 30 passed
.\.venv\Scripts\python.exe scripts\verify_pooled_objective.py
git status --short
```
