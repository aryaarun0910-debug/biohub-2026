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
5. `reports/ENVIRONMENT_TRAPS.md` — **13 defects that have each cost real time**
6. `reports/EXPERIMENT_LEDGER.md` — closed methods

## 1. Deployment state

| system | public | note |
|---|---:|---|
| **P0-B** clean 0.913 base + reverse-time (w=0.20) | **0.914** | current base |
| P0-A exact clean 0.913 reproduction | 0.913 | reproduced the public notebook exactly |
| v122 | 0.908 | previous best; best *pooled OOF* arm (0.69909) |
| E0c | 0.889 | scientific anchor only; pooled OOF 0.66539 |

`P0-B − P0-A = +0.001` = **exactly one unit of LB resolution**. The arms differ by ~0.05% of
nodes, so this cannot separate a real gain from rounding. Reverse-time is **not harmful**;
it is **not established as beneficial**. Do not re-litigate with another global blend weight.

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

## 4. What is deployable and corpus-verified

| mechanism | pooled Δ | basis |
|---|---:|---|
| node budget (keep_frac 0.975) | **+0.00157** | corpus, arm A |
| ssl × geometry veto | **+0.00141** | corpus LOFO, P(>+0.005) = 0.0004 |

That is the complete list. ~+0.003 combined *if* independent and *if* they transfer to P0-B —
neither established. Everything larger is a GT oracle.

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

**DONE 2026-07-31 — the substrate question is answered. Reach is 22/26 on the P0-A/P0-B substrate.**

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

**GREEN: 22/26 beats E0c 20/26, clean903 20/26 and v122 15/26 with node recall holding.** The 0.914
platform and the division track multiply rather than compete. Next: replay H0c/H2a on *this*
substrate (`scripts/win_bet/phaseb_h0c_replay.py`, `phaseb_h2a_hybrid_oracle.py`) and build the
deployable mother gate. The +0.06 figure remains a **GT oracle** — the selector is still unbuilt.

**Two live defects this exposed.** (a) The export cell hard-fails on out-degree > 2; it should
record and report rather than discard a completed run. (b) The pipeline can emit an out-degree-3
node — it did not occur in the P0-A/P0-B test submissions (A9 PASS, max out-degree 2), but it is a
latent structural-audit failure for any future candidate.

**Node ratio:** adj (0.89859) exceeds raw (0.88224), so the multiplier is 1.0185 and the implied
mean node ratio is **−0.1853** — this arm UNDER-predicts. But it is the `strict` arm with the
secondary model and DeepCenter OFF, so **it is not P0-B's ratio and must not be transferred.**

**Do not compare 0.89859 with E0c's published 44b6 0.7595** — weighting conventions are unreconciled.

**P0-A contains an all-training-data model** (`unet_transformer_alltrain_seed314159_v1`,
`train_datasets: 199`) blended at **0.475 detection weight**. It is a *public deployment* platform,
not OOF-valid. Only fold 0 + arm `strict` is comparable to E0c/clean903/v122.

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
