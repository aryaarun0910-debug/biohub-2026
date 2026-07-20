# Current handoff

**Updated:** 2026-07-20
**Branch:** `master`

> **2026-07-19 scoring reset complete.** The organizer patched edge
> canonicalization and division evaluation at official commit `075fc5f`. The
> dependency is pinned, 69 focused/upstream tests pass, and the full 199-crop
> patched E0c rescore reproduced 0.7595/0.6490. No-fork and Oracle C were also
> re-run; their conclusions and values are stable across the scoring epoch.

## Objective

Win the private leaderboard while remaining prize-eligible. Public score is a
deployment signal; model selection is embryo-held-out.

## Scores and decisions

| System | Public | OOF `44b6` | OOF `6bba` | Decision |
|---|---:|---:|---:|---|
| **E0c wrapper (AUTHORITATIVE, deployment-exact)** | **0.889** | **0.7595** | **0.6484** | **THE baseline — gate all deltas vs this (min-fold 0.6490)** |
| E0b wrapper (min-len 6, no gap-refine) | 0.889 | 0.7601 | 0.6450 | superseded ablation of E0c |
| raw greedy OOF (pre-wrapper) | — | 0.6562 | 0.5593 | weak; do NOT gate against this |
| fork-suppressed organizer | — | 0.6595 | 0.5680 | old safe OOF baseline |
| Trackastra hint inside motion relinker | 0.889 | — | — | neutral; do not repeat |
| direct Trackastra + pruning | 0.865 | 0.6948 | 0.6044 | BELOW wrapper on both folds — rejected |

**E0c FROZEN (2026-07-13) — AUTHORITATIVE baseline = 0.7595 / 0.6484 (min-fold 0.6490).**
Deployment-exact wrapper (min-len 7 uniform, image gap-refine ON), parity-verified EXACT
on all 15 diagnostics across the 4 saved test movies vs `run_stats.csv`, scored over the
199 cached post-wrapper graphs (`e0c_score.py`, authoritative edge+division). Gate ALL
future deltas vs this, both folds, no regime-slice regression. Division-J ~0 (0/93/26 and
4/582/121) — low-ceiling on sparse OOF. Notes: (a) beats E0b's min-fold (0.6456→0.6490)
because min-len 7 helped 6bba; (b) numpy-vs-authoritative EDGE parity across 199 crops =
194 exact, **5 mismatch (max adj-J 0.055)** → `metric_numpy` is a fast diagnostic ONLY,
not authoritative (its 1-crop parity did not hold on the population). Earlier context: the
wrapper OOF exceeds the direct Trackastra-fusion OOF (0.6948/0.6044) on both folds, proving
fusion's "+0.035" was vs the weak greedy — Trackastra replacement is conclusively dead.

**Post-patch verification (2026-07-19):** official scorer `075fc5f` gives composite
0.7595 (44b6; division TP/FP/FN 0/93/26) and 0.6490 (6bba; 4/582/121), so the
frozen floor is unchanged. Removing every fork is neutral/-0.0002. Patched Oracle C
also reproduces add-only +0.0045/+0.0037 and add-replace +0.0182/+0.0138. The only
material division ceiling requires near-perfect joint fork detection plus assignment
conflict resolution; the failed high-precision division seed cannot realize it.

**Phase-B asset already cached** (same E0c pass, no reprocessing): 6.15M pre-assignment
candidate rows (4.82M wrapper-selected) in `artifacts/kaggle/e0c_cache/candidates/`.

**E0c HYBRID PIPELINE (deployment-exact, cached, resumable).** The old serial
`e0_replay` is superseded. Two decoupled stages, cache = `artifacts/kaggle/e0c_cache/`
(gitignored; per-crop atomic writes + status manifest => safe to interrupt/resume):

Stage 1 — exact wrapper pass (gap-refine ON, min-len 7 uniform), caches post-wrapper
graph + FULL candidate surface + diagnostics. Sharded for parallelism (start 4; raise if
RAM allows). Already-done crops are skipped by config-hash. **TO RESUME, just re-run:**
```powershell
# 4 parallel shards (one per terminal, or background):
.\.venv\Scripts\python.exe scripts\win_bet\e0c_run.py --shard 0/4
.\.venv\Scripts\python.exe scripts\win_bet\e0c_run.py --shard 1/4
.\.venv\Scripts\python.exe scripts\win_bet\e0c_run.py --shard 2/4
.\.venv\Scripts\python.exe scripts\win_bet\e0c_run.py --shard 3/4
```
Stage 2-4 — score the cache once Stage 1 is complete (fast, re-runnable):
```powershell
.\.venv\Scripts\python.exe scripts\win_bet\e0c_score.py --workers 4
```
This prints per-fold authoritative adj-edge-J + division-J + composite (the baseline),
and numpy-vs-authoritative edge parity across all crops. Record numbers in journal +
HANDOFF, mark E0c authoritative, commit. metric_numpy is a fast EDGE diagnostic only
(no divisions) — never authoritative.

Then Phase B (candidate surface already cached — no reprocessing): label the cached
candidates via scorer pred→GT matching (positive iff both endpoints match GT AND the GT
edge exists), compare the breadth model vs the wrapper's COMPOSITE decision (cost/selected,
not edge_prob alone), and gate on exact graph-level gain over E0c on both folds (not AUC).

## Current architecture direction — disciplined floor (2026-07-14 pivot)

The three-layer plan below the fold (`THREE_LAYER_WIN_ARCHITECTURE_2026-07-12.md`,
`reports/research/brain/ROADMAP.md`, `WIN_BET.md`, `SYNTHESIS.md` — all now flagged
superseded) was fully executed against the exact E0c baseline. **Every winning-scale
lever measured has failed its both-fold gate:**

| Lever | Exact result vs E0c | Verdict |
|---|---|---|
| Candidate reranking (breadth-trained scorer vs wrapper `edge_prob`) | beats wrapper on neither fold | saturated |
| Division learned posterior (leak-free, daughter-swap-invariant) | recall@P0.9 mean 0.045 (~0 on 3/4 embryos) | dead |
| Isolated de-novo detection (DAXI tracklets, Stages 2-4) | oracle recovery 0%/8.8% (need >=20%) | dead |
| Temporal accumulation v3 (exact-path controls; spatial + oracle flow) | 44b6 spatial contrast +0.000061, CI [-0.000129,+0.000330]; 6bba -0.01687, CI [-0.04569,-0.000147]; oracle lower bounds also <=0 | **dead** |

Full evidence and reasoning: [FINAL_SYNTHESIS_2026-07-14.md](reports/research/brain/FINAL_SYNTHESIS_2026-07-14.md)
(division/isolated verdicts + the addendum recording the moonshot pilot). Journal trail:
`reports/journal/JOURNAL.md`, entries 2026-07-13 through 2026-07-14.

**Current position:** keep **E0c (public 0.889, OOF 0.7595/0.6490)** frozen as the
fallback. The clean-public-0.903 wrapper delta completed all 199 crops but failed the
bilateral gate: 0.7614 (+0.0019) on 44b6 and 0.6457 (-0.0033) on 6bba, worsening the
min-fold. Do not regenerate its 0.970 detections. Kaggle kernel
`aryaarun07/biohub-daxi-cache-20` v3 completed all 20 crops in 27.8 minutes and its
outputs are downloaded. The powered temporal-signal gate is complete and negative:
`signal_gate_pass=false`. Spatial-flow accumulation is null on 44b6 and significantly
harmful on 6bba; even oracle GT motion has no positive crop-bootstrap lower bound.
Track 3 affinity training is therefore not justified.

## Immediate queue

1. Monitor submission `54854143` (clean v122); it is accepted and PENDING scoring.
   Kernel v1 completed, and the downloaded 237,023-row artifact passed every structural
   audit. SHA256: `4E36B4797C0F07FF7B3C55C8FD6C73C4E9EC5AF264C4B4F21828B1F97CA9D616`.
2. Build the coupled C0/C1 OOF cache gate: detector threshold 0.96875 vs 0.9690 with
   ILP disappearance 1.5. This is distinct from the failed clean-0.903 wrapper-only
   test on frozen 0.990 detections.
3. Prepare full-data organizer-model training, then external NIS3D/Zebrahub image
   pretraining. Follow `reports/ATTACK_REGIME_2026-07-20.md`; do not reopen temporal
   accumulation, Trackastra replacement, division posterior, or geometry reranking.
4. Keep E0c as the private-safe fallback until a new core model passes the bilateral
   +0.005 min-fold gate. Public submissions are deployment probes, not model selection.
5. Two untracked local artifact groups from earlier work are not yet
   committed: `reports/inventory/{e0c_run_*.txt,phaseB_label.txt,daxi_cache.txt}` (raw
   operational logs backing already-journaled 07-13 results) and unrelated stale WIP
   from 2026-07-03 (`reports/inventory/phase1_v3{,_smooth}.csv` full-199-crop extension,
   `.claude/settings.json`, `.gitignore`) predating the E0c pivot — confirm intent before
   committing or discarding either group.

## Canonical local artifacts

- `data/train`, `data/test`: extracted competition data.
- `artifacts/kaggle/oof_clean`: canonical learned OOF GEFFs.
- `artifacts/kaggle/trackastra_full_v6/trackastra_edges_split_{0,1}`:
  complete Trackastra candidate tables.
- `artifacts/kaggle/trackastra_full_v6/materialized_fused_pruned_*`:
  authoritative pruned fusion OOF graphs.
- `artifacts/kaggle/lb897_calibration`: 0.889 anchor artifact.
- `artifacts/kaggle/lb897_trackastra_v2`: neutral hint artifact.
- `artifacts/kaggle/lb897_trackastra_v3`: rejected direct-replacement artifact.
- `artifacts/kaggle/weights_dataset`: canonical fold-weight bundle.

## Active Kaggle references

- Anchor submission: `54534923`, score 0.889.
- Neutral hint submission: `54588144`, score 0.889.
- Direct Trackastra submission: `54601594`, score 0.865.
- Deployment kernel: `aryaarun07/biohub-lb897-trackastra-fusion`.
- Clean frontier probe: `aryaarun07/biohub-clean-v122-reproduction` v1 (COMPLETE;
  T4, internet off); submission `54854143` pending.

## Guardrails

- The unmatched-fork division evaluator pathology is diagnostic only and must
  never enter a submission.
- Exact public-source trajectory transfer into an identified hidden crop needs
  written host clearance; generic public-data pretraining is permitted.
- Preserve dirty user files unless their ownership and purpose are established.
- Do not infer hidden quality from the four visible placeholder movies.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git status --short
```
