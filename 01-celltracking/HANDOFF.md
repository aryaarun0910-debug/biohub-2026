# Current handoff

**Updated:** 2026-08-07

**Public:** **0.915** (P3 harmonic)

**GPU running:** none

**Submission ready:** none

## Decision

**Close scalar and 33-parameter-head re-acceptance. Do not launch the 199-crop D1 export
and do not submit a D1 candidate.** The completed 19-crop, four-cell cross-encoded pilot
returned `CALIBRATION_PER_CROP` in both directions and no candidate-consistent operating
point.

The key correction is substrate-level. The large apparent detection prize belongs mainly
to the sparse-trained split-1 LOEO detector, while P3 deploys split 0:

| checkpoint / role | family | GT | M | C | T | L |
|---|---|---:|---:|---:|---:|---:|
| split 1 source | 44b6 | 2,366 | 2,087 | 62 | 203 | 14 |
| split 1 target | 6bba | 9,604 | 7,173 | 736 | 1,098 | 597 |
| split 0 source | 6bba | 9,604 | 8,295 | 700 | **10** | 599 |
| split 0 target | 44b6 | 2,366 | 2,103 | 168 | **0** | 95 |

`T` is an unmatched GT node with a rejected local maximum inside 7 um. On the actual
deployment checkpoint it is essentially absent in this stratified pilot. Therefore the
previous `+0.08088` detection ceiling must not be quoted as a P3-deployment opportunity;
it was measured on fold-routed LOEO substrates.

The candidate-exact split-1 gate also kills add-only re-acceptance: H0 AUC 0.6747 and the
linear head AUC 0.8394, but both have zero recall at 0.55--0.90 Horvitz-corrected precision.
At 10% T recall the linear head reaches only about 1.6% precision, versus roughly 55%
break-even. Only 220/1,098 target T rows have an exported feature at the exact <=7 um
candidate; the original gate incorrectly used the strongest feature anywhere inside
15 um and produced impossible above-threshold "rejected" logits. That result is withdrawn
and the corrected contract is locked by tests.

## One execution sequence

1. Let the already-running probability-at-the-deployed-gate association oracle finish;
   it is resumable and owns seven CPU workers. Do not start a duplicate.
2. Re-price every remaining detection claim on the **deployed split-0 substrate**, not
   fold-routed LOEO. No new 199-crop export is authorised merely to make this table larger.
3. The next model primitive, if funded, must address fixed-count candidate replacement or
   C/L separation (accepted peaks competing for GT), not admission thresholding. A paired
   minimal training ablation is required; its unchanged control must reproduce split 0.
4. No Kaggle submission is ready. Public remains 0.915. Spend a slot only after complete
   graph inference, wrapper replay and exact scoring show a credible deployment-substrate
   gain; AUC, a GT oracle, or the split-1 T mass cannot authorise one.

## Active research question

Can a new detector objective improve the ordering/separation of the already-present local
maxima at fixed per-frame count without damaging the split-0 association representation?
The smallest defensible candidates are positive-unlabelled/ignore-aware supervision and a
centre-separation or offset head. They require a paired training smoke first. A global
threshold, learned 33-parameter replacement head, TTA expansion, recentering, and wrapper
repair are already closed.

## Verification

```powershell
git status
git rev-list --left-right --count origin/master...master
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\claims_table.py --check
```

Expected user-owned dirty files: `.claude/settings.json`, `.gitignore`.

Large pilot artifacts and fitted outputs live outside Git under:
`C:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH\`.
Historical detail remains recoverable from tag `pre-lean-2026-08-07`; do not reconstruct
it in new Markdown files.
