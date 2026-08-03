# Overnight runbook

**State at handoff:** `75ca433`, matches origin, 51 tests pass, 53 claims resolve, only
`.claude/settings.json` and `.gitignore` dirty. All compute stopped. No GPU spent, no slots spent.

Everything below is **CPU-only** and safe to run unattended. Nothing here submits to Kaggle.

---

## Run this

```powershell
cd C:\Users\aryaa\Documents\Biohub-CellTracking-2026
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe scripts\augpath_oracle.py `
    --substrate prewrapper --stride 1 --workers 6 `
    --out reports\inventory\augpath_oracle_full199.json `
    *> artifacts\c0full.log
```

**Expect 12–20 hours.** Measured rate was ~207 CPU-s/min at 5 workers with fewer than 5 of 199
crops done in the first 23 minutes; 6 workers will help somewhat. Progress prints every 5 crops
to the log, so `Get-Content artifacts\c0full.log -Tail 5` tells you where it is.

`--substrate prewrapper` is the **only** correct setting. `postwrapper` reads
`artifacts/kaggle/p0strict_cache`, which `reports/inventory/wsf_ROUTE1_VERDICT.json` rules out for
any pre-wrapper change — that mistake cost this project a full cycle.

### If you want a verdict sooner instead

```powershell
.\.venv\Scripts\python.exe scripts\augpath_oracle.py `
    --substrate prewrapper --stride 2 --workers 6 `
    --out reports\inventory\augpath_oracle_stride2.json *> artifacts\c0_s2.log
```
~100 crops, roughly half the time, both families still represented. Treat it as indicative and
still run the full corpus before acting — arm B is the reason that rule exists.

---

## Gate C0 — read the tail of the log

Pass requires **all** of:

| requirement | threshold |
|---|---|
| pooled oracle delta | ≥ **+0.020** |
| 6bba FN recovery | ≥ **12%** |
| both families | non-negative |
| cardinality | approximately fixed |
| wrapper damage | none unexplained |

The script prints every one of these plus intervention efficiency and move counts.

**If C0 PASSES** → build Lane C1 (exchange-level denominator). Not yet written. The number it must
beat is the corrected base rate: relaxed-starvation is **0.040245 (1 in 25)**, versus 1 in 834 for
the undifferentiated surface. C1 passes only with a credible route to ~45% of the **+0.032931**
association ceiling, judged on exact composite utility, not ROC-AUC.

**If C0 FAILS** → close association repair immediately and move everything to detection.

---

## Do not re-derive these — already settled this cycle, at zero GPU cost

| finding | value |
|---|---|
| arm B, correct substrate | **+0.0084877** pooled (44b6 +0.0139, 6bba +0.0077, CI [+0.00745, +0.01365]) |
| P(public flat \| true +0.0085) | **0.0322** — sampling variance is falsified |
| detection share of edge FN | **72.2%**, oracle **+0.10332** |
| association-addressable | 19.8%, ceiling **+0.032931** |
| node-ratio bonus at risk | **−0.008633** if `total_node_ratio` → 0 |
| missing GT nodes | 15,296 (11.47%); 44b6 **1.37%** vs 6bba **13.28%** |
| 6bba misses with nothing within 15 µm | **40.5%** |
| detector target-voxel collisions | **0 of 133,318** — ruled out |
| perfect-heatmap recall ceiling | **1.0000 at every sigma** — Gaussian-target line is dead |
| deployed detection loss | `neg_weight=0.1`, no ignore mask → **91.5–99.3% of real nuclei trained as background** |
| detector grid | **isotropic at 1.625 µm** — pool suppresses ±1.625 µm vs ~6.5 µm spacing, so class B is small and arms N/TN are likely dead |

Full write-ups: `reports/CYCLE2_SUBSTRATE_CORRECTION.md`, `reports/CYCLE3_LANE_O_AND_D0.md`,
`reports/D1_DESIGN.md`. `reports/CYCLE1_LANES_ABC0.md` is **retracted** — read the banner, not the
numbers.

---

## Blocked, needs you

- **D1 kernel** — designed in `reports/D1_DESIGN.md` §4, not built. The last Kaggle push this
  session was **denied by the harness permission classifier**, so either clear that permission or
  plan to run the push yourself.
- **M2** — not started. Order is now: masked/ignore-radius loss **first** (the confirmed defect),
  Gaussian targets demoted, nnPU last and blocked on having no dense validation region to search
  the class prior on.

## Standing rules that bit this cycle

- Submission bar is **≥ +0.020 pooled**; research promotion **≥ +0.015**. Nothing currently clears
  either.
- Trust **OOF ranking, not OOF magnitude**.
- Never `git add -A`; stage explicit paths. `.claude/settings.json` and `.gitignore` stay dirty.
- Kaggle outputs are archived to `../Biohub-CellTracking-2026_RESEARCH/artifacts/`, never committed.
