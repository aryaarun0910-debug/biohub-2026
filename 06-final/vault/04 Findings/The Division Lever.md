---
tags:
  - finding
  - key
---

# The Division Lever — the only arithmetic that reaches 0.975

**The gap to first place is a division problem, not an edge problem.**

Using the in-kernel validator's own internally-consistent numbers
(`proxy 0.9491 = adj 0.9260 + 0.1 × divJ 0.2308`, matching the 0.947 board):

| | |
|---|---|
| division term now | 0.1 × 0.2308 = **0.0231** |
| division term at divJ = 1 | 0.1000 |
| **untapped headroom** | **0.0769** |

| to reach | via divisions | via edges |
|---|---|---|
| 0.964 (7th) | divJ 0.23 → **0.38** | adj +0.0149 |
| **0.975 (1st)** | divJ 0.23 → **0.49** | adj +0.0259 |

The edge route needs +0.026 when [[Error Budget]]'s oracle ceiling for
eliminating **all** edge error is +0.049 — 53% of every remaining mistake,
detector and linker combined. Implausible. The division route needs divJ 0.23 →
0.49, and the candidate graph already covers **88% of division forks**.

**The field is stuck in the same place.** A competitor spent a submission to
isolate it: 0.929 → 0.907 with divisions off, so **divJ ≈ 0.22** — essentially
ours. Their observation is the important one: *two people on the same 0.948 can
have divJ 0.30 or 0.15 and need completely different things*. **Whoever is at
0.973 is almost certainly there on divisions.**

## Why our attempt failed, and what changed

[[Failed Division Classifier]] hit **AUC 0.456 — chance** — fit to ~304 real
division events on frozen features. That is a data problem, not a method problem.

**The CC0 synthetic set (discussion `732103`) has 165,267 divisions.**

**A contradiction, now resolved by inspection.** hikaggler reported synthetic
"buys nothing on divisions — the generator doesn't model mitosis over time".
**That is false for this dataset:**

```
metadata.json: n_sequences 2174, seq_len 6, total_divisions 165267,
               total_nodes 4056226, division_rate 0.0407 (real rate 0.26%)
               sister_separation_um 7.24, median_step 1.86 um/frame
```

Inspecting `seq_0018.npz` directly: 6 frames, node counts growing 187 → 236, and
**49/49 flagged division parents have exactly 2 outgoing edges into the next
frame**. Mitosis *is* time-resolved. Do not inherit that claim — it was checkable
and it was wrong.

Detectability is calibrated too: a DoG detector recovers ~0.89 of synthetic
nuclei against 0.91–0.94 on real ones, so synthetic is slightly **harder**, and
the pooling matches the evaluator's stride exactly.

⚠ hikaggler did find **497 sequences beat all 2,174** for pretraining. More is
not better.

Related: [[Training Recipe Research]], [[Divisions Played Out]], [[Error Budget]]
