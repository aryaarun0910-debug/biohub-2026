# Error atlas — where our own system loses points, and whether the loss is structured

Data-first forensics on the two deployed LOEO exports and the pre-ILP candidate graph.
Every number below is measured locally and reproducible from the commands in §9.
`[MEASURED]` and `[INFERENCE]` are marked throughout.

---

## 0. Headline

**(a) The loss is NOT concentrated. There is no small set of crops to attack.**
Over the pooled 199-crop LOEO objective, the **worst decile (20 crops) carries 31.6 % of the
total loss** — against the 10 % it would carry if loss were uniform, and against the **19.5 % of
the edge weight those same crops already hold**. Adjusted for their size, the worst decile is
**1.62×** over-represented in loss. Making all 20 of them *literally perfect* is worth +0.0845;
lifting fold 1's worst decile to fold 1's median crop is worth +0.0337. Every one of the 20 is 6bba, and
their distinguishing property is **detection rate (median 0.785 vs 0.955 elsewhere)**, not
anything the linker does. Within crops the loss is flatter still: on 6bba the miss rate varies
between 0.19 and 0.24 across all z-bands, all time-bands and all local-density bands.
**Strategic read: this is a broad, diffuse, detector-dominated loss. A targeting strategy has
nothing to target.** `[MEASURED]`

**(b) Nothing clears the deletion-precision bar on the embryo that matters.**
Best measured deletion precision on 6bba (fold 1, 85.3 % of the pooled weight) is
**58.3 % ± 2.5 % (n = 400)** against a break-even bar of **58.88 %**, giving a simulated
**ΔSCORE = −0.00001** — indistinguishable from zero. The winning signal is a *new* one,
`rev_margin_um` (the source-side geometric margin: how much closer the chosen target is than the
next-nearest node in the next frame), and it beats everything previously tried, including
`edge_prob`. It still does not clear the bar. On 44b6 (fold 0, bar 53.08 %) the same signal
reaches **61.4 %** and is worth **+0.0012** — real, and negligible.

A 34-term crop-held-out logistic combination reaches **AUC 0.7752**, the best discrimination
measured anywhere in this study, and its deletion precision is **49.5 %** — *worse* than the
single geometric feature it contains. **AUC and the operating point are decoupled for this
metric; AUC must stop being used to select abstention signals.** `[MEASURED]`

**The abstention lane is closed.** Not "unproven" — measured at break-even, with the best signal
available, at every operating point, on 100,550 labelled scored edges.

---

## 1. Substrate and reproduction of the official numbers

`scripts/win_bet/ea_atlas.py` replays the official matching
(`tracking_cellmot.metrics.evaluate` → `tracksdata.DistanceMatching(max_distance=7.0,
scale=(1.625, 0.40625, 0.40625))`) crop by crop and dumps the intermediate tables the scorer
discards: per predicted node its matched GT id, per predicted edge its scored/TP/free status,
per GT edge whether it was recovered and whether its endpoints were detected.

Three assertions run per crop and all 327 crop-runs passed:
`edges.matched.sum() == edge_tp`, `edges.pred_valid.sum() − edge_tp == edge_fp`,
`len(gtedges) − edge_tp == edge_fn`.

| | fold 0 (44b6) | fold 1 (6bba) | pooled 199 |
|---|---:|---:|---:|
| crops | 71 | 128 | 199 |
| `adj_edge_jaccard` | 0.90180 | 0.70376 | 0.73280 |
| `division_jaccard` | 0.01515 | 0.00475 | 0.00654 |
| **SCORE** | **0.90332** | **0.70423** | **0.73346** |
| edge TP / FP / FN | 18,746 / 1,384 / 1,080 | 86,194 / 14,356 / 22,863 | 104,940 / 15,740 / 23,943 |
| division TP / FP / FN | 2 / 106 / 24 | 3 / 507 / 122 | 5 / 613 / 146 |
| GT edges | 19,826 | 109,057 | 128,883 |
| emitted edges | 1,771,878 | 1,877,129 | 3,649,007 |
| — of which **metric-free** | 1,751,748 (98.86 %) | 1,776,579 (94.64 %) | 3,528,327 (96.69 %) |

SCORE 0.90332 and 0.70423 reproduce the stated 0.9033 / 0.7042 exactly. My FN counts reconcile
with the prior `edge_loss_structure` report to within one edge (22,863 vs 22,862; reachable-
unlinked 6,861 vs 6,860), which is an independent confirmation of both instruments. `[MEASURED]`

**First structural fact: 96.7 % of what we emit is invisible to the metric.** GT annotation is
sparse, and an edge is only scored when at least one endpoint matches an annotated GT node with
non-zero degree (`pred_valid = out_valid(source) OR in_valid(target)`). 44b6 crops average ~283
scored edges; 6bba crops average ~785. Any pruner operates on 3.6 M edges to move ~120 k of
them, which is why deletion budgets must be quoted as *fraction of scored edges touched*, not
fraction of edges deleted. `[MEASURED]`

---

## 2. The exact loss identity

The pooled shortfall decomposes without approximation. With `w_i = TP_i+FP_i+FN_i` and
`r_i = (N_pred_i − N_est_i)/N_est_i`:

```
1 − adj_edge_jaccard  =  Σ_i [ FP_i + FN_i + 0.1·r_i·TP_i ]  /  Σ_i w_i
```

Verified against the scorer to 2.2e-16 on both folds. Every crop's contribution to the shortfall
is therefore **its FP count plus its FN count plus a node-count term, in edge units**, directly
comparable across crops.

| shortfall component | fold 0 | fold 1 | pooled 199 |
|---|---:|---:|---:|
| total | 0.09820 | 0.29624 | 0.26720 |
| from **FP** | 0.06525 (66 %) | 0.11632 (39 %) | 0.10883 (41 %) |
| from **FN** | 0.05092 (52 %) | 0.18526 (63 %) | 0.16555 (62 %) |
| from node-count term | **−0.01797** | −0.00534 | −0.00719 |

The node-count term is a **bonus**, not a penalty: we under-predict nodes relative to
`estimated_number_of_nodes` on both folds (median `r` = −0.085 on 6bba, −0.136 on 44b6), and that under-
prediction is currently worth **+0.018 on 44b6 and +0.005 on 6bba**. This matters because it
prices every "add more nodes" proposal: recovering detection has to overcome the loss of this
bonus. `[MEASURED]`

### 2.1 Where the FN come from

| FN bucket | fold 0 | fold 1 |
|---|---:|---:|
| FN total | 1,080 | 22,863 |
| both endpoints detected, unlinked (**linker's fault**) | 716 (66.3 %) | 6,861 (30.0 %) |
| one endpoint undetected | 227 | 4,281 |
| **both endpoints undetected** | 137 | **11,721 (51.3 %)** |
| detection ceiling (both endpoints detected) | 0.98164 | 0.85327 |

On 6bba, **the single largest bucket in the entire error atlas is "both endpoints of the GT edge
were never detected" — 11,721 edges, 9.50 % of fold-1 edge weight and 32.1 % of the entire
fold-1 shortfall.** That is a detector problem with no linker-side remedy whatsoever. `[MEASURED]`

Of the 6,861 reachable-but-unlinked FN on fold 1, **5,537 (80.7 %) are mis-links** — the
predicted node holding the GT source *did* emit an edge, just to the wrong place — and only
1,324 (19.3 %) are true abstentions where nothing was emitted. Fold 0: 574 / 142, same 80/20
split. `[MEASURED]`

### 2.2 Where the FP come from — and why deletion cannot be the lever

| FP bucket | fold 0 | fold 1 |
|---|---:|---:|
| FP total | 1,384 | 14,356 |
| both ends matched, wrong pair | 21 (1.5 %) | 91 (0.6 %) |
| exactly one end matched to a GT node | 1,363 (98.5 %) | 14,265 (99.4 %) |
| **"substitution": neither endpoint already carries a TP** | **1,278 (92.3 %)** | **13,865 (96.6 %)** |
| "surplus": the source already has its TP (a spurious fork) | 106 (7.7 %) | 491 (3.4 %) |

**Over 96 % of FP on 6bba are substitutions, not surplus.** The FP is the wrong edge that took
the place of the missing right one. FP and FN are overwhelmingly two views of a single event: a
mis-link. `[MEASURED]`

This is the mechanical reason abstention is a weak lever. Deleting a substitution FP removes it
from the denominator (+1 unit) but leaves the FN in place. *Correcting* the same edge is worth
+1 TP, −1 FP and −1 FN simultaneously. Measured single-lever oracle bounds on pooled
`adj_edge_jaccard`:

| oracle lever | fold 0 | fold 1 |
|---|---:|---:|
| delete **every** FP | 0.96475 (+0.06295) | 0.79640 (+0.09264) |
| link **every** reachable FN, keep the FPs | 0.93626 (+0.03446) | 0.75977 (+0.05602) |
| **relink**: reachable FN becomes TP *and* consumes the FP that replaced it | **0.96887 (+0.06707)** | **0.80450 (+0.10074)** |
| both levers together | 1.00162 (clips to 1) | 0.85979 |

`[MEASURED]` — these are oracles and bound what any method can win, not what any method will win.

---

## 3. Concentration — the top-decile question

Loss units per crop (`FP + FN + 0.1·r·TP`), pooled over all 199 crops:

| slice | share of loss | share of edge weight | over-representation |
|---|---:|---:|---:|
| top 5 % of crops (10) | 19.95 % | 10.35 % | 1.93× |
| **top 10 % of crops (20)** | **31.63 %** | **19.53 %** | **1.62×** |
| top 20 % (40) | 50.30 % | 34.32 % | 1.47× |
| top 50 % (100) | 85.90 % | 72.22 % | 1.19× |

Gini of loss is 0.311 on fold 1 and 0.537 on fold 0, versus a Gini of *weight* of 0.404 (f0).
**On the dominant fold the loss is barely more concentrated than the crop-size distribution
itself.** `[MEASURED]`

Counterfactuals on the pooled 199-crop objective:

- worst decile made **perfect**: 0.73280 → **0.81733** (+0.0845)
- worst decile lifted to the **median crop**: fold 1 alone 0.70376 → 0.73741 (+0.0337);
  pooled 199, 0.73280 → 0.77580 (+0.0430), though the pooled median (0.7874) sits between the two
  family modes so the pooled version of this counterfactual is the less meaningful one
- all 20 worst crops are 6bba.

Family split of the pooled objective: **6bba holds 85.33 % of the edge weight and 94.61 % of the
loss**; 44b6 holds 14.67 % / 5.39 %. 44b6 work is worth at most ~5 % of the available headroom.
`[MEASURED]`

### 3.1 Profile of the worst decile

| median over crops | worst 20 | other 179 |
|---|---:|---:|
| **GT node detection rate** | **0.785** | **0.955** |
| `adj_edge_jaccard` | 0.567 | 0.804 |
| FN share of that crop's FP+FN | 0.677 | 0.570 |
| predicted nodes | 22,272 | 15,622 |
| predicted nodes per frame | 223 | 156 |
| mean displacement (µm) | 1.485 | 1.659 |
| `total_node_ratio` | −0.061 | −0.094 |

The worst crops are **bigger and denser**, and they fail at **detection**, not at motion. The
worst crop, `6bba_6feb10f0`, scores 0.0877 with TP 121 / FP 41 / FN 1,222 — a near-total
detection collapse, not a linking error. `[MEASURED]`

---

## 4. Structure inside crops — the flatness result

Miss rate per GT edge, fold 1 (109,057 GT edges):

| z (voxel) | 0–8 | 8–16 | 16–24 | 24–32 | 32–40 | 40–48 | 48–56 | 56+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| miss rate | .207 | .213 | .196 | .210 | .199 | .193 | .234 | .240 |

| time fraction | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–.9 | .9–1 |
|---|---:|---:|---:|---:|---:|---:|
| miss rate | .230 | .205 | .213 | .203 | .207 | .213 |

| local GT density (r=15 µm) | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| miss rate | .212 | .202 | .124 | .061 |

**Flat in depth, flat in time, and *decreasing* in density.** No spatial or temporal region of the
data carries a disproportionate share. The only real gradient is displacement: `[MEASURED]`

| displacement (µm) | <1 | 1–2 | 2–3 | 3–4 | 4–5 | 5–7 | 7–10 | >10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **f1** n | 30,473 | 29,668 | 23,043 | 12,384 | 5,904 | 5,098 | 2,148 | 339 |
| **f1** miss rate | .182 | .187 | .182 | .209 | .251 | .366 | .621 | .912 |
| **f0** miss rate | .032 | .045 | .043 | .058 | .098 | .268 | .685 | 1.000 |

Real, but small in mass: the >5 µm tail holds **7.0 % of fold-1 GT edges and 15.3 % of fold-1
FN** (2.2× enrichment), and on fold 0 **2.7 % of GT edges / 22.8 % of FN**. `[MEASURED]`

**A hard linking horizon exists and is measurable.** Across 3.65 M emitted edges, essentially
none exceed ~7 µm displacement: 148 of 1,771,878 on fold 0 and 147 of 1,877,129 on fold 1
(0.008 %), with p99.99 = 6.9 µm and a global maximum of 9.31 µm. Every GT edge longer than that
is unrecoverable in practice — 339 on fold 1 at a 91 % miss rate, 45 on fold 0 at
100 %. Together worth ~0.003 of pooled `adj_edge_jaccard`. `[MEASURED]`

Divisions are the one sharply structured failure: GT nodes with out-degree 2 have a **61.2 %**
miss rate on their outgoing edges (fold 1, n = 250) and **50.0 %** on fold 0 (n = 52), versus
~21 % / ~5 % for ordinary nodes. But the population is 250 edges of 109,057 — worth ≤0.002.
`[MEASURED]`

---

## 5. The abstention question, answered

`scripts/win_bet/ea_features.py` computes, for every emitted edge, every separating feature that
does not require re-running the model — displacement, both geometric margins, candidate degrees,
local predicted-node density, motion-regime-normalised displacement, acceleration and direction
agreement against the source's own incoming edge, depth, time — and joins `edge_prob`,
`src_margin` and `rank_in_src` from the pre-ILP candidate dump.

`scripts/win_bet/ea_abstain.py` then ranks **all** emitted edges by each feature, deletes the
worst k at 14 budgets, and **recomputes the exact pooled `adj_edge_jaccard`** after the deletion.
Free deletions are metric-neutral and are excluded from the precision ratio, so
`deletion_precision = FP_deleted / (FP_deleted + TP_deleted)` is exactly the quantity the
break-even bar refers to. Break-even is derived from the substrate itself,
`p* = Σw / (Σw + ΣTP)`: **58.88 % on fold 1** (consistent with the 59.0 % design bar) and
**53.08 % on fold 0**.

### 5.1 Discrimination (AUC for ranking FP before TP, on scored edges only)

| feature | fold 0 | fold 1 |
|---|---:|---:|
| **`rev_margin_um`** (source-side geometric margin) | **0.6795** | **0.7277** |
| `nn_margin_um` (target-side geometric margin) | 0.6546 | 0.7152 |
| `disp_um` | 0.6219 | 0.5456 |
| `dens15_src` | 0.5840 | 0.5945 |
| `accel_um` | 0.5745 | 0.5332 |
| `edge_prob` (from the pre-ILP dump) | n/a | 0.6265 |
| `src_margin` / `rank_in_src` (probability margin) | n/a | 0.5196 / 0.5192 |
| logistic combination, crop-held-out, rank-transformed | 0.7104 | **0.7752** |

`rev_margin_um` and `nn_margin_um` are new here and are the strongest single signals on both
folds. Note the probability-margin features (`src_margin`, `rank_in_src`) — the direct analogue
of the 0.918 team's shipped `logitdiff` pruner — are **essentially useless at AUC 0.52**. That is
a mechanical consequence of the candidate dump: **every candidate target has in-degree exactly 1**
(2,162,040 candidate edges over 2,162,040 distinct targets), so the head has already collapsed to
a single parent per child and there is no target-side runner-up left to take a margin against.
The margin that *can* be taken — source-side, among a source's competing children — carries
almost no signal. `[MEASURED]`

### 5.2 The operating point — the number that decides the lane

Deletion precision against the bar, fold 1 (128 crops, 100,550 scored edges, bar **0.5888**):

| k deleted (of 100,550 scored) | `rev_margin_um` | `nn_margin_um` | `edge_prob` | LOGREG (AUC 0.775) |
|---|---:|---:|---:|---:|
| 25 | .520 | .520 | — | — |
| 100 | .560 | .540 | .250 | .500 |
| 200 | .555 | .530 | — | .475 |
| **400** | **.5825** | .5025 | .2475 | .4950 |
| 800 | .5737 | .474 | — | .5075 |
| 1,600 | .532 | .433 | — | .5081 |
| 3,200 | .479 | .401 | — | .4744 |

Peak is **0.5825 at k = 400** — binomial SE 0.0247, so the bar at 0.5888 sits **within one
standard error**. Simulated exact pooled effect at the best operating point found anywhere in the
sweep: **ΔadjJ = −0.00001**. `[MEASURED]`

Fold 0 (bar 0.5308) is the mirror image — the bar is lower because `J` is higher:

| k deleted (of 20,130 scored) | 25 | 50 | 100 | 200 | 400 | 800 |
|---|---:|---:|---:|---:|---:|---:|
| `rev_margin_um` precision | **.720** | **.700** | **.630** | **.565** | .458 | .334 |

Clears comfortably out to k ≈ 200, and the whole lever is worth **+0.00115** — because only 35
of 1,384 FPs are recoverable at that precision. `[MEASURED]`

### 5.3 Two findings that should change how this lane is evaluated

**AUC is the wrong selector.** The rank-transformed 34-term combination has the best AUC measured
anywhere in this study (0.7752 on fold 1) and the *worst* useful operating point of the serious
candidates (0.495 at k = 400 vs 0.5825 for one of its own inputs). The bar is a statement about
the extreme tail of the ranking; AUC integrates over the whole ranking. They are decoupled here,
empirically. This reproduces — with a stronger model — the "good AUC, bad operating point"
signature named in `edge_loss_structure_2026-08-18`. `[MEASURED]`

**Spurious forks are precisely located and still unattackable.** Every scored fork is the same
shape: of 116 scored out-degree-2 sources on fold 0, **106 are exactly one TP + one FP**; on fold
1, **491 of 595**. TP+TP occurs once in each fold. Deleting the wrong child with an oracle is
worth **+0.00487 (f0) / +0.00340 (f1)** on `adj_edge_jaccard` (and −0.0015 / −0.0005 on the
division term, since removing all forks zeroes `division_jaccard`), so **net +0.0034 / +0.0029 on
SCORE**. But no feature can pick which of the two is the FP: `[MEASURED]`

| accuracy at picking the FP within a TP/FP fork pair | fold 0 (n=106) | fold 1 (n=491) |
|---|---:|---:|
| `rev_margin_um` | .594 | **.487** |
| `disp_um` | .594 | .487 |
| `cos_prev` | .608 | .444 |
| `accel_um` | .578 | .501 |
| `edge_prob` | n/a | **.484** |
| `src_margin` | n/a | .485 |

**On fold 1, every signal is at or below chance.** Blind deletion of the second child yields
56.2 % precision (309 FP / 550 total) — itself below the 58.88 % bar — and ΔSCORE −0.00068. The fork lever is oracle-positive and
method-negative. `[MEASURED]`

---

## 6. What the solver left on the table

`scripts/win_bet/ea_solver_gap.py` scores the pre-ILP candidate graph (2,523,479 nodes /
2,162,040 candidate edges) against GT with the same matching, then compares three sets keyed on
**GT edge identity** — so no cross-run node-id join is involved.

| fold-1 GT edges (109,057) | count | share |
|---|---:|---:|
| DETECTABLE in the candidate graph (both endpoints matched) | 98,056 | 0.8991 |
| AVAILABLE (some uncapped candidate edge maps onto the GT edge) | 81,055 | 0.7432 |
| CHOSEN (TP in the deployed export) | 86,194 | 0.7904 |
| available **and** chosen | 75,226 | 0.6898 |
| **available, not chosen** ("linker headroom") | 5,829 | 0.0534 |
| detectable, **not** available ("candidate-generator headroom") | 17,001 | 0.1559 |
| chosen, **not** available (cross-run difference) | 10,968 | 0.1006 |

**A perfect linker over this candidate set is worth +0.0012** (0.70376 → 0.70498). By contrast
removing all FP is worth +0.0926 and the candidate generator is leaving 17,001 detectable GT
edges un-nominated. **The solver is not the bottleneck; the candidate set is.** This independently
confirms §3.1 of `edge_loss_structure_2026-08-18` ("the deployed final edge set contains more
reachable GT edges than the head nominated") — indeed the deployed graph recovers 86,194 GT edges,
*more* than the 81,055 the base run's candidate set even contains. `[MEASURED, with the caveat in
§8]`

Of the 6,007 correct-but-unchosen candidates, `edge_prob` median is 0.846 (25th pct 0.665) versus
0.973 (0.865) for the chosen ones — lower, but far from separable. `[MEASURED]`

### 6.1 A detector-side finding that fell out of this

The pre-ILP graph has 2,523,479 nodes; the deployed export has 1,957,978. Comparing GT-node
detection between the two runs on identical crops:

- GT nodes detected in the pre-ILP graph: **0.9119**
- GT nodes detected in the deployed export: **0.8691**
- matched in pre-ILP only: **5,381**; matched in deployed only: **548**

Only **86.45 %** of pre-ILP GT-matched node ids survive into the deployed export. The asymmetry
(5,381 vs 548) is an order of magnitude beyond plausible run-to-run noise. **The output node
filter appears to be discarding ~5,400 GT-matched nodes on fold 1 — 4.3 pp of detection
ceiling.** `[MEASURED]` — but see §8; this is the one finding here that is genuinely
cross-run-confounded, and it needs a same-run A/B before anyone acts on it. Note also that
keeping those nodes raises `N_pred` and therefore erodes the +0.005 node-count bonus, so the net
is not obviously positive. `[INFERENCE]`

---

## 7. Per-crop pattern and hidden-test triage

Per-crop `adj_edge_jaccard` spans 0.552–1.059 on fold 0 and **0.088–0.927** on fold 1
(median 0.915 / 0.726). Correlates, Spearman, within fold:

| descriptor | needs GT? | fold 0 | fold 1 |
|---|---|---:|---:|
| GT node detection rate | yes | +0.550 | **+0.676** |
| mean GT-edge displacement | yes | −0.435 | −0.130 |
| `frac_orphan` (predicted nodes with no parent) | **no** | −0.156 | **−0.428** |
| `edges_per_node` | **no** | +0.156 | **+0.428** |
| mean `nn_margin_um` over emitted edges | **no** | **−0.503** | −0.258 |
| mean `accel_um` | **no** | −0.218 | −0.255 |
| `frac_fork` | **no** | −0.226 | −0.278 |
| predicted nodes per frame | **no** | −0.270 | −0.141 |

Detection rate dominates, and it needs GT. The best **GT-free** predictor differs by fold — mean
geometric margin on 44b6 (ρ = −0.50), track fragmentation on 6bba (ρ = ±0.43). Both give
R² ≈ 0.2. **Usable for flagging which hidden-test crops are likely weak, not for deciding
anything.** Pooled-over-folds correlations look stronger (mean `accel_um` ρ = −0.478) but are
inflated by the family split — 44b6 crops are uniformly good and 6bba uniformly bad — so the
within-fold numbers are the honest ones. `[MEASURED]`

Answering the question directly: **our weakness is concentrated by family, not by crop.** 6bba is
0.7038 and 44b6 is 0.9018; within 6bba the spread is wide but unpredictable from anything we can
compute without GT. If the hidden test resembles 6bba we score ~0.70; if it resembles 44b6 we
score ~0.90; and no crop-level triage available to us narrows that. `[INFERENCE from MEASURED
correlations]`

---

## 8. Caveats

1. **The pre-ILP dump is from a different run than the deployed export.** `preilp_split1.parquet`
   was produced by the p4 kernel whose emitted graph is the p3-base graph; the scored export is
   the subvoxel arm. Node ids correspond (99.1 % of deployed ids present, 97.8 % of those
   coordinate-consistent within 3.5 µm at the same `t`), and `edge_prob` attaches to **90.5 % of
   scored fold-1 edges**. But the run difference is real and it is large: 10,968 GT edges
   (10.1 %) are chosen-but-not-available, which is **twice** the 5,829 available-not-chosen signal
   in §6. The linker-headroom figure is therefore an order-of-magnitude statement, not a precise
   one. Everything in §2–§5 and §7 is **single-run and free of this confound**; §6 and §6.1 are
   not.
2. `edge_prob`'s numbers here (AUC 0.6265, deletion precision 0.25) are **worse** than the prior
   in-run measurement (0.701 / 0.496) and the cross-run join is the likely cause. Take the prior
   in-run figures as the better estimate of `edge_prob`. It does not change the conclusion: 0.496
   is also below the bar.
3. Deletion simulations hold `num_pred_nodes` fixed. A real pruner that also drops orphaned nodes
   would change `total_node_ratio`; since `r < 0` today, dropping nodes would *increase* the
   bonus, so these Δ figures are **conservative** by a small amount.
4. Edge-deletion simulations do not recompute the division term except where stated (§5.3), where
   it is included explicitly.
5. Break-even `p*` is quoted pooled; the true objective weights per crop, so the exact bar varies
   slightly by crop. The pooled recompute in §5.2 is the authority and does not rely on `p*`.

---

## 9. Reproduction

```bat
set PYTHONIOENCODING=utf-8

REM per-edge / per-GT-edge atlas (scorer-exact; ~25 min per fold, CPU)
.venv\Scripts\python.exe scripts\win_bet\ea_atlas.py --csv c:\temp\subvoxel_f0\loeo_split0_strict.csv.gz --tag f0 --out-dir c:\temp\error_atlas
.venv\Scripts\python.exe scripts\win_bet\ea_atlas.py --csv c:\temp\subvoxel_f1\loeo_split1_strict.csv.gz --tag f1 --out-dir c:\temp\error_atlas
.venv\Scripts\python.exe scripts\win_bet\ea_atlas.py --parquet c:\temp\preilp_f1_v2\preilp_split1.parquet --tag pre1 --out-dir c:\temp\error_atlas

REM §1-§4, §7
.venv\Scripts\python.exe scripts\win_bet\ea_analyze.py --tag f0 --dir c:\temp\error_atlas --json-out c:\temp\error_atlas\analyze_f0.json
.venv\Scripts\python.exe scripts\win_bet\ea_analyze.py --tag f1 --dir c:\temp\error_atlas --json-out c:\temp\error_atlas\analyze_f1.json

REM §5 features + abstention sweep
.venv\Scripts\python.exe scripts\win_bet\ea_features.py --tag f0 --dir c:\temp\error_atlas --csv c:\temp\subvoxel_f0\loeo_split0_strict.csv.gz
.venv\Scripts\python.exe scripts\win_bet\ea_features.py --tag f1 --dir c:\temp\error_atlas --csv c:\temp\subvoxel_f1\loeo_split1_strict.csv.gz --preilp c:\temp\preilp_f1_v2\preilp_split1.parquet
.venv\Scripts\python.exe scripts\win_bet\ea_abstain.py --tag f0 --dir c:\temp\error_atlas --json-out c:\temp\error_atlas\abstain_f0.json
.venv\Scripts\python.exe scripts\win_bet\ea_abstain.py --tag f1 --dir c:\temp\error_atlas --json-out c:\temp\error_atlas\abstain_f1.json

REM §6
.venv\Scripts\python.exe scripts\win_bet\ea_solver_gap.py --dir c:\temp\error_atlas --pre-tag pre1 --dep-tag f1 --preilp c:\temp\preilp_f1_v2\preilp_split1.parquet --json-out c:\temp\error_atlas\solvergap.json
```

Scripts: `scripts/win_bet/ea_atlas.py`, `ea_analyze.py`, `ea_features.py`, `ea_abstain.py`,
`ea_solver_gap.py`. Raw tables (outside Git): `c:\temp\error_atlas\{nodes,edges,gtedges,gtnodes,
crops,cropstats,feats,sweep}_{f0,f1,pre1}.parquet`. No new Python dependency was added — the
logistic fit is a numpy Newton-IRLS in `ea_abstain.py`.

---

## 10. What this changes

**Closed by measurement.** Abstention on 6bba: the best signal available, at every operating
point, on 100,550 labelled edges, is at break-even (58.3 % ± 2.5 % vs 58.88 %) for
ΔSCORE = −0.00001. Fork pruning on 6bba: every signal at or below chance at picking which child
of a fork is wrong. Better ILP objective / better linking over the existing candidates: bounded
at +0.0012. These three should not be reopened without a genuinely new signal — and "new signal"
now means one measured at its **operating point**, not by AUC.

**Opened or re-priced by measurement.** `[INFERENCE]` from the measured decomposition:

1. **Relinking beats abstention by construction** and is the largest reachable single lever
   (+0.1007 oracle on fold 1 vs +0.0926 for delete-all-FP). 96.6 % of 6bba FP are substitutions,
   and 80.7 % of reachable FN are mis-links — the same events. Any method that *moves* an edge
   rather than *removing* it collects both halves. The break-even for relinking is far softer than
   for deletion: a relink that is right gains 3 units of the identity in §2 and one that is wrong
   costs 1.
2. **Detection is 63 % of the fold-1 shortfall and 51 % of FN is "neither endpoint detected".**
   No linker work touches it. The detection ceiling is 0.853 while we achieve 0.704, and §6.1
   suggests part of that ceiling is being thrown away by our own output filter.
3. **The node-count bonus is worth +0.005 (6bba) / +0.018 (44b6) today** and prices every
   detector-recall proposal. This should be in the objective for any such experiment.
4. **The linking horizon at ~7 µm is a hard, measured constraint** costing ~0.003.

**Cheapest next falsification** `[INFERENCE]`: a same-run A/B of the output node filter (§6.1) —
export the graph with and without node pruning from one kernel run and score both. It is the only
finding here that is confounded, it is the only one that plausibly moves detection, and it costs
one kernel run with no training.
