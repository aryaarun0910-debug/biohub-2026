# System review — 2026-09-13

## Experiments: 25 numbered, 23 run

| # | what it settled | verdict |
|---|---|---|
| 1 | division census | 150/151 admitted by parent≤12 / sister≤18 |
| 2–3 | duplicate sensitivity, centroid cliff | cliff between 2.0 and 2.5 µm; loss quadruples |
| 4 | submission gate | stdlib-only validator |
| 5 | oracle linking ceiling | 1:1 Hungarian misses **all** 151 divisions by construction |
| 6 | resolve validation | discriminators earn their place (at oracle density) |
| 7 / 7b | fork acceptance, LOEO | cosine gate is monotone loss → off; distance gates slack |
| 8 / 8b | candidate generation | `motion_gate_um` 10→14, **+0.0396**; surface is 1-D in effective radius |
| 9 | fork gates at wider candidates | hypothesis refuted, no change |
| 10 | repair knobs | disable both (oracle-conditional; **still open**) |
| 11 | detector sizing | compute is *not* the constraint at any sane width |
| 12 | I/O budget | 19× headroom — but contradicted by a measured 6 s/volume |
| 13 | Zebrahub geometry | 761,849 divisions; corroborates 7/7b out of domain |
| 14 | error atlas + false divisions | 96.2% of FPs are *invented*, not unannotated |
| 15 | learned fork acceptance | logistic beats best threshold held out, 0.832 vs 0.806 |
| 16 | `fork_accept_p` LOEO | honest gain +0.0059, not the +0.0087 a pooled sweep claimed |
| 17 | lost-edge recoverability | velocity/Kalman linking is **dead**; losses are GT artefacts |
| 18 | node-count economics | **confounded** — isolated nodes, one embryo, unseeded |
| 19 | ranker head-to-head | their rule is worse than its own best feature |
| 20 | offline sweep harness | works; needs division-bearing data |
| 21 | audit of their gates | gates reject **74.8%** of real divisions; predicted their recall to 0.2 pp |
| 22 | annotation structure | node-count pool (0.0912) is **unreachable** |
| 23 | gate sweep on their graphs | loosening alone **reduces** recall via crowding |
| 24 | division discriminator | *running* — precision at the margin vs 24% break-even |
| 25 | is the rejected region separable | *queued* — **the stopping condition** |

## Knowledge base

1,197 active facts (1,002 high-confidence, 75 quarantined pre-rule hints), 33 superseded,
16 retracted, 172 sources, **19 added today**. 79 experiment rows. Provenance audit clean:
every high-confidence active fact carries a source and a verbatim quote.

**2 open decisions**: repair's oracle-conditional disable, and the review's hold on paid compute.

## Code

683 lines in `src/biohub`, 4,071 in `tools`, 20 tests green. Three instruments built today:
the error atlas (per-item causal attribution), `kb.py` (one way into the base, with a provenance
audit that found 66 violations of our own rule), and `div_ci.py` (paired bootstrap — without it
the assay cannot resolve anything below ±0.0068 of score).

## Where the score actually is

```
score = edgeJ × (1.1 − 0.1·N_pred/n_total) + 0.1·divJ      max 1.1972
```

| pool | size | status |
|---|---|---|
| node count | 0.0912 | **closed** (EXP-22: annotations fill the volume) |
| divisions | 0.0769 | **narrowing** (gates closed; discriminator is the last route) |
| edge quality | 0.0740 | **mostly closed** (EXP-17: GT artefacts; `wrong_association_edges = 0`) |

**The exchange rate**: one true division is worth 3.2 false ones — any new division must be right
**>24%** of the time. Their gates admit at 75%; the gate-rejected region yields 7–12.5%.

## Kaggle

28 submissions. #27 (byte-identical repro) and #28 (ranker) pending — the first LB calibration
points we will ever have had. `v-recall` and `v-sym` complete and **deliberately not submitted**:
both failed the exchange rate on their own in-kernel diagnostics, saving two slots.

## Honest state

The 0.947 is reproduced and runs locally against our ground truth. Three independent methods agree
their division recall is 25%. Every gate-loosening route is closed, and closed for a structural
reason: a gate has no discrimination. What survives is a discriminator that can separate true from
false *inside* the rejected region — EXP-24 trains it, EXP-25 asks whether that region is separable
at all. If EXP-25 says no and #28 returns null, divisions are closed and the campaign has no
identified route above 0.947.
