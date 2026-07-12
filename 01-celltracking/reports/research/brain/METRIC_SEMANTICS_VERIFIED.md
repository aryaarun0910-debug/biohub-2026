# Metric semantics verified against the vendored scorer

**Date:** 2026-07-12  
**Code:** `vendor/kaggle-cell-tracking/src/tracking_cellmot/metrics.py` and
`division_metrics.py`.

This file separates scorer facts from strategic inference. Where a strategy
depends on calibration, hidden metadata or transfer, it is labelled an experiment.

## Edge score

The combined score is:

```text
adjusted_edge_jaccard + 0.1 * division_jaccard
```

Per sample:

```text
adjusted_J = max(0, J * (1 - 0.1 * (N_pred - N_est) / N_est))
N_pred = graph.num_nodes()
```

`N_est` is the GEFF `estimated_number_of_nodes`. It is available for labelled
training crops but **not supplied for hidden test movies at inference**. Therefore:

- `N_pred=N_est` is the neutral-multiplier point during OOF analysis;
- it is not a deployable hidden count target;
- deployment needs a global scale-free policy or a count estimator based only on
  images/graph statistics.

The multiplier is not clipped above 1. Under-prediction is rewarded if raw Jaccard
is preserved; `N_pred=N_est` is not necessarily the mathematical optimum.

## Matching

Tracksdata performs optimal bipartite distance matching at each time point with:

```text
max_distance = 7 microns
scale = (1.625, 0.40625, 0.40625)
```

Correspondence is one-to-one under the bipartite matching. Extra nodes can still
hurt by stealing a match from a node whose incident edges would otherwise score.

## Counted and ignored predicted edges

A predicted edge is metric-valid when at least one matched endpoint corresponds
to an annotated GT node with a relevant incoming/outgoing GT edge. A prediction
whose endpoints are genuinely off annotation contributes neither TP nor counted
FP to edge Jaccard.

This does **not** make arbitrary off-annotation proposals harmless:

- every node changes `N_pred` and the multiplier;
- nodes can steal bipartite assignments;
- an edge believed to be off annotation may match an annotated endpoint;
- graph constraints can displace a better edge.

Pruning can improve score through three mechanisms: fewer counted-FP edges,
fewer assignment thieves and a better count multiplier.

## Division score

Division Jaccard is:

```text
TP_div / (TP_div + FP_div + FN_div)
```

Its contribution has a mathematical range of **0 to +0.1**. A realistic system
may earn only +0.01 to +0.05, but +0.025 is not a scorer-imposed ceiling.

The evaluator counts a predicted dividing node as an FP only when it matches an
annotated GT node that continues. Forks in unannotated regions or at annotation
ends may be ignored. This is a scorer fact, not permission to create artificial
forks. The unmatched-fork/weak-component pathology remains quarantined and must
not enter a submission.

## What follows strategically

Verified:

1. Component and node selection materially affect the metric.
2. Off-annotation edge proposals can be ignored by edge Jaccard.
3. Optimal matching creates assignment interactions.
4. Division can contribute as much as +0.1.

Experiments, not verified facts:

1. A global Dinkelbach lambda may improve edge/component selection.
2. TP/counted-FP/ignored probabilities may transfer across embryos.
3. A scale-free global pruning policy may outperform the wrapper.
4. An image-derived count estimator may recover part of the hidden count signal.

Any Dinkelbach experiment must exclude explicit hidden `N_est` dependence and be
evaluated against the full 0.889-wrapper OOF baseline.
