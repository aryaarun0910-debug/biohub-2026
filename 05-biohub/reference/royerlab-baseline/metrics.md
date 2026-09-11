# Evaluating tracking predictions

A tracking graph has nodes (cell detections with a timepoint and centroid)
and directed edges linking a cell at one timepoint to the same cell — or its
daughters — at the next. A ground-truth **cell division** has exactly two
outgoing edges. During evaluation, any predicted node with at least two
outgoing edges is treated as a predicted fork.

## Edge Jaccard

Our ground-truth annotations are **sparse**: we haven't annotated every
cell in the videos. The ground truth contains a subset of the true nodes
and the edges between those nodes.

The metric proceeds as follows:

1. **Node matching** pairs predicted nodes with ground-truth nodes by
   centroid distance, up to a maximum distance of **7 µm**. Matching
   uses an optimal bipartite assignment, so each predicted node pairs
   with at most one ground-truth node.
2. **Edge matching** counts a predicted edge as a **true positive (TP)** when
   both endpoints are matched to ground-truth nodes connected by a
   ground-truth edge. Every ground-truth edge without such a match is a
   **false negative (FN)**. A predicted edge that is not a TP is counted
   as a **false positive (FP)** in either of these cases:

   - A predicted edge with a target node that matches a GT node that is connected to another source node.
   - A predicted edge with a source node that matches a GT node that is connected to another target node.

   All other predicted edges are ignored by our metric.

The edge Jaccard is then `TP / (TP + FP + FN)`.

Because the ground truth is sparse, a correct prediction will inevitably include nodes and edges the ground truth doesn't cover. Predicted nodes that do not match a ground-truth node are not counted as false positives.
![Edge Jaccard on the `simple` example](assets/figure.svg)
<!-- *Example showing how predicted edges are labelled TP, FP, and FN against a sparse ground truth to compute the edge Jaccard.* -->

### Adjusted edge Jaccard

To penalise false-positive node predictions, the edge Jaccard is scaled by
a penalty on the total number of predicted nodes:

```
adjusted_jaccard = max(0, jaccard · (1 − a · (T_pred − T_true) / T_true))
```

where `T_pred` is the total number of predicted nodes, `T_true` is a provided
coarse estimate of the total number of true nodes (including those the ground
truth doesn't annotate), and `a = 0.1` is the weighting coefficient.

## Division Jaccard

The exact timepoint at which a cell visibly splits is somewhat subjective,
so the division metric uses a local window around each GT split:

```
grandparent → dividing parent → children → grandchildren
```

This window permits a predicted fork one timepoint before or after the GT
split without using graph-wide reachability. Predicted nodes are matched
independently against each GT division window with the same timepoint-aware,
7 µm optimal assignment used by the edge metric.

### True positives and false negatives

A predicted fork can recover a GT division only when all these conditions
hold:

- **Local parent anchor.** At least one prediction node matches the GT dividing
  parent or its immediate predecessor. The predicted fork must be one of these
  matched parent-side nodes or an immediate successor of one.
- **Two distinct daughter branches.** Each GT daughter lineage consists of one
  child and its immediate children. Each predicted branch likewise consists of
  one direct child of the fork and that child's immediate children. A bipartite
  matching must associate the two GT daughter lineages with two different
  predicted branches. The two supporting matches may occur at different
  timepoints within the window.
- **Directed local topology.** The parent anchor must be the predicted fork or
  its immediate predecessor, and the daughter matches must lie downstream of
  the fork on their assigned branches. Merely sharing a weakly connected
  component is not sufficient.
- **Valid branch evidence.** A predicted fork is rejected if two of its child
  branches have matched evidence in different reliable GT connected
  components. A matched direct child supplies its branch's component. If the
  child is unmatched, an unambiguous matched grandchild may supply it instead.
  If fallback grandchildren in one branch point to several GT components,
  that branch supplies no component evidence for the parent fork. Direct-child
  evidence takes precedence, so downstream mistakes do not invalidate
  correctly matched children.
- **Unmerged branches.** A direct child used by the fork must have that fork as
  its sole parent. When grandchild fallback is needed, each grandchild must
  belong to that child alone. A locally merged or shared branch is rejected.

Candidate GT divisions and predicted forks are then paired by a
maximum-cardinality bipartite matching. One predicted fork can recover at most
one GT division, and one GT division can receive at most one predicted fork.
Every paired GT division is a **true positive (TP)**; every unpaired GT
division is a **false negative (FN)**.

### False positives

A predicted fork that is not a TP is a **false positive (FP)** when any of the
following supplies enough evidence to evaluate it:

- the fork itself matches a GT node with outgoing edges, which means that part
  of the GT lineage is annotated;
- it is a local candidate for a GT division but fails directed topology or is
  left over after bipartite pairing;
- two distinct child branches have nearest matched evidence in different GT
  connected components; or
- its local branches merge and therefore cannot represent distinct daughter
  paths.

These categories are combined as sets of predicted fork IDs, so a fork caught
by several rules counts only once. An unmatched fork can therefore still be an
FP when its child or grandchild matches provide cross-component evidence.
Otherwise, unmatched, structurally valid forks with no local GT evidence are
ignored.

The division Jaccard is `TP / (TP + FP + FN)`.

![Division Jaccard on the `simple` example](assets/division.svg)

![Division Jaccard on the `simple` example](assets/late_division.svg)
*A predicted fork one timepoint after the ground-truth split still counts as a
TP when it satisfies the local topology rules.*

## Final score

We aggregate the results from each video into a single score by **micro-averaging**:
per-sample TP, FP, and FN counts are summed across the whole split
*before* the Jaccard is computed, so larger samples contribute
proportionally more than small ones and a sample with zero events
doesn't skew the average.

Concretely:

- **Adjusted edge Jaccard** is the per-sample adjusted Jaccard
  weight-averaged by sample size `w_i = TP_i + FP_i + FN_i`.

- **Division Jaccard** uses the summed division TP/FP/FN across all videos.

The final combined score is
```
score = adjusted_edge_jaccard + w · division_jaccard
```

with a small weight `w = 0.1` on the division term.