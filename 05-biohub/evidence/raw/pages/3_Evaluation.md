Submissions are evaluated using a combined tracking metric that measures both edge accuracy (how well cells are linked across time) and division detection (how well cell mitosis events are identified). The combined score is:
  
  $$\text{score} = \text{adjusted_edge_jaccard} + 0.1 \times \text{division_jaccard}$$

Edge Jaccard: Predicted nodes are matched to ground-truth nodes per timepoint via optimal bipartite assignment on scaled centroid distance (max 7.0 µm, physical scale z=1.625, y=x=0.40625 µm/voxel). A predicted edge is a true positive when both endpoints match ground-truth nodes connected by a ground-truth edge. The edge Jaccard is TP / (TP + FP + FN), adjusted by a penalty on over-predicting the total number of nodes.
  
Division Jaccard: A cell division is a node with two or more outgoing edges. For each ground-truth division, the predicted graph is checked for a connected component that covers the pre-split stage and touches both daughter lineages. Division TP/FP/FN are computed and combined into a micro-averaged Jaccard.

Per-sample adjusted edge Jaccards are weight-averaged by (TP + FP + FN); division Jaccards are micro-averaged across all samples.

Additional details about the metric can be found [here](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md). Note that cells are sparsely labeled in the ground truth, which the metric accounts for. Due to the nature of the metric, it is possible for scores to exceed 1.0.

## Submission File

Your submission CSV must contain two row types, nodes (cell detections) and edges (links between cells), grouped by dataset. The file should contain a header and have the following format:

    id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
    0,44b6,node,1,0,32,128,128,-1,-1
    1,44b6,node,2,1,33,130,125,-1,-1
    2,6bba,edge,-1,-1,-1,-1,-1,1,2
    etc.

- Node rows: `row_type=node` with `node_id`, `t`, `z`, `y`, `x` (integer centroid coordinates in voxels). Set empty `source_id` and `target_id` to `-1`.
- Edge rows: `row_type=edge` with `source_id` and `target_id` referencing node IDs. Set `node_id`, `t`, `z`, `y`, `x` to `-1`.
- The `id` column is a required throwaway index (consecutive integers).
- The dataset column must match the folder names in the test set (without the `.zarr` extension).
- Every dataset in the test set must appear in the submission.

