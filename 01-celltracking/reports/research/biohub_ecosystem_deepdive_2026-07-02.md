# Royer Lab / Biohub Ecosystem — Deep Intel (Round 2, 2026-07-02)

NEW/DEEPER than the two prior reports. §1-2 read line-by-line from the host code:
`github.com/royerlab/kaggle-cell-tracking-competition` (package `tracking_cellmot` in `src/`)
and `github.com/royerlab/tracksdata`. NOTE: there is no separate `tracking-cellmot` repo.

## 1. Host "intended" solution — a SINGLE jointly-trained UNetNodeTransformer
Detection + linking end-to-end; gradients flow from transformer edge loss through
feature-indexing into the U-Net. NO ILP in baseline training or default inference
(ILP is opt-in `--use-ilp` post-processing).

### 1.1 Temporal U-Net (`models/temporal_unet.py`)
- in `(B,T,1,Z,Y,X)` -> out `(B,T,32,Z,Y,X)`. Encoder channels (32,64,128), 3 stages,
  MaxPool3d(2) before each stage but the first; conv block = [Conv3d(3^3,bias=F)->BN->ReLU]x2.
- Temporal self-attention (`nn.MultiheadAttention`, 4 heads, over time axis per voxel) after
  each encoder block EXCEPT full-res stage 0 (replaced by Identity — too costly). With
  window_size=2, temporal attention = attention over a FRAME PAIR.
- Decoder: trilinear upsample + skip concat. head=Conv3d(32,32,1). grad checkpointing on.

### 1.2 Detection head + loss
- `detect_head=Conv3d(32,1,1)` -> per-voxel logits.
- GT points -> SINGLE positive voxels at rounded/downsampled coords; else negative.
- Loss = BCE-with-logits, per-voxel weighted & count-normalized: `w_pos=1/n_pos`,
  `w_neg=neg_weight/n_neg`, reduction sum then /B. **det_neg_weight=1e-2** (negatives
  downweighted 100x) — the mechanism for sparse GT (unannotated real cells barely penalized).

### 1.3 Train-time detect->GT matching (NOT the scorer): local-max via max_pool3d,
`det_threshold=0.3`, physical NMS `pool_kernel_um=5.0`, GREEDY nearest assign at <=5 um.

### 1.4 Transformer edge scorer (`models/simple_node_transformer.py`)
- feat=unet(32)+sinusoidal pos(32)=64; hidden 128, 4 heads, 4 blocks, dropout 0.3.
- Bi-directional cross-attention (t <-> t+1). pair_mlp(concat(q_i,k_j,rel_pos/100)) -> logit (N_t,N_t1).
- **Edge loss**: `probs=softmax(logits, dim=0)` (over SOURCE => each t+1 target picks ONE parent;
  a source can win two columns => DIVISIONS allowed). Focal (gamma=2) * BCE, masked to
  rows/cols that have any GT edge (unannotated ignored).
- **LATENT LEVERS**: (a) division up-weight `weight[div_rows]=1.0` is a NO-OP (== ones) — turn up
  to chase the 0.1 division term; (b) checkpoint selection uses `acc*recall`, NOT the LB metric.

### 1.5 Defaults / recipe
epochs 50 (README says released model NOT converged, "3 epochs" example); AdamW; lr 1e-3
(train) vs 1e-4 (CLI) — DISCREPANCY; batch 16; downsample (1,4,4) [Z untouched -> near isotropic];
window_size 2; det_loss_weight 1e1 (train) vs 1e0 (CLI); det_neg_weight 1e-2; pool_kernel_um 5.
Aug MINIMAL: brightness U(-0.1,0.1) + 50%/axis flips (8 symmetries). No rot/scale/elastic/noise.
Norm: per-video quantile 0.1%-99.9% from zarr attrs `image_statistics.quantiles`.

### 1.6 Inference (`predict_unet_transformer.py`)
Default GREEDY (not ILP): sort edges by prob desc, accept respecting max_children=2 (divisions),
max_parents=1 (no merges), softmax source, threshold 0.5. det_tta (flip Y,X,XY — Z never flipped),
pool_kernel_um 3.0, `--det-threshold` default **0.99** (high, avoids over-detection FPs).
Opt-in ILP (`td.solvers.ILPSolver`) weights: edge -1.0, appearance 0.1, disappearance 0.1,
division 1.0; caps None when ILP on.

## 2. METRIC INTERNALS (for faithful numpy reimplementation)
### 2.1 Node match = per-timepoint OPTIMAL bipartite (not greedy)
Group nodes by t. Per t: scaled `cdist` (um), keep d<=max_distance, weight=`1/(1+d)`,
solve `scipy.sparse.csgraph.min_weight_full_bipartite_matching(weights, maximize=True)`
(fallback dense `linear_sum_assignment(maximize=True)` with -1 fill, drop filled). Competition:
max_distance=7.0 um, scale=(1.625,0.40625,0.40625). Each pred node <-> <=1 GT node per frame.
### 2.2 matched_edge_mask: pred edge u->v is matched iff u,v match GT g_u,g_v AND GT has g_u->g_v
(directed inner-join on remapped ids). Dedup duplicate pred edges (keep matched).
### 2.3 Edge Jaccard: edge_tp=sum(mask); GT node out_valid if outdeg>0, in_valid if indeg>0;
pred edge pred_valid = out_valid(matched src) OR in_valid(matched tgt) (FP only in ANNOTATED region);
edge_fp=pred_valid-tp; edge_fn=gt_edges-tp; J=tp/(tp+fp+fn).
adj = max(0, J*(1-0.1*(N_pred-N_true)/N_true)); N_true=GEFF extra `estimated_number_of_nodes`.
### 2.4 Division TP: per GT division subgraph, re-match full pred at 7um; TP requires: >=1 matched
pred node in the pre-split single-node stage; matched pred nodes cover >=2 distinct daughter
lineages (possibly different t => +-1 t tolerance); all in ONE weakly-connected pred component;
component has >=1 pred node with outdeg>=2. Max-cardinality bipartite pairs GT divs <-> pred forks.
FP: matched pred divs (outdeg>=2, matched GT has outdeg>=1) - tp, floored at 0. Unannotated forks ignored.
### 2.5 score = adj_edge_jaccard (weighted by w_i=tp+fp+fn per sample) + 0.1*division_jaccard (micro).
Edges dominate ~10:1.

## 3. Royer papers
- Ultrack ILP (Nat Methods 2025): binary node/appear/disappear/division(/merge) + edge vars;
  flow constraints appear+in==node+merge, disappear+out==node+division; overlap node_a+node_b<=1.
  Library defaults appear/disappear/division=-0.001, power=4, gap 0.001. (Competition ILP flag uses
  0.1/0.1/1.0.) Seg hypotheses: foreground(threshold 0.5) + contour-driven hierarchical watershed
  (min_frontier fuses weak boundaries) -> ILP picks level globally in time.
- Zebrahub (Cell 2024): DaXi single-objective light-sheet, first ~24hpf, lineages via Ultrack.
  Competition data is this ZSNS/DaXi family.
- DaXi (Nat Methods 2022): ~450nm lateral, ~2um axial => strong anisotropy (Z ~4x coarser) that the
  code hardcodes (downsample (1,4,4), Z never flipped, physical-um matching).

## 4. GEFF schema (zarr group)
nodes/ids (N), nodes/props/{name}/values(+missing); edges/ids (E,2) [src,tgt] directed,
edges/props/{name}/values. `.zattrs["geff"]`: version, directed, node/edge_props_metadata, axes,
`extra` (holds `estimated_number_of_nodes`). Read via IndexedRXGraph.from_geff; write to_geff.

## 5. public.czbiohub.org assets (offline, same domain)
- NEW: `unet-simview.pt` (84 MiB) at ultrack/unet_weights/ — 2nd pretrained U-Net; warm-start via
  `--unet-weights` (strict=False).
- ZSNS001,001_tail,002,003,004,005 OME-Zarr embryos + track CSVs (ZSNS001 849MiB, etc.; no CSV for 002)
  — same DaXi domain, dense-ish GT lineages for offline pretraining.
- tracks_benchmark/: ZSNS001_nodes.zarr/_edges.zarr/_tracks.zarr — node/edge graph benchmark.

## Highest-leverage takeaways
1. Score ~10:1 edges:div, edge Jaccard node-count-penalized (a=0.1). Keep N_pred ~ estimated_N;
   `--det-threshold 0.99` exists to avoid over-detection FPs.
2. Scoring matcher = optimal bipartite per-frame at 7um on 1/(1+d) — reimplement exactly.
3. Host-code levers: division up-weight is a no-op (turn up); checkpoint metric != LB metric (align).
4. Aug is bare + model under-trained -> easy wins from longer training + richer aug.
5. unet-simview.pt + ZSNS embryos/CSVs are same-domain offline pretraining assets.

Flagged: Ultrack/Zebrahub PDF full text not opened (403); §3 numeric details from
abstract/source/snippets. ILP structure verified from code; paper's verbatim objective not read.
