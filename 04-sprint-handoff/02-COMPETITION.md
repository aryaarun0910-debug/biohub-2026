# 02 THE COMPETITION

Everything here was verified against the two campaigns' own records on 2026-09-07.
Re-check the deadlines and the rules page yourself. Everything else is stable.

## The task

3D light-sheet microscopy of developing embryos. Each movie is a time series of
volumes. You must find every cell nucleus in every frame, link each nucleus to its
parent in the previous frame, and mark the divisions where one parent has two
children. The output is a lineage graph.

Two embryos, `44b6` and `6bba`, roughly 199 training movies between them. Each movie
ships as a `.zarr` volume and a `.geff` graph of annotations. The annotations are
**sparse**: only some cells are labelled, which is why no voxel and no candidate can
safely be called a negative. Both campaigns learned this the hard way and both ended
up training with positive-unlabelled losses.

## The metric

The authoritative scorer is the organizers' own package, pinned at commit
`075fc5f5a52d11077f9dc2b074644618f26939e2`. Vendor it and pin it. Do not
reimplement it as the scorer of record, because a moving `main` silently changes
your historical numbers. A pure-numpy reimplementation is useful as a fast inner
loop, but only after it agrees with the pinned source to floating-point noise.

Matching radius is **7.0 µm**. Predicted nodes match ground-truth nodes one to one
within that radius; the score is built from edge agreement and division agreement
on the matched nodes, with a penalty tied to an estimate of the total node count.
An edgeless prediction is not scorable, so a graph with no edges is a bug, not a
low score.

## The submission

- A **Kaggle notebook**, not a file upload.
- Runtime cap **12 hours**. Internet **disabled**. Output must be `submission.csv`.
- Every test dataset must be covered, with valid integer voxel coordinates and edge endpoints that reference real nodes.
- Roughly five submissions per day. Two concurrent GPU sessions maximum. The GPU is a **T4 x2**, compute capability 7.5.

CSV schema, from a submission that was accepted:

```
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6_0113de3b,node,0,0,20,180,84,-1,-1
```

Node rows carry the coordinates and `-1` for the edge columns. Edge rows carry
`source_id` and `target_id` and `-1` for the coordinates. Coordinates are integer
voxels in `(z, y, x)`.

**Because the notebook runs on Kaggle, the model must run on CUDA or CPU there.**
This is the constraint that rules MLX out of the submission path entirely. See 04.

## Deadlines

| | |
|---|---|
| Entry deadline | 2026-09-22 23:59 UTC |
| Final deadline | 2026-09-29 23:59 UTC |

The entry deadline usually also governs team mergers. A submission already went in
under this account, so entry should be satisfied, but confirm it.

## The test split, and the trap in it

The visible `test/` directory contains only **four placeholder movies**, which the
host confirmed are byte-identical copies of training movies, replaced at rerun.

Never select a model, a threshold or anything else on them. Both campaigns wrote
that prohibition into their rules independently. A number measured on those four
movies is a proxy for nothing and will not survive the rerun. Read them for
inference plumbing only.

## Platform traps that already cost GPU sessions

Each of these was discovered by paying for it once. They are properties of the
platform, not beliefs about the science, so they carry forward without any caveat.

- **`torch.autocast` is thread-local.** With `DataParallel` on T4 x2 it does not reach the replica threads, so mixed precision silently does not apply where you think it does.
- **`polars`' compiled backend imports cleanly on Kaggle and then dies on first call.** Import success is not a working dependency.
- **`np.savez_compressed` appends `.npz` to the filename.** If you built an atomic write around an exact path, it breaks after the GPU work is finished, not before.
- **Kaggle slugifies the kernel title, not the id.** A build can therefore create a kernel you did not intend. Check the resolved id before pushing.
- **Notebook-only competitions return a truncated 400 on `CreateSubmission`** if you try to submit a file directly. It is not a transient error.
- **`kaggle.exe` was blocked by Windows App Control** on the old laptop. Irrelevant on macOS, noted so nobody re-diagnoses it.

## Public notebooks are contaminated

At least ten highly upvoted public notebooks inject a hub node at `t = -1000` to
game the metric. One ships it enabled by default. If you reproduce a strong public
notebook as a baseline, you will inherit the exploit without noticing, and your
local score will not mean what you think it means.

Check any borrowed pipeline for nodes at implausible timepoints before believing
any number it produces.
