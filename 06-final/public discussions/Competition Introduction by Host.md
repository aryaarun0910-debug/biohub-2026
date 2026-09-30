# Welcome to the Biohub — Cell Tracking During Development Challenge

Hello everyone,

We at **Biohub** are excited to present this challenge to you.

Advances in **machine learning** and **computer vision** have transformed many areas of science, ranging from image analysis to biological discovery.

These techniques are increasingly being applied to complex microscopy datasets, where even expert researchers face major challenges involving:

- Dataset scale
- Imaging noise
- Biological complexity
- Dense cellular populations
- Complex spatial and temporal structures

---

# Motivation

Tracking cells across time in **3D microscopy** is a fundamental problem in biological research.

Scientists use **time-lapse 3D imaging** to study how cells:

- Grow
- Move
- Interact
- Divide
- Form complex biological structures

However, analyzing these datasets remains a major bottleneck.

In many studies, researchers still spend countless hours manually:

1. Detecting cells.
2. Linking cells across frames.
3. Identifying division events.
4. Reconstructing cellular lineages.

This becomes particularly difficult when datasets contain thousands of visually similar cells that move, deform, interact, and divide over time.

---

# Challenges with Existing Methods

Automated cell-tracking tools already exist, but they often struggle under real-world conditions.

Common challenges include:

- Dense cell populations
- Imaging noise
- Irregular cell shapes
- Cell divisions
- Cell deformation
- Complex 3D structures
- Large numbers of visually similar neighboring cells

These conditions can result in:

- Incorrect cell associations
- Broken tracks
- Missed detections
- Incorrect division events
- Incorrect lineage reconstruction

Tracking errors ultimately limit scientists' ability to scale their analyses and reliably compare results across datasets and experiments.

---

# Competition Goal

This challenge is based on real microscopy datasets from **developing zebrafish embryos**.

The goal is to establish a shared benchmark for robust **3D+time cell tracking**.

Participants are tasked with developing algorithms capable of:

1. **Detecting cells**
2. **Associating cells across time**
3. **Identifying cell division events**
4. **Reconstructing accurate cell lineages**

The organizers hope the challenge will encourage the development of new methods that are:

- Reliable
- Generalizable
- Scalable
- Robust to real microscopy conditions
- Useful for biological research

---

# Getting Started

The organizers provide a Python library for working with the competition datasets.

It includes functionality for:

- Loading 3D+time datasets
- Training a small baseline model
- Running inference
- Visualizing ground truth
- Visualizing model predictions
- Understanding the evaluation metric
- Computing competition metrics

Repository:

[Biohub Cell Tracking Competition Toolkit](https://github.com/royerlab/kaggle-cell-tracking-competition)

---

# Data

By **node and edge count**, this is the largest cell-tracking dataset published to date.

However, the provided ground-truth annotations are **sparse**.

This creates an important challenge:

> Participants must develop methods that can learn effectively from sparsely annotated datasets.

During inference, models are expected to track **all cells present in the videos**.

However, submissions are evaluated using a **random sparse subset** of those cells.

Conceptually:

```text
Full Video
    │
    ▼
All Cells Must Be Tracked
    │
    ▼
Complete Predicted Tracking Graph
    │
    ▼
Evaluation Against
Random Sparse Ground Truth
```

Therefore:

```text
Unlabeled cell ≠ Negative cell
```

A missing ground-truth annotation does not imply that no cell exists at that position.

---

# Open Problems and Research Ideas

## 1. Data Perspective

Participants are strongly encouraged to explore the **Cell Tracking Challenge**:

[Cell Tracking Challenge](https://celltrackingchallenge.net/)

The Cell Tracking Challenge is a widely used benchmark in the field and contains numerous:

- 2D+time datasets
- 3D+time datasets
- Cell detection benchmarks
- Cell segmentation benchmarks
- Cell tracking benchmarks

It also provides comparisons of state-of-the-art approaches to cell tracking.

These datasets may be useful for:

- Understanding established tracking approaches
- Studying failure cases
- Comparing evaluation methodologies
- Exploring external training data
- Investigating transfer learning
- Pretraining models before adapting them to the Biohub dataset

---

# Useful Packages

Several packages may be useful when developing solutions for the competition.

## GEFF

**Graph Exchange File Format**

Repository:

[geff](https://github.com/live-image-tracking-tools/geff)

Potential use:

- Reading tracking graphs
- Writing tracking graphs
- Handling graph-based cell lineage data

---

## Zarr

Documentation:

[Zarr](https://zarr.readthedocs.io/en/stable/)

Potential use:

- Loading competition image volumes
- Working efficiently with chunked multidimensional arrays
- Processing large 3D+time microscopy datasets

The competition image data uses **Zarr v3**.

---

## tracksdata

Repository:

[tracksdata](https://github.com/royerlab/tracksdata)

Potential use:

- Representing tracking data
- Manipulating trajectories
- Working with graph-based cell tracks

---

## CuPy

Documentation:

[CuPy](https://docs.cupy.dev/en/stable/index.html)

Potential use:

- GPU-accelerated NumPy-style operations
- Fast numerical processing
- GPU-based image processing pipelines

---

## cuCIM

Repository:

[cuCIM](https://github.com/rapidsai/cucim)

Potential use:

- GPU-accelerated image processing
- Large-scale multidimensional image analysis
- CUDA-based scientific imaging pipelines

---

## napari

Website:

[napari](https://napari.org/)

> Native usage only.

Potential use:

- Interactive visualization of microscopy data
- Inspecting 3D volumes
- Visualizing detections
- Inspecting tracking errors
- Exploring predicted cell lineages

---

## ndv

Repository:

[ndv](https://github.com/pyapp-kit/ndv)

Potential use:

- Multidimensional image visualization
- Interactive exploration of scientific image data

---

## motile

Repository:

[motile](https://github.com/funkelab/motile)

Potential use:

- Global cell tracking
- Graph optimization
- Cell association
- Lineage reconstruction

---

# Competitive Tracking Methods

The organizers highlight several existing tracking frameworks and research methods that may be useful references.

---

## Ultrack

Repository:

[ultrack](https://github.com/royerlab/ultrack)

A cell-tracking framework designed for microscopy data.

Potential areas to investigate:

- Global tracking
- Segmentation-to-track pipelines
- Graph optimization
- Cell division handling
- Large-scale tracking

---

## Trackastra

Repository:

[trackastra](https://github.com/weigertlab/trackastra)

Potential areas to investigate:

- Learning-based cell association
- Transformer-based tracking
- Temporal relationships
- Cell lineage reconstruction

---

## ByoTrack

Repository:

[byotrack](https://github.com/raphaelreme/byotrack)

Potential areas to investigate:

- Modular tracking pipelines
- Combining detection and association methods
- Benchmarking different trackers

---

## LapTrack

Repository:

[laptrack](https://github.com/yfukai/laptrack)

Potential areas to investigate:

- Linear assignment
- Frame-to-frame association
- Gap closing
- Cell division handling

---

## CELLECT

Repository:

[CELLECT](https://github.com/zzz333za/CELLECT)

Potential areas to investigate:

- Learning-based cell tracking
- Cell association
- Lineage reconstruction

---

## ASCENT

Repository:

[ascent](https://github.com/lu-lab/ascent)

Potential areas to investigate:

- Representation learning
- Self-supervised learning
- Cell tracking
- Learning temporal cell features

---

## OrganoidTracker

Repository:

[OrganoidTracker](https://github.com/jvzonlab/OrganoidTracker)

Potential areas to investigate:

- 3D cell tracking
- Organoid lineage reconstruction
- Cell division tracking
- Manual and automated lineage analysis

---

# Resource Summary

| Resource | Primary Purpose |
|---|---|
| [Competition Toolkit](https://github.com/royerlab/kaggle-cell-tracking-competition) | Baselines, inference, visualization, metrics |
| [Cell Tracking Challenge](https://celltrackingchallenge.net/) | External datasets and benchmarks |
| [GEFF](https://github.com/live-image-tracking-tools/geff) | Tracking graph format |
| [Zarr](https://zarr.readthedocs.io/en/stable/) | Multidimensional array storage |
| [tracksdata](https://github.com/royerlab/tracksdata) | Tracking-data manipulation |
| [CuPy](https://docs.cupy.dev/en/stable/index.html) | GPU numerical computing |
| [cuCIM](https://github.com/rapidsai/cucim) | GPU image processing |
| [napari](https://napari.org/) | Interactive microscopy visualization |
| [ndv](https://github.com/pyapp-kit/ndv) | Multidimensional visualization |
| [motile](https://github.com/funkelab/motile) | Tracking optimization |
| [Ultrack](https://github.com/royerlab/ultrack) | Cell tracking framework |
| [Trackastra](https://github.com/weigertlab/trackastra) | Learning-based tracking |
| [ByoTrack](https://github.com/raphaelreme/byotrack) | Modular tracking framework |
| [LapTrack](https://github.com/yfukai/laptrack) | Assignment-based tracking |
| [CELLECT](https://github.com/zzz333za/CELLECT) | Cell tracking research method |
| [ASCENT](https://github.com/lu-lab/ascent) | Learned representations for tracking |
| [OrganoidTracker](https://github.com/jvzonlab/OrganoidTracker) | 3D lineage tracking |

---

# Core Research Problem

At a high level, the challenge can be represented as:

```text
Sparse Ground Truth
        │
        ▼
3D+Time Microscopy
        │
        ▼
┌──────────────────────┐
│   Cell Detection     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Feature / Motion     │
│ Representation       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Cell Association     │
│ Across Time          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Gap Closing / Track  │
│ Stitching            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Division Detection   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Lineage Graph        │
│ Reconstruction       │
└──────────────────────┘
```

The core challenge is therefore not simply detecting individual cells.

A successful solution must jointly reason about:

```text
Detection
    +
Motion
    +
Temporal Association
    +
Track Continuity
    +
Cell Division
    +
Lineage Structure
```

while learning from **sparse supervision**.

---

# Closing Message

The organizers encourage participants to enjoy the competition and explore new approaches to this difficult research problem.

They also strongly encourage competitors to **open-source their code and solutions** so that the wider scientific and machine-learning communities can benefit from the resulting work.

> We cannot wait to see your solutions.

— On behalf of all the organizers