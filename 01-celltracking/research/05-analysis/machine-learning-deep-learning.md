---
id: 05-analysis/machine-learning-deep-learning
title: Machine Learning / Deep Learning
area: 05-analysis
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- ml
- training
---

# Machine Learning / Deep Learning

> Models, training, and evaluation. Mechanisms: [../02-theory/mechanisms.md](../02-theory/mechanisms.md).

## Models in play

- **Detector** — TemporalUNet3D / DeepCenter (public 50ep; DeepCenter center-prior epoch 500
  used as a veto). The retrain target for `bet-zebrahub-retrain`.
- **Edge predictor** — the associator whose logits feed harmonic fusion + `motion_relink`
  (fold weights `edge_predictor_best_split_{0,1}.pth`).
- **Candidate ranker (planned)** — the learned FP-suppressor of `bet-learned-ranker`; the LEAD
  mechanism is over-propose (recall ~0.98) + transformer re-scoring.

## Training / OOF

Fold OOF is produced by the `kaggle_train_oof` / `kaggle_predict_score` kernels; evaluated
leave-one-embryo-out ([../03-experimentation/experimental-design.md](../03-experimentation/experimental-design.md)).
Cross-family transfer is the known failure mode — always evaluate fit-on-one / eval-the-other.

## Evaluation discipline

Every model delta gates on ≥ +0.005 min-fold patched-scorer LOEO, both folds non-regressive.
Numbers are artifact-backed via [../06-knowledge-system/results.md](../06-knowledge-system/results.md).
