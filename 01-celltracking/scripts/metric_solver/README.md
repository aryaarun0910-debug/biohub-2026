# Metric-aligned edge selection prototype

This prototype replaces a generic MAP edge cost with the expected-Jaccard
parametric weight `p_tp - lambda*p_fp`, while enforcing at most one parent and
at most two children. It does not alter detections.

`p_fp` means **probability of a metric-counted false-positive edge**. It is not
silently set to `1-p_tp`: with sparse annotations, an edge can be ignored by the
metric. Fit/calibrate `p_tp` and `p_fp` separately on embryo-held-out predictions.

Fixed-lambda hidden/test solve (lambda selected strictly from OOF):

```powershell
.\.venv\Scripts\python.exe -m scripts.metric_solver.cli `
  --candidates candidates.csv --nodes detections.geff `
  --lambda 0.72 --backend scip --output reweighted.csv
```

OOF-only Dinkelbach solve, where the true edge count is known:

```powershell
.\.venv\Scripts\python.exe -m scripts.metric_solver.cli `
  --candidates oof_candidates.csv --nodes oof_nodes.geff `
  --ground-truth-edges 12000 --backend auto --output oof_selected.csv
```

The output retains every candidate and adds calibrated probabilities,
`metric_weight`, `lambda`, and `selected`. `auto` tries direct PySCIPOpt, then
SciPy MILP, then deterministic greedy. `tracksdata` explicitly uses the repo's
division-native `ILPSolver` (ilpy/SCIP); direct SCIP/SciPy implement precisely
the minimal parent/child-capacity model and are the recommended prototype gate.

Calibration flags are independent for TP and FP (`--tp-temperature`,
`--tp-bias`, `--fp-temperature`, `--fp-bias`). The missing-FP policies exist
only for explicit ablations; the default is a hard error.
