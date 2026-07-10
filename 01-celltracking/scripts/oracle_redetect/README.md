# OOF oracle and redetection diagnostics

The checkout currently contains raw images and GT for both embryos, but no learned
OOF prediction GEFFs or pre-solver candidate-edge dumps. Download both held-out-fold
prediction outputs before running the oracle measurement.

```powershell
.venv\Scripts\python.exe scripts\oracle_redetect\measure_oracles.py `
  --pred-dir artifacts\oof_44b6 `
  --pred-dir artifacts\oof_6bba `
  --gt-dir data\train `
  --out-csv artifacts\oracle_rows.csv `
  --out-json artifacts\oracle_summary.json
```

With no `--candidate-dir`, `candidate_oracle` is the oracle over the edges already
selected in each predicted GEFF. To measure the real pre-solver candidate ceiling,
export `<crop>.csv` files with `source_id,target_id` and pass their directory.

The three relevant levels are:

- `baseline`: the submitted graph under exact matching, FP-region, and count semantics.
- `candidate_oracle`: retain only candidate edges that become true positives after the
  metric's actual one-to-one assignment.
- `endpoint_oracle`: connect every GT edge whose two prediction endpoints are actually
  matched. This isolates the fixed-detection/assignment ceiling.
- `existential_endpoint_recall`: a deliberately loose diagnostic using any in-gate
  detection at each endpoint; competition between detections can make it unattainable.

The edge score excludes the separate `+ 0.1 * division_jaccard` term. It answers how
much edge-J is available; it is not a full-score oracle.

Run one redetection crop only after predictions exist:

```powershell
.venv\Scripts\python.exe scripts\oracle_redetect\redetect_diagnostic.py `
  --pred-dir artifacts\oof_6bba --fold 6bba --max-crops 1
```

This reports conditional rescue of known no-candidate endpoints. GT chooses the audit
cases, so the result cannot justify deployment until a blind dangling-track query audit
measures added detections, FP-eligible edges, count ratio, and exact adjusted edge-J.

The blind experiment supplies that audit. Its query generation and confidence/count
selection do not touch GT. It searches one frame beyond predicted track starts/ends,
globally deduplicates peaks, and defaults to a one-crop dry run:

```powershell
.venv\Scripts\python.exe scripts\oracle_redetect\blind_redetect_experiment.py `
  --pred-dir artifacts\oof_44b6 --fold 44b6 --max-crops 1
```

After checking query/proposal volume, use the same configuration with `--apply`; GT is
then loaded only to calculate exact before/after TP, FP, FN, count multiplier, and adjJ:

```powershell
.venv\Scripts\python.exe scripts\oracle_redetect\blind_redetect_experiment.py `
  --pred-dir artifacts\oof_44b6 --fold 44b6 --max-crops 10 --apply `
  --out-csv artifacts\blind_redetect_44b6_first10.csv
```

The default budget is at most 500 nodes and at most 2% of the existing prediction per
crop (the tighter limit wins). `--max-queries 1000` is also a compute guard. Confidence
is normalized within each search patch, so tune its threshold strictly by held-out
embryo and prefer a frozen cross-fold setting over crop-specific optimization.
