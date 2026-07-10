# Per-embryo target association adaptation

This is a bounded, label-free calibration layer for the private-shuffle hypothesis. It
does **not** train or modify the detector/linker. For exactly one embryo it unions organizer
and Trackastra candidate edges, finds consensus anchors using mutual top-1, score margin and
a physical gate, estimates robust embryo motion, measures `t -> t+1 -> t+2` path consistency,
and fits a small residual logistic calibrator. The adapted posterior is conservatively blended
with the unadapted ensemble. All fitted state dies at process exit; invoke it separately for
every embryo.

If consensus is sparse, covers too few timepoints, is motion-incoherent, has too few competing
negatives, or creates excessive score drift, the output is the unadapted ensemble and the JSON
diagnostic says why. `p_fp` is preserved/ensembled independently: it is never set to
`1 - p_tp`, which would be wrong under this competition's sparse edge annotation.

## Input and output

Each candidate CSV needs:

```text
source_id,target_id,p_tp,p_fp,source_t,target_t,source_z,source_y,source_x,target_z,target_y,target_x
```

`source_node_id,target_node_id,score` are accepted aliases for Trackastra. If those files do
not contain `p_fp`, pass an explicitly OOF-calibrated `--default-p-fp`; omission is a hard error.
Coordinates are raw ZYX voxels and default DaXi spacing is `1.625,0.40625,0.40625` um.
The output begins with `source_id,target_id,p_tp,p_fp`, so it can be passed directly to
`scripts.metric_solver.cli`; remaining columns are audit diagnostics.

## Reverse-fold simulation (the only go/no-go evidence)

First export organizer and Trackastra **candidate** CSVs with endpoint geometry for each held-out
fold. Never use held-out labels during adaptation. Then run independent resets:

```powershell
.\.venv\Scripts\python.exe -m scripts.target_adapt.cli `
  --embryo 44b6 `
  --input organizer=artifacts\target_adapt\44b6\organizer_candidates.csv `
  --input trackastra=artifacts\target_adapt\44b6\trackastra_candidates.csv `
  --default-p-fp <OOF_TRAINED_FP_PRIOR> `
  --output artifacts\target_adapt\44b6\adapted_candidates.csv `
  --diagnostics artifacts\target_adapt\44b6\diagnostics.json

.\.venv\Scripts\python.exe -m scripts.metric_solver.cli `
  --candidates artifacts\target_adapt\44b6\adapted_candidates.csv `
  --nodes artifacts\oof_44b6\<crop>.geff --lambda <OOF_LAMBDA> `
  --backend auto --output artifacts\target_adapt\44b6\selected.csv

.\.venv\Scripts\python.exe -m scripts.target_adapt.cli `
  --embryo 6bba `
  --input organizer=artifacts\target_adapt\6bba\organizer_candidates.csv `
  --input trackastra=artifacts\target_adapt\6bba\trackastra_candidates.csv `
  --default-p-fp <OOF_TRAINED_FP_PRIOR> `
  --output artifacts\target_adapt\6bba\adapted_candidates.csv `
  --diagnostics artifacts\target_adapt\6bba\diagnostics.json

.\.venv\Scripts\python.exe -m scripts.metric_solver.cli `
  --candidates artifacts\target_adapt\6bba\adapted_candidates.csv `
  --nodes artifacts\oof_6bba\<crop>.geff --lambda <OOF_LAMBDA> `
  --backend auto --output artifacts\target_adapt\6bba\selected.csv
```

The current solver operates per node graph/crop, so production should adapt on the union of the
entire embryo first, then split the adapted candidates back by crop for solving. Compare adapted
versus unadapted exact full-fold edge-J and adjusted-J. The predeclared gate from the winning-path
review is improvement of at least `+0.010` edge-J on **both** reverse folds with no count/division
regression. Freeze every threshold before touching test inference.

## Rules status (2026-07-10 snapshot)

`reports/RULES_TEST_TIME_ADAPTATION_2026-07-10.md` records an authenticated rules retrieval.
On its face, fully automated self-supervised adaptation on provided unlabeled Competition Data is
**permitted**: Competition Data may be used algorithmically, public pretrained models are allowed,
and the ban is on hand labeling/human prediction of validation or test records. Keep the snapshot
and diagnostics in the provenance bundle. Manual correction, public-trajectory label transfer,
submission-score label reconstruction, the unmatched-fork evaluator exploit, and notebook internet
remain prohibited/quarantined. Organizer confirmation would reduce residual adjudication risk.
