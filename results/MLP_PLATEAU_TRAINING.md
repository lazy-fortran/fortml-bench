# Metric-aware plateau trainer

This companion lane checks typed `MLP_SCHEDULE_PLATEAU` integration in
`mlp_train`. The independent NumPy recurrence uses four one-update epochs,
patience two, `min_delta=0.02`, and factor `0.4`; it predicts a final learning
rate of `1.6e-13`. The Fortran release app matches that value and emits four
optimizer updates. The source fixture additionally checks minimizing and
maximizing transitions, active-branch products, malformed fields, formatted
checkpoint round-trip, and interrupted versus uninterrupted trajectories.

The CUDA row is intentionally `unavailable`: metric reduction and optimizer
state are not resident, and the trainer never hides a host fallback behind a
CUDA request.

Run:

```bash
python -B scripts/bench_mlp_plateau_training.py \
  --fortml ../fortml \
  --output results/mlp_plateau_training.csv
```

The CSV pins the exact FortML and benchmark revisions used to generate the
rows.
