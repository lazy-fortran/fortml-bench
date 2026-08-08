# Metric-aware plateau trainer

This lane checks typed `MLP_SCHEDULE_PLATEAU` integration in `mlp_train`.
The independent NumPy recurrence uses four one-update epochs, patience two,
`min_delta=0.02`, and factor `0.4`; it predicts a final learning rate of
`1.6e-13`.  The Fortran release app matches that value to machine precision and
also emits four optimizer updates.  The source fixture independently checks
minimize/maximize transitions, active-branch base/factor products, malformed
fields, formatted checkpoint round-trip, and interrupted versus uninterrupted
trainer trajectories.

The CUDA row is intentionally `unavailable`: metric reduction and optimizer
state are not resident, and the trainer never hides a host fallback behind a
CUDA request.

Run:

```bash
python -B scripts/bench_mlp_plateau_schedule.py \
  --fortml ../fortml \
  --output results/mlp_plateau_schedule.csv
```

The CSV pins the exact FortML and benchmark commits used to generate the rows.
