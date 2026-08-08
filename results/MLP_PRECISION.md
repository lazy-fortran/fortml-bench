# MLP training precision capability benchmark

This lane gates the explicit `mlp_training_options_t%precision_kind` contract
with an independent NumPy one-layer linear-MSE SGD recurrence. FP64 must match
the recurrence; recognized FP32, FP16, and BF16 modes must return a typed
`FORTNUM_NOT_IMPLEMENTED` before mutating parameters. CUDA remains an explicit
unavailable row because the trainer has no resident mixed-precision state.

Run:

```bash
python -B scripts/bench_mlp_precision.py \
  --fortml ../fortml --output results/mlp_precision.csv
```

The checked-in CSV contains one independent FP64 row, one FortML FP64
correctness/timing row, three typed lower-precision refusal rows, and one CUDA
residency refusal row. The source and benchmark revisions are recorded in each
row; no unsupported precision is timed as if it were implemented.

The recorded FP64 release-test gate took `2.901566014974378` seconds on the
host (including the `fo` test build). The independent recurrence matched to
zero reported error; FP32/FP16/BF16 remain explicit unavailable rows.
