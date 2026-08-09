# Cross-validation scoring

This release lane checks the new `fortml_cross_validation` control plane against
an independent NumPy oracle. The fixture has 17 rows and four contiguous K-fold
test folds. Each fold's score is `theta + mean(test_index)`, its derivative is
one, and its weight is the held-out row count. The benchmark checks every fold
value and weight, the weighted mean, the oriented FortOpt minimization value,
and the parameter gradient before retaining timing. It also records the
NumPy oracle and a typed CUDA-unavailable row; no host timing is labeled GPU.

Run it from this repository with:

```bash
python -B scripts/bench_cross_validation.py \
  --fortml ../fortml --output results/cross_validation.csv
```

The CSV records source and benchmark revisions, compiler flags, correctness
error, and per-operation timings. The reference path is intentionally separate
from FortML's Fortran callback and aggregation implementation.
