# Radius-neighbor multi-output regression benchmark

`bench_radius_neighbors_multioutput.py` reconstructs the 320-row, three-feature
fixture used by the release app and independently computes the seven-query
uniform multi-output radius reduction in NumPy. The app emits both target
columns; every value must match before FortML CPU timings are retained. The
gate tolerance is `3e-12`. Queries without a selected row use the fitted vector
outlier value.

Radius membership is discontinuous. FortML therefore exposes typed JVP/VJP
boundary products and reports CUDA as `unavailable` until a resident
radius-search reduction kernel is linked; host timings are never relabeled as
accelerator execution.

```bash
.venv/bin/python -B scripts/bench_radius_neighbors_multioutput.py \
    --fortml ../fortml --output results/radius_neighbors_multioutput.csv
```

The CSV contains the independent NumPy oracle, CPU fit and prediction timings,
the maximum absolute error, source and benchmark revisions, and the typed CUDA
capability row.
