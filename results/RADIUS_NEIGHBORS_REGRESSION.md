# Radius-neighbor scalar regression benchmark

`bench_radius_neighbors_regression.py` reconstructs the six-query, 96-row
fixture and inverse-distance weighted reduction independently in NumPy. The
FortML release app must emit every scalar prediction before its CPU fit/predict
timings are retained. The gate tolerance is `3e-12`; empty neighborhoods use
the explicit fitted outlier value.

Radius selection is discontinuous. FortML therefore exposes typed JVP/VJP
refusals and reports CUDA as `unavailable` until a resident radius-search
reduction kernel is linked; host timings are never relabeled as accelerator
execution.

```bash
.venv/bin/python -B scripts/bench_radius_neighbors_regression.py \
    --fortml ../fortml --output results/radius_neighbors_regression.csv
```

The CSV records the independent oracle row, CPU fit and prediction timings,
the maximum absolute prediction error, source and benchmark revisions, and
the typed CUDA capability row.
