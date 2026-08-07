# Radius-neighbor classification benchmark

`bench_radius_neighbors.py` reconstructs the six-query, 96-row fixture and
inverse-distance vote reduction independently in NumPy. The release app must
emit every class, probability, and prediction before its CPU fit/predict
timings are retained. The gate tolerance is `3e-12`; empty neighborhoods use
the fitted in-training outlier class.

Radius neighbor selection is discontinuous. FortML therefore exposes typed
JVP/VJP refusals and reports CUDA as `unavailable` until a resident radius
search kernel is linked; no host timing is relabeled as GPU execution.

```bash
.venv/bin/python -B scripts/bench_radius_neighbors.py \
    --fortml ../fortml --output results/radius_neighbors.csv
```
