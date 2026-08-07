# Weighted Poisson/Gamma GLM regression

This release lane compares `fortml_glm_regression` with an independent NumPy
Newton solve of the bounded (`[-30,30]`) weighted L2-regularized log-link
objectives. The fixture has 256 rows and three features, uses `alpha=0.05`, and
uses the same deterministic targets and weights in both implementations. The
raw CSV records fit time, objective value, prediction mean, and the maximum
objective/prediction discrepancy against the NumPy oracle.

The CUDA row is intentionally `unavailable`: no resident GLM kernel is linked,
so a selected CUDA context returns `FORTNUM_NOT_IMPLEMENTED`. CPU timings are
not presented as GPU measurements.

Reproduce:

```bash
python3 -B scripts/bench_glm_regression.py \
  --fortml ../fortml --output results/glm_regression.csv
```

Raw data: [`glm_regression.csv`](glm_regression.csv).
