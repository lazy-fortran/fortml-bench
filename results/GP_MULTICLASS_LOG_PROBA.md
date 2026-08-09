# Multiclass Laplace-GP log-probability gate

This lane closes the sklearn-style `predict_log_proba` surface for the
sorted-label one-vs-rest Laplace-GP classifier.  The independent NumPy oracle
checks the OVR normalization and logarithm, including the input JVP central
difference.  The FortML gate fits a three-class model and checks value/log
round trips, input JVP/VJP duality, packed kernel-parameter JVP/VJP duality,
shape refusal, CPU dispatch, and the explicit CUDA refusal.

Reproduce from the benchmark checkout:

```bash
python3 -B scripts/bench_gp_multiclass_log_proba.py \
  --fortml ../fortml \
  --output results/gp_multiclass_log_proba.csv
```

The NumPy normalization oracle reports a maximum central-difference error
below `2e-8` and an exact round trip to floating-point precision.  The CSV
records the public-contract gate and keeps the CUDA row `unavailable`: OVR
Laplace covariance states and the normalization reduction are not resident,
so no host fallback is counted as GPU support.
