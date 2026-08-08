# Locally-periodic exact-GP benchmark

`bench_local_periodic_gp.py` compares a vectorized NumPy covariance and dense
Cholesky posterior against a separate scalar-loop covariance oracle, then runs
FortML's independent `test_local_periodic_gp` gate. The fixture covers the
four logarithmic kernel parameters, exact posterior mean/variance, input
gradient/mixed-Hessian limits, parameter JVP/VJP/HVP products, and the typed
static-operator/CUDA refusal.

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_local_periodic_gp.py \
  --fortml ../fortml --output results/local_periodic_gp.csv
```

The committed CSV records the source and benchmark revisions, Python/NumPy,
compiler flags, and six CPU/refusal rows. The scalar-loop covariance agrees to
machine precision. The posterior variance is positive (`1.227275432322503e+00`
minimum on this fixture). The FortML public-contract gate passes. No GPU
timing is reported: `kernel_operator_t` and resident CUDA return
the typed `FORTNUM_DOMAIN_ERROR` refusal until their static ABI carries the
four-parameter locally-periodic leaf.
