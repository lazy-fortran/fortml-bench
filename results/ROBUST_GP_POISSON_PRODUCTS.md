# Robust-GP Poisson products benchmark

`bench_robust_gp_poisson_products.py` checks the normalized Poisson
log-rate likelihood and fixed-state Laplace posterior HVPs against independent
NumPy formulas. It also runs the focused Fortran product oracle, times the
release app, and records the explicit CUDA refusal.

Run it from this repository with:

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_robust_gp_poisson_products.py \
  --fortml ../fortml --output results/robust_gp_poisson_products.csv
```

The release checksum rows are retained only when they agree with the NumPy
oracle at `2e-10` absolute error. Revision columns are pinned after the source
and benchmark commits; generated builds are not part of the benchmark state.
