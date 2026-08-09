# Wave 11 derivative products

The release CSV contains nine rows for two bounded production slices:

- a direct NumPy Poisson log-rate value, gradient norm, and Hessian-vector
  checksum, followed by the FortML `test_poisson_likelihood` gate;
- an independent NumPy affine full-batch SGD recurrence and directional
  derivative, followed by the FortML affine optimizer-group HVP gate.

Both gates include a CUDA `unavailable` row with the typed resident-executor
boundary.  No CPU fallback is timed as GPU work.  Reproduce the rows with:

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
python -B scripts/bench_wave11_derivative_products.py \
  --fortml ../fortml --output results/wave11_derivative_products.csv
```

The CSV records clean FortML and benchmark revisions.  The NumPy fixtures are
implemented independently of FortML; the Fortran tests provide the separate
behavioural oracle for products, optimizer integration, and capability
refusals.
