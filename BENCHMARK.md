# Benchmark lanes

## Basis-composed linear regression

The basis-linear lane covers polynomial, cubic B-spline, and Fourier feature
maps with a fitted multi-output linear model.  It checks prediction, JVP, and
VJP products against independent NumPy finite-difference and adjoint oracles,
then records FortML release timings and the explicit CUDA capability boundary.

```bash
python -B scripts/bench_basis_linear_regression.py \
  --fortml ../fortml --output results/basis_linear_regression.csv
```

See [`results/BASIS_LINEAR_REGRESSION.md`](results/BASIS_LINEAR_REGRESSION.md)
for the fixture and reproducibility details.
