# Matérn derivative-GP mixed HVP

`bench_derivative_gp_matern_hvp.py` independently assembles one-dimensional
Matérn 3/2 and 5/2 value/first-derivative covariance blocks, then
central-differences the dense likelihood gradient in NumPy. It compares each
checksum with the FortML CPU release app, whose production path uses analytic
radial parameter products and generated Matérn value/HVP kernels. Finite
differences are confined to the independent behavioral oracle.

The run records one CPU timing and checksum row for each kernel in
`results/derivative_gp_matern_hvp.csv`. CUDA is an explicit
`FORTNUM_NOT_IMPLEMENTED` row for each kernel because no resident
derivative-GP covariance/factorization graph is linked; no host fallback is
used.

```bash
FO_SCAN_FALLBACK=regex python3 -B scripts/bench_derivative_gp_matern_hvp.py \
  --fortml ../fortml \
  --output results/derivative_gp_matern_hvp.csv
```
