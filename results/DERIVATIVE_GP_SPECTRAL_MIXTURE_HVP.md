# Spectral-mixture derivative-GP mixed HVP

`bench_derivative_gp_spectral_mixture_hvp.py` independently assembles the
two-dimensional value/first-derivative covariance blocks, central-differences
the dense likelihood gradient in NumPy, and compares the checksum with the
FortML CPU release app. The production path uses analytic four-jets through
each separable spectral-mixture factor; finite differences are confined to the
independent oracle.

The run records the CPU timing and checksum in
`results/derivative_gp_spectral_mixture_hvp.csv`. CUDA is an explicit
`FORTNUM_NOT_IMPLEMENTED` row because no resident derivative-GP
covariance/factorization graph is linked; no host fallback is used.

```bash
FO_SCAN_FALLBACK=regex python3 -B scripts/bench_derivative_gp_spectral_mixture_hvp.py \
  --fortml ../fortml \
  --output results/derivative_gp_spectral_mixture_hvp.csv
```
