# Exact GP posterior covariance

This lane compares `gp_regression_t%predict_covariance` with an independent NumPy dense solve for a shared RBF query set. The full latent posterior matrix is checked for checksum, symmetry, and a diagonal match with the marginal variance path. CPU dispatch is the reference, and selected CUDA returns `FORTNUM_NOT_IMPLEMENTED` without claiming a host fallback.

Oracle covariance checksum: `0.7519868471770201`, variance checksum: `0.2038218026529446`.

Reproduce with:

```text
python scripts/bench_gp_posterior_covariance.py --fortml ../fortml --output results/gp_posterior_covariance.csv --report results/GP_POSTERIOR_COVARIANCE.md
```

Pinned source revision: `6a47c36b66febb544ca3e6b62f8a1110355d274d`.
Pinned benchmark revision: `8c1d8c26f8baff0e042f534be8328acda9900e81`.
