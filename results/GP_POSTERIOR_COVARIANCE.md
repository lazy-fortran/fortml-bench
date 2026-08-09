# Exact GP posterior covariance

This lane compares `gp_regression_t%predict_covariance` with an independent NumPy dense solve for a shared RBF query set. The full latent posterior matrix is checked for checksum, symmetry, and a diagonal match with the marginal variance path. CPU dispatch is the reference, and selected CUDA returns `FORTNUM_NOT_IMPLEMENTED` without claiming a host fallback. The same lane checks the full-matrix hyperparameter JVP and VJP against independent NumPy solve-state products. Covariance derivative device calls have the same explicit CUDA refusal.

Oracle covariance checksum: `0.7519868471770201`, variance checksum: `0.2038218026529446`.

Reproduce with:

```text
python scripts/bench_gp_posterior_covariance.py --fortml ../fortml --output results/gp_posterior_covariance.csv --report results/GP_POSTERIOR_COVARIANCE.md
```

Pinned source revision: `ee39cc0537bd10e5d23282f1b194843938749488`.
Pinned benchmark revision: `1056baa498bd9d94f2f8ec98dd98df9cad17fb7d`.
