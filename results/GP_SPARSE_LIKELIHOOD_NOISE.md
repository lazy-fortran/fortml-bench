# Sparse-GP Gaussian likelihood noise products

This lane checks the one-coordinate Gaussian likelihood block of
`sparse_gp_t`. The packed coordinate is `log(noise_variance)`. The independent
NumPy oracle assembles the RBF inducing covariance, Cholesky solve, variational
marginal variances, Gaussian expected likelihood, and analytic KL term. It
checks the ELBO gradient and scalar Hessian-vector product against central
differences. The FortML gate checks JVP, VJP, HVP, transactional malformed and
overflowing updates, and the CPU/CUDA boundary.

Run it from this checkout with:

```bash
python3 -B scripts/bench_gp_sparse_likelihood_noise.py \
  --fortml ../fortml \
  --output results/gp_sparse_likelihood_noise.csv
```

The recorded source revision is `a2c55bf`, and the benchmark source is pinned
after the benchmark documentation commit. The NumPy oracle reports a
log-noise gradient of `1.720479112421458e+01` and an HVP of
`-2.070479121485391e+01`. The FortML behavioral gate passed in 23.81 seconds
on the clean checkout. The CUDA row is `unavailable` because
the inducing solve and ELBO reduction are not resident. The device entry point
returns `FORTNUM_NOT_IMPLEMENTED` and performs no host fallback.
