# Student-t process likelihood products

`bench_student_t_likelihood.py` verifies the fixed-state Student-t process
likelihood coordinate `theta = log(nu - 2)`. Its NumPy oracle independently
constructs the dense RBF covariance, Cholesky factorization, Mahalanobis
statistic, and multivariate Student-t marginal density. Central differences of
that density check the FortML release probe's likelihood value, JVP, VJP, and
HVP. The FortML behavioral test separately checks the JVP/HVP finite
differences, JVP/VJP adjoint identity, transactional parameter-update refusal,
CPU dispatch, and typed CUDA refusal.

This is a fixed-state derivative contract: it holds the fitted covariance,
factorization, and Mahalanobis statistic constant. It does not benchmark or
claim likelihood-only optimization, cross-products through kernel/noise, or
derivatives through refitting. CUDA is recorded as unavailable when it returns
`FORTNUM_NOT_IMPLEMENTED`; there is no host fallback.

Run it from `fortml-bench` with the matching FortML checkout:

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_student_t_likelihood.py \
  --fortml ../fortml --output results/student_t_likelihood.csv
```
