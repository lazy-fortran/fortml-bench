# Local-periodic derivative-observation GP products

`bench_derivative_gp_local_periodic.py` checks FortML's local-periodic
mixed value/first-derivative Gaussian process against a separate NumPy dense
Cholesky oracle. The oracle assembles the scalar value, input gradient, and
mixed Hessian directly, then central-differences the complete likelihood in
the packed logarithmic kernel/noise coordinates. The fixture covers value,
first-derivative, and second mixed input components in both training and
query observations.

The FortML gate passes the independent posterior and hyperparameter-gradient
checks after the analytic local-periodic parameter JVP is enabled. Query-input
JVP remains a deliberate `FORTNUM_NOT_IMPLEMENTED` capability boundary until
the corresponding third-input derivative is generated. Resident CUDA remains
an explicit typed refusal because the derivative-GP covariance/factorization
graph is not linked.

```bash
python -B scripts/bench_derivative_gp_local_periodic.py \
  --fortml ../fortml --output results/derivative_gp_local_periodic.csv
```

The recorded fixture has five training observations, three mixed-component
queries, a minimum posterior variance of `1.218188755371994`, and a
FortML behavioral-gate runtime of approximately five seconds on the recorded
CPU. The CSV stores exact source and benchmark revisions, environment
metadata, independent-oracle rows, and the typed CUDA/query-product refusal
rows.
