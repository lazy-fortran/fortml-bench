# Local-periodic derivative-observation GP products

`bench_derivative_gp_local_periodic.py` checks FortML's local-periodic
mixed value/first-derivative Gaussian process against a separate NumPy dense
Cholesky oracle. The oracle assembles the scalar value, input gradient, and
mixed Hessian directly, then central-differences the complete likelihood in
the packed logarithmic kernel/noise coordinates. The fixture covers value,
first-derivative, and second mixed input components in both training and
query observations.

The FortML gate passes the independent posterior, hyperparameter-gradient, and
query-input JVP checks after the analytic local-periodic third-input rule is
enabled. The query fixture includes value, first-feature, and second-feature
components and a query coincident with a training row, so the removable radial
limits are exercised. Resident CUDA remains an explicit typed refusal because
the derivative-GP covariance/factorization graph is not linked.

```bash
python -B scripts/bench_derivative_gp_local_periodic.py \
  --fortml ../fortml --output results/derivative_gp_local_periodic.csv
```

The recorded fixture has five training observations, three mixed-component
queries, a minimum posterior variance of `1.218188755371994`, and a
FortML behavioral-gate runtime of approximately six seconds on the recorded
CPU. The CSV stores exact source and benchmark revisions, environment
metadata, independent-oracle rows, query-JVP norms, and the typed CUDA
refusal row.
