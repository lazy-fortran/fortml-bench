# Batched multi-output GP products

This lane checks the shape contract for independent batches of query points in
FortML's intrinsic-coregionalization exact GP. The public arrays are
`query(batch, query, feature)` and `mean(batch, query, output)`. The independent
NumPy oracle assembles the output-major dense covariance
`(W W^T + diag(independent)) ⊗ K`, solves the fitted posterior, and evaluates
each batch member directly. It also differentiates the RBF query mean and
checks the input JVP against a central difference and the input VJP by scalar
duality.

The Fortran contract gate is `test_multi_output_gp_batch`. It checks the
dense mean oracle, JVP finite difference, VJP adjoint, CPU dispatch, malformed
shapes, nonfinite directions, and typed CUDA refusals. The CUDA row is
`unavailable`: coregionalized batch covariance, factorization, and derivative
state are not resident, and no hidden host fallback is timed.

Reproduce the three-row CSV with:

```bash
python -B scripts/bench_multi_output_gp_batch.py \
  --fortml ../fortml --output results/multi_output_gp_batch.csv
```

The CSV records the FortML source and benchmark revisions. The independent
oracle row is valid before the Fortran gate is run; a missing compiler or
backend should be represented as a parseable refusal row rather than omitted.
