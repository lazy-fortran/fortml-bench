# Grouped MLP regularization benchmark

This lane checks the named parameter-group objective against an independent
NumPy linear-ridge oracle. The packed vector is
`[weight,bias,log(lambda_weight),log(lambda_bias)]`; value, gradient norm, JVP,
and mixed HVP norm are checked before timings are recorded. It also exercises
the bounded grouped L-BFGS-B adapter. Its optimizer objective, raw gradient
norm, and active log-L2 coordinates are checked against an independent
closed-form ridge solution (`log(lambda)=-3`); the iteration count is recorded
as a solver diagnostic. This remains a correctness lane, not an end-to-end
neural-training performance claim.

Run it with:

```bash
python3 scripts/bench_mlp_grouped_training.py \
  --fortml ../fortml --output results/mlp_grouped_training.csv
```

The CPU rows use the exact analytic objective and independent NumPy oracles.
The CUDA row is intentionally `unavailable`: grouped MLP derivatives require a
resident network graph, so the current API returns `FORTNUM_NOT_IMPLEMENTED`
without copying through the host.  The CSV records source and benchmark
revisions, compiler flags, and the oracle tolerances.
