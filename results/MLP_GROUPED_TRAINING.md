# Grouped MLP regularization benchmark

This lane checks the named parameter-group objective against an independent
NumPy linear-ridge oracle.  The packed vector is
`[weight,bias,log(lambda_weight),log(lambda_bias)]`; value, gradient norm, JVP,
and mixed HVP norm are checked before timings are recorded.  It is a
correctness lane, not an end-to-end neural-training performance claim.

Run it with:

```bash
python3 scripts/bench_mlp_grouped_training.py \
  --fortml ../fortml --output results/mlp_grouped_training.csv
```

The CPU row uses the exact analytic objective and independent NumPy oracle.
The CUDA row is intentionally `unavailable`: grouped MLP derivatives require a
resident network graph, so the current API returns `FORTNUM_NOT_IMPLEMENTED`
without copying through the host.  The CSV records source and benchmark
revisions, compiler flags, and the oracle tolerances.
