# Missing-indicator benchmark

This lane compares the dense `missing_indicator_t` transformer with an
independent NumPy NaN-mask oracle on a fixed 64-by-6 float64 fixture. The
fixture has missing entries in columns 2 and 5. Both `all` (six output
columns) and `missing-only` (two output columns) fit-time policies are checked.

For each policy the harness compares every transform entry, verifies the exact
zero input JVP and VJP products, records release-build CPU timings, and keeps a
typed CUDA-unavailable row until a resident indicator kernel exists. Raw data
are in [`missing_indicator.csv`](missing_indicator.csv); reproduce with:

```sh
python3 scripts/bench_missing_indicator.py --fortml ../fortml \
    --output results/missing_indicator.csv
```

This is a dense correctness/performance lane. Sparse CSR/CSC views and GPU
throughput are separate roadmap work and are not inferred from these timings.
