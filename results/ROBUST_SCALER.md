# Robust-scaler benchmark

This lane checks dense `robust_scaler_t` on the release application's fixed
96-by-5 float64 fixture.  The independent NumPy oracle uses the same
linear-interpolation percentiles as the estimator: the median is the center,
the 25--75 percentile difference is the scale, and a constant feature uses a
unit scale.  It checks the complete transform, inverse transform, and diagonal
input JVP arrays before retaining the NumPy timings.

The FortML release app emits transform and JVP checksums.  Both checksums must
match the NumPy oracle to `5e-11` before the CPU transform timing is recorded;
the app currently does not emit a separate inverse or JVP timing.  A typed
CUDA-unavailable row is retained because no resident robust-scaler kernel is
linked, and no host timing is relabeled as device work.

Reproduce the lane with:

```sh
python3 -B scripts/bench_robust_scaler.py --fortml ../fortml \
    --output results/robust_scaler.csv
```

Raw rows, compiler flags, source revisions, and oracle errors are in
[`robust_scaler.csv`](robust_scaler.csv).  The inverse and JVP correctness
checks are independent of the FortML release application's timing loop.
