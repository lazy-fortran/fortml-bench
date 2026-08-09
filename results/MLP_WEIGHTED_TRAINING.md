# Weighted MLP training benchmark

Recorded 2026-08-09 from the weighted-training release app and an
independent NumPy affine oracle.

| Field | Value |
| --- | --- |
| FortML revision | `143590856846190bf55fa8dd9b15d801828338d0` |
| Benchmark revision | `cc3fbec0138207ced3231dacf79719b8cc52aee9` |
| Compiler | `gfortran` with `-O3` release app protocol |
| Precision | CPU `float64` |
| Fixture | Four affine rows, weights `[1, 0, 2, 0.5]`, `lambda=0.07` |
| Acceptance | Absolute oracle error `<= 2e-12` |

The NumPy oracle evaluates

\[
 L = \frac{\sum_i w_i\|f(x_i)-y_i\|^2}{2\sum_i w_i}
     + \frac{\lambda}{2}\|\theta\|^2
\]

and the weighted SGD update with batch size two and two-step accumulation.
The release app reports zero loss error, `1.11e-16` maximum gradient error, and
zero parameter error for the independent recurrence. Its weighted validation
loss is `0.2634173702851677`. A malformed negative-weight call returns the
domain-error status and leaves parameters unchanged.

The CSV includes NumPy oracle rows, release-app correctness rows, and the
explicit CUDA refusal. The CUDA row is a capability record. CPU work is never
reported as resident GPU training.

Run the lane with:

```bash
python -B scripts/bench_mlp_weighted_training.py \
  --fortml ../fortml \
  --output results/mlp_weighted_training.csv
```
