# Deterministic random forest

`bench_random_forest.py` exercises the FortML bootstrap CART ensemble on 240
three-feature samples and six separated queries. The independent oracle uses a
direct NumPy piecewise class rule (`x[:,0] < -0.65`, middle class, or
`x[:,0] > 0.65`) and does not require scikit-learn. The app must report all
six query labels, a zero probability-simplex error, finite timings, and the
typed CUDA refusal before rows are written.

Run:

```bash
python -B scripts/bench_random_forest.py \
  --fortml ../fortml --output results/random_forest.csv
```

The CSV records the NumPy oracle, FortML CPU fit/predict timings, and an
explicit CUDA `unavailable` row. The workload uses 32 trees, depth 6, and
seed 1729. Tree ensembles currently have no resident CUDA kernel, so the
CUDA row is capability evidence rather than a performance result; no hidden
host fallback is counted.
