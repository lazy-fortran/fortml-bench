# Sparse CSC preprocessing

This correctness-gated lane measures the sparse-safe `StandardScaler` branch
(`sparse_standard_scaler_t`) on a four-row, three-feature real CSC fixture with
five stored values. The independent NumPy oracle expands implicit zeros,
computes population means/scales, and checks transformed, inverse, JVP, and VJP
stored values before accepting timings.

The release app performs 20,000 repetitions per phase with one host thread.
The raw record is [`sparse_preprocessing.csv`](sparse_preprocessing.csv), and
the source and benchmark revisions are recorded in every row. The Fortran
transform, inverse, JVP, and VJP rows all have maximum absolute error zero
against the independent oracle. A CUDA row is retained as `unavailable` with a
typed refusal because no resident sparse preprocessing kernel is linked; no
host timing is relabeled as device evidence.

Reproduce from this checkout with:

```sh
python3 scripts/bench_sparse_preprocessing.py \
  --fortml ../fortml --output results/sparse_preprocessing.csv
```

Use `--no-build` only after rebuilding the FortML release app with the selected
compiler and optimization flags.
