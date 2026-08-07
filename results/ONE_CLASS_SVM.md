# Dense RBF one-class SVM benchmark

`bench_one_class_svm.py` records a bounded correctness and timing lane for a
dense RBF one-class SVM.  The fixture has 48 two-dimensional training points
and four query points, with `nu=0.5` and `gamma=0.8`.

The NumPy reference implements the dual capped-simplex problem independently:
projected-gradient updates enforce `0 <= alpha_i <= 1/(nu n)` and
`sum(alpha)=1`, then the KKT support interval supplies the offset.  The lane
checks the dual constraints, signed query scores, and `+1`/`-1` anomaly labels
before retaining CPU NumPy fit and prediction timings.

The release app now emits complete support weights, the KKT offset, query
scores, and anomaly labels.  The benchmark retains FortML CPU fit and predict
timings only after every emitted value matches the independent oracle.  The
CUDA row remains an explicit typed refusal (`FORTNUM_NOT_IMPLEMENTED`); no host
timing is relabeled as device evidence.  Fit active-set/hyperparameter
derivatives and a resident CUDA kernel are separate implementation work.

Run the lane with:

```bash
.venv/bin/python -B scripts/bench_one_class_svm.py \
  --fortml ../fortml --output results/one_class_svm.csv
```

The raw machine-readable record is [`one_class_svm.csv`](one_class_svm.csv).
Each row records the FortML source revision and benchmark revision used to
generate it; `+dirty` provenance is a failure requiring regeneration from
clean checkouts.
