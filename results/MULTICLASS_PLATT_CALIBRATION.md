# Multiclass Platt calibration

This release lane checks the weighted one-vs-rest Platt policy of
`multiclass_probability_calibrator_t`.  FortML first computes a stable raw
softmax for each logit row, fits one weighted sigmoid to each class indicator,
and normalizes the positive sigmoid values back to a simplex.  Classes are
sorted arbitrary integer labels and the packed map parameters are interleaved
`[slope_1, intercept_1, ..., slope_C, intercept_C]`.

## Correctness gate

The independent NumPy oracle uses the same weighted fixture, stable softmax,
logistic objective, damped Newton solve, and `L2=0.05`, but has no FortML code
in its operation graph.  The release app emits its complete fixture, sorted
classes, fitted parameters, predictions, and probabilities before timing.

| Backend | Fit (s) | Predict (s) | Accuracy | Max probability error | Max parameter error | Simplex error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NumPy oracle | 1.0442e-3 | 3.0485e-5 | 0.953125 | 0 | 0 | 2.22e-16 |
| FortML CPU | 9.6125e-5 | 6.4922e-6 | 0.953125 | 2.58e-12 | 1.73e-10 | 2.22e-16 |

The fixture contains 192 rows, three classes (`-4`, `17`, `91`), and positive
deterministic sample weights.  All four smooth products (input JVP/VJP and
packed-parameter JVP/VJP) return `FORTNUM_OK`.  A selected CUDA context returns
`FORTNUM_NOT_IMPLEMENTED` with no host fallback; this is recorded as an
explicit device-capability row in the CSV.

## Provenance

- FortML source: `e1359ce1ab3d0b491dca1568f0fe545f84c6a560`
- Benchmark generator: `cb6261754a0360ffbef15fc8f2c1ead8301058f6`
- Compiler: `gfortran`; flags: `-O3`; threads: `OMP_NUM_THREADS=1`
- Python `3.14.6`; NumPy `2.5.1`
- CSV: [`multiclass_platt_calibration.csv`](multiclass_platt_calibration.csv)

The CSV is regenerated only after the source and generator revisions are clean;
the final CSV commit supplies the benchmark-repository revision recorded in
its rows.
