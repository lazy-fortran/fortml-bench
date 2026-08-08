# Weighted multiclass isotonic calibration

This lane checks the standalone `multiclass_probability_calibrator_t` weighted
one-vs-rest isotonic policy on 192 deterministic three-class logit rows with
sorted labels `[-4, 17, 91]`. The Fortran fit first computes a stable softmax,
fits one weighted pool-adjacent-violators map per class, linearly interpolates
the maps at prediction, and renormalizes every row to the probability simplex.

The Python lane is an independent NumPy oracle: it replays the stable softmax,
weighted PAVA block merges, interpolation, normalization, and calibrated
argmax labels without importing FortML. The release app exports every score,
weight, label, class, prediction, and probability before timing. The script
rejects any fixture mismatch, simplex error above `4e-14`, or probability error
above `4e-12` before accepting a timing row. It also checks the typed
`FORTNUM_NOT_IMPLEMENTED` status for isotonic score JVP/VJP products and for
the selected CUDA prediction boundary.

The checked-in CSV records separate NumPy and Fortran fit/predict timings,
correctness errors, knot counts, status-boundary rows, compiler and NumPy
versions, and clean FortML/benchmark revisions:
[`multiclass_isotonic_calibration.csv`](multiclass_isotonic_calibration.csv).

Reproduce the lane with:

```bash
python3 -B scripts/bench_multiclass_isotonic_calibration.py \
  --fortml ../fortml --output results/multiclass_isotonic_calibration.csv
```

Multiclass Platt scaling and resident CUDA isotonic kernels remain explicit
roadmap boundaries; the benchmark records no hidden host fallback.
