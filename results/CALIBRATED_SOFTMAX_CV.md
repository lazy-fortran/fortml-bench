# Leakage-safe multiclass calibrated softmax cross-validation

This lane exercises `calibrated_softmax_classifier_t` on 96 rows, two
features, three deterministic stratified folds, and sorted integer labels
`[-4, 17, 91]`.  Every fold fits an independent softmax model and writes
held-out logits.  Positive temperature scaling is fitted only on those OOF
logits before a deployment model is refitted on all rows.

The Fortran app exports the packed deployment coefficients, intercepts,
temperature, labels, predictions, probabilities, and both OOF log losses.
The benchmark replays the coefficient layout and temperature with an
independent NumPy softmax oracle.  The recorded CPU run has maximum replay
error `2.220446049250313e-16`, probability-simplex error
`2.220446049250313e-16`, uncalibrated OOF log loss
`0.16213853637082157`, and calibrated OOF log loss
`0.03859049794844933`.  Fit time is `6.14588e-4 s`; prediction time is
`2.74283e-6 s` per batch on this small fixture.

CUDA is recorded as `unavailable` with a typed `FORTNUM_NOT_IMPLEMENTED`
boundary.  The complete softmax-plus-calibration graph is not resident on an
accelerator and no hidden host fallback is used.

Reproduce the lane with:

```bash
python3 -B scripts/bench_calibrated_softmax_cv.py \
  --fortml ../fortml --output results/calibrated_softmax_cv.csv
```

The CSV records source revisions, compiler and NumPy versions, timings,
independent-oracle errors, OOF diagnostics, and the device contract.
