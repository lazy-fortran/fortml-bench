# Leakage-safe calibrated logistic cross-validation

This lane exercises `calibrated_logistic_classifier_t` on 96 rows, two
features, three deterministic stratified folds, and arbitrary sorted labels
`[-3, 42]`. Each fold fits a separate logistic model and writes held-out
margins. The temperature map is fitted only from those out-of-fold margins,
then the deployment model is refit on all rows.

The release app exports the packed deployment vector, labels, predictions,
probabilities, fold count, and both out-of-fold log losses. The benchmark's
independent NumPy oracle replays the final coefficients and temperature,
checks every probability row, and checks the sorted-label prediction rule.
The recorded run has a maximum replay error of `2.0817e-17`, zero probability
simplex error, uncalibrated out-of-fold log loss `0.4690011479566449`, and
calibrated out-of-fold log loss `0.1526771421492992`. The CPU timings are
`2.18612e-4 s` for fitting and `6.09860e-6 s` per prediction batch. These are
correctness-gated reference timings for a small dense fixture.

The CUDA row is `unavailable` with a typed `FORTNUM_NOT_IMPLEMENTED` result.
The logistic model, calibration map, and batch are not resident on an
accelerator. Isotonic maps are available for fitting and prediction, while
their active-set derivative products return the same typed refusal.

Reproduce the lane with:

```bash
python3 -B scripts/bench_calibrated_logistic_cv.py \
  --fortml ../fortml --output results/calibrated_logistic_cv.csv
```

The CSV records the FortML source and benchmark revisions, compiler, NumPy
version, timings, independent-oracle errors, diagnostics, and the device
boundary.
