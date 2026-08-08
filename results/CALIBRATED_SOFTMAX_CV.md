# Leakage-safe multiclass calibrated softmax cross-validation

This lane exercises `calibrated_softmax_classifier_t` on 96 rows, two
features, three deterministic stratified folds, and sorted integer labels
`[-4, 17, 91]`. Every fold fits an independent softmax model and writes one
held-out logit row per sample. The selected calibration head is fitted only on
those OOF logits before the deployment model is refitted on all rows.

The wrapper now covers all three multiclass policies:

| method | packed calibration block | independent correctness gate | calibrated OOF log loss |
| --- | ---: | --- | ---: |
| temperature | one positive scalar | NumPy packed softmax replay, max error `2.220446049250313e-16` | `0.03859049794844933` |
| sigmoid (Platt) | `2 * n_classes` interleaved slope/intercept values | NumPy packed softmax/Platt replay, max error `2.220446049250313e-16` | `0.030199653456406416` |
| isotonic | fitted PAVA knots (zero trainable coordinates) | independent finite/simplex/label oracle; knot buffers are not packed parameters | `0.0` |

All methods have simplex error at most `2.220446049250313e-16`; the
uncalibrated OOF log loss is `0.16213853637082157`. Temperature and Platt
input/packed-parameter JVP/VJP products are smooth and exact. Isotonic
prediction is complete, while active-set JVP/VJP products return
`FORTNUM_NOT_IMPLEMENTED` explicitly. The CPU runs take approximately
`5.7e-4 s` to fit and `2.5e-6`--`9.2e-6 s` per prediction batch on this small
fixture. Each policy also probes a malformed two-row refit after deployment;
the candidate-commit contract preserves the prior probabilities and labels,
and the CSV records this as `transactional_refit_preserved=1`.

The source revision is `85cadc3`; the benchmark generator revision is
`500b9a5`. CUDA is recorded as `unavailable` with a typed
`FORTNUM_NOT_IMPLEMENTED` boundary. The complete softmax-plus-calibration
graph is not resident on an accelerator and no hidden host fallback is used.

Reproduce the lane with:

```bash
python3 -B scripts/bench_calibrated_softmax_cv.py \
  --fortml ../fortml --output results/calibrated_softmax_cv.csv
```

The CSV records one group per calibration policy with source revisions,
compiler and NumPy versions, timings, independent-oracle errors, OOF
diagnostics, packed-count checks, and the device contract.
