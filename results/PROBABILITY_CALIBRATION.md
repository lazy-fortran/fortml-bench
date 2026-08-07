# Probability calibration

This lane covers the binary `probability_calibrator_t` contract in FortML.
It uses one deterministic 256-row scalar-score fixture with arbitrary integer
labels (`10`, `42`) and checks the complete output array before retaining any
timing.  The NumPy implementation is independent: sigmoid calibration uses a
two-parameter damped Newton solve with the declared L2 objective, while
isotonic calibration uses weighted pool-adjacent-violators followed by linear
interpolation between weighted score knots.

Run it with:

```bash
python -B scripts/bench_probability_calibration.py \
  --fortml ../fortml --output results/probability_calibration.csv
```

The release app must emit labels, predictions, and both probability columns
for every row and method.  The recorded CPU run agrees with the independent
oracle to `5.84e-14` for sigmoid and `1.11e-16` for isotonic.  It records fit
and prediction timings for NumPy and FortML.  CUDA rows are explicit
`unavailable` capability records because no resident calibration kernel is
linked; a host fallback is never timed as GPU work.

The derivative boundary is part of the API contract: sigmoid score and
parameter JVP/VJP products are smooth, while isotonic score products are
available away from knots.  Isotonic knot products and fitted-parameter
products return `FORTNUM_NOT_IMPLEMENTED` rather than silently differentiating
through a changing PAVA active set.
