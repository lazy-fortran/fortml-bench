# Multiclass probability calibration

This lane covers `multiclass_probability_calibrator_t` on 192 deterministic
logit rows and three arbitrary integer classes (`-4`, `17`, `91`). It fits one
positive temperature by softmax negative log likelihood with the declared L2
term. The independent NumPy oracle uses the same positive-domain
inverse-temperature Newton recurrence, while prediction uses an independent
stable softmax implementation.

The release app exports sorted classes, every input label, prediction, all
probability columns, and the fitted temperature. The benchmark checks all of
those values before retaining fit or prediction timings. The raw record is
[`multiclass_probability_calibration.csv`](multiclass_probability_calibration.csv).
The recorded FortML temperature differs from the NumPy oracle by
`1.20e-09`, with probability error below `1.20e-09`; the difference is from
the release app's formatted CSV precision. The recorded CPU timings are
`2.62625e-04 s` per fit and `4.515625e-06 s` per prediction on the captured host.

Reproduce with:

```bash
python -B scripts/bench_multiclass_probability_calibration.py \
    --fortml ../fortml --output results/multiclass_probability_calibration.csv
```

CUDA is recorded as `unavailable`: no resident multiclass calibration kernel
is linked, and the benchmark never relabels host execution as GPU work.
