# Relaxed Bernoulli Naive Bayes workload

`scripts/bench_bernoulli_nb.py` writes the raw
[`bernoulli_naive_bayes.csv`](bernoulli_naive_bayes.csv).  The deterministic
fixture has 512 samples, eight features, and sorted arbitrary integer labels
`[-7, 3, 11]`.  Features are relaxed Bernoulli values in `[0.1, 0.9]`, which
keeps the input directional derivative inside the model's finite-domain
contract.  Smoothing is `alpha=0.5`.

The NumPy oracle independently computes class priors, smoothed feature
probabilities, the stable log-softmax, predictions, and the analytic input
JVP.  It checks the complete probability simplex, accuracy, and log loss
before recording fit, predict, or JVP timings.  The optional scikit-learn row
uses `BernoulliNB(alpha=0.5, binarize=None)` and is contextual; its complete
probability and prediction arrays agree with the NumPy oracle to floating-point
roundoff.

The intended FortML release-app protocol is:

```bash
.venv/bin/python -B scripts/bench_bernoulli_nb.py \
    --fortml ../fortml --output results/bernoulli_naive_bayes.csv
```

The harness builds with `fo build --flag -O3`, invokes
`fortml_bench_bernoulli_nb`, and sets `FORTML_BENCH_BERNOULLI_ORACLE`.  The
release app writes one-based CSV quantities `log_probability`, `probability`,
and `prediction` and emits `bernoulli_nb_fit`, `bernoulli_nb_predict`, and
`bernoulli_nb_jvp` timing records.  Every output array is checked against the
independent oracle.  The current FortML main includes this app, so the clean
release record contains passing FortML fit, predict, and JVP rows.  If a
target is absent in a future checkout, the same harness retains explicit
`unavailable` refusals rather than inferred timings.

The raw CSV records the exact FortML and benchmark revisions, compiler flags,
Python/NumPy/scikit-learn versions, and the refusal reason.  A missing target,
compiler, or optional package is always a machine-readable refusal; it never
silently removes a lane or relabels a NumPy timing as FortML evidence.
