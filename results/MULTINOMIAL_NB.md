# Multinomial Naive Bayes workload

`scripts/bench_multinomial_nb.py` writes the raw
[`multinomial_naive_bayes.csv`](multinomial_naive_bayes.csv).  The deterministic
fixture has 512 samples, eight nonnegative real-valued count features, and
sorted arbitrary integer labels `[-7, 3, 11]`.  It uses token-mass smoothing
with `alpha=0.75` and a deterministic input tangent.

The NumPy implementation is the behavioral oracle.  It independently rebuilds
weighted class counts, smoothed feature probabilities, priors, stable
log-softmax probabilities, predictions, and the analytic input JVP.  It checks
complete arrays, the probability simplex, accuracy, log loss, and finite JVP
values before timing fit, predict, and JVP.  The optional scikit-learn
`MultinomialNB(alpha=0.75)` rows are contextual and are checked against the
same NumPy probabilities.  Scikit-learn has no input-JVP API, so that row is an
explicit `unavailable` record.

Run the lane serially from this repository:

```bash
.venv/bin/python -B scripts/bench_multinomial_nb.py \
    --fortml ../fortml --output results/multinomial_naive_bayes.csv
```

The harness builds with `fo build --flag -O3`, invokes the release target
`fortml_bench_multinomial_nb`, and sets
`FORTML_BENCH_MULTINOMIAL_ORACLE`.  The app writes one-based CSV quantities
`log_probability`, `probability`, `log_probability_jvp`, and `prediction`, and
emits `multinomial_nb_fit`, `multinomial_nb_predict`, and
`multinomial_nb_jvp` timing records.  Every output array is compared with the
independent oracle; the JVP tolerance is `2e-10`.  A missing target or optional
dependency remains a machine-readable `unavailable` refusal rather than an
inferred timing.

The raw CSV records the FortML and benchmark revisions, compiler flags,
Python/NumPy/scikit-learn versions, and refusal reasons.  The checked-in clean
record is CPU-only; no host timing is presented as CUDA evidence.
