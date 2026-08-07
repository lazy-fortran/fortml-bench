# Complement Naive Bayes workload

`scripts/bench_complement_nb.py` writes
[`complement_naive_bayes.csv`](complement_naive_bayes.csv).  The deterministic
fixture has 512 samples, eight nonnegative real-valued count features, and
sorted arbitrary integer labels `[-7, 3, 11]`.  Smoothing is `alpha=0.75` and
the input tangent is deterministic.

The NumPy implementation is the behavioral oracle.  It independently rebuilds
class masses, complement feature masses, smoothed complement probabilities,
positive `-log(q)` weights, priors, stable log-softmax probabilities,
predictions, and the analytic input JVP.  It requires complete finite arrays,
simplex normalization, and finite JVP values before timing fit, prediction,
and JVP.

Run the lane serially from this repository:

```bash
.venv/bin/python -B scripts/bench_complement_nb.py \
  --fortml ../fortml --output results/complement_naive_bayes.csv
```

The optional scikit-learn row is contextual.  `ComplementNB` has the same
smoothed feature weights, but its multiclass `_joint_log_likelihood` omits the
class-prior intercept; FortML's documented contract includes that intercept.
The row is therefore timed and reported without claiming bitwise prediction
equivalence.  Scikit-learn has no differentiable input-JVP API, so its JVP row
is an explicit refusal.

The intended FortML release target is `fortml_bench_complement_nb`.  It should
read `FORTML_BENCH_COMPLEMENT_NB_ORACLE`, write one-based CSV quantities
`log_probability`, `probability`, `log_probability_jvp`, and `prediction`, and
emit `complement_nb_fit`, `complement_nb_predict`, and `complement_nb_jvp`
timing records.  The harness checks every output element against NumPy with a
`2e-10` tolerance.  A missing target, build failure, or incomplete output is
retained as explicit `unavailable` rows and never treated as a timing result.

The checked-in CSV records source revisions, compiler flags, Python/NumPy/
scikit-learn versions, and the CPU-only boundary.  It contains no CUDA claim.
