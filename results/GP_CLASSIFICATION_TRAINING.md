# GP classification hyperparameter-training benchmark

This lane exercises the bounded FortOpt L-BFGS-B adapters for binary Laplace
classification and shared-kernel one-vs-rest multiclass classification. The
independent NumPy oracle reproduces the damped Newton mode solve, the negative
converged mode log-posterior, and the envelope gradient with respect to the
RBF log variance and log length scale. The bound solution and the invalid-bound
refusal are checked before timings are retained.

This is deliberately a mode-log-posterior benchmark, not a claim of full
Laplace evidence or likelihood-parameter optimization. The CPU variational
Bernoulli-GP contract has its own lane in
[`GP_VARIATIONAL_CLASSIFICATION.md`](GP_VARIATIONAL_CLASSIFICATION.md);
implicit mode HVPs, likelihood catalogs, and device-resident training remain
separate roadmap work.

Run:

```bash
.venv/bin/python -B scripts/bench_gp_classification_training.py \
  --fortml ../fortml --output results/gp_classification_training.csv
```

The checked-in CSV contains binary and shared-kernel multiclass CPU rows with
source revisions and independent objective/gradient errors.
