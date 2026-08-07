# Exact-GP hyperparameter multistart benchmark

This lane exercises the exact-GP FortOpt L-BFGS-B adapter with four bounded
starts. The first start reuses the fitted state; the remaining starts are
seeded uniform draws. FortML retains only finite converged runs and restores
the lowest negative log marginal likelihood. A NumPy dense covariance solve
and central finite-difference gradient independently check the reported
objective and gradient norm before the timing row is written.

The CUDA row is an explicit unavailable result. Exact-GP factorization and
hyperparameter optimization do not claim resident CUDA execution, and the
release application checks the `FORTNUM_NOT_IMPLEMENTED` refusal.

Run:

```bash
.venv/bin/python -B scripts/bench_gp_hyperparameter_training.py \
  --fortml ../fortml --output results/gp_hyperparameter_training.csv
```

Raw data: [`gp_hyperparameter_training.csv`](gp_hyperparameter_training.csv).
