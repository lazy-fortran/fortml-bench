# Binary Laplace-GP log-probability gate

This lane checks the sklearn-style `predict_log_proba` surface for binary
Laplace GP classification.  The independent NumPy oracle evaluates both
MacKay-logistic and integrated-probit predictive maps, their input/parameter
chain rule, and the finite floor used when a probit tail rounds to zero.  The
FortML gate additionally fits the model, updates fixed-state kernel
parameters, checks parameter and input JVP/VJP products, and verifies the
explicit CUDA refusal.

Reproduce from the benchmark checkout:

```bash
python3 -B scripts/bench_gp_classification_log_proba.py \
  --fortml ../fortml \
  --output results/gp_classification_log_proba.csv
```

The recorded run used Python 3.14.6, NumPy 2.5.1, gfortran `-O3`, and source
revision `96dc11d55544acf203b8ebf59912fe9fd446f3cb` (the source worktree was
clean apart from its generated `fo` build cache).  The independent value/JVP
oracle's maximum absolute error was `1.998719523221837e-10`; the fitted
public-contract gate passed.
The CSV keeps the CUDA row as `unavailable`: covariance solves and Laplace
state are not resident, and no hidden host fallback is counted as GPU support.
