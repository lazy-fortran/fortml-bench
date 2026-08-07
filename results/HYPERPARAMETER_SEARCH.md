# Hyperparameter search

This lane exercises `fortml_hyperparameter_search` with a three-parameter
quadratic objective. NumPy reconstructs the 5×5×5 Cartesian grid and its
analytic minimizer before accepting the Fortran rows. The grid evaluates 125
points; FortOpt L-BFGS-B must reach the exact minimum within the documented
tolerance.

The CUDA row is explicitly `unavailable`: generic search has no resident CUDA
objective/search state and returns `FORTNUM_NOT_IMPLEMENTED` rather than
falling back to the host.

Reproduce:

```bash
.venv/bin/python -B scripts/bench_hyperparameter_search.py \
  --fortml ../fortml --output results/hyperparameter_search.csv
```

Raw data: [`hyperparameter_search.csv`](hyperparameter_search.csv).
