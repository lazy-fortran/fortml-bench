# Successive-halving hyperparameter search

This lane checks a deterministic multi-fidelity search over a three-parameter
quadratic objective. The independent NumPy oracle expects 64 candidates,
resource rungs 1, 2, 4, 8, and 16, and 124 total objective evaluations after
factor-two pruning. Every callback returns a value and analytic parameter
gradient. The surviving vector is refined at resource 16 through FortOpt
L-BFGS-B, whose value must equal the analytic resource penalty `0.25 / 16`.

The CUDA row records `FORTNUM_NOT_IMPLEMENTED` because the generic search state
is CPU-owned. No host fallback is counted as resident GPU execution.

Reproduce:

```bash
.venv/bin/python -B scripts/bench_hyperparameter_successive_halving.py   --fortml ../fortml --output results/hyperparameter_successive_halving.csv   --report results/HYPERPARAMETER_SUCCESSIVE_HALVING.md
```

Raw data: [`hyperparameter_successive_halving.csv`](hyperparameter_successive_halving.csv).
