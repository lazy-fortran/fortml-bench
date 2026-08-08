# Variational GP classification correctness gate

This lane records the inducing-point Bernoulli variational-GP contract. The
independent NumPy fixture evaluates a two-inducing-point RBF ELBO with a fixed
normal table and checks the packed mean/log-Cholesky parameter gradient by
central finite differences. The Fortran gate independently exercises its
seeded Monte Carlo objective, analytic gradient, directional parameter JVP,
query-coordinate JVP/VJP products, variable-size minibatch scaling, malformed
labels, bounded FortOpt L-BFGS-B convergence, and CPU/CUDA dispatch.

The CSV contains one CPU correctness-gate row and one explicit CUDA capability
refusal. Gate wall time is not a model-throughput measurement, and no GPU
performance claim is inferred from the CPU row.

Reproduce:

```bash
python3 -B scripts/bench_gp_variational_classification.py \
  --fortml ../fortml --output results/gp_variational_classification.csv
```

The CUDA row remains `unavailable` until the inducing solve, likelihood table,
and reduction are implemented as a resident graph; a host fallback would make
the device contract misleading.
