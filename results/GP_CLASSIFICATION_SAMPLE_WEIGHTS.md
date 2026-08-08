# Weighted Laplace-GP classification correctness gate

This lane records weighted binary and one-vs-rest Laplace-GP fitting. The
independent NumPy fixture implements the damped Newton Laplace recurrence for a
small RBF binary problem, including a zero-weight row, and checks the weighted
mode log posterior's kernel envelope gradient against refitted central
differences. The FortML gate independently checks logistic and probit fits,
weighted OVR composition, malformed-weight refusals, CPU prediction dispatch,
and the typed CUDA refusal.

The CSV contains independent-oracle, CPU public-contract, and explicit CUDA
capability rows. Gate wall time is not a model-throughput measurement, and no
GPU performance claim is inferred from the CPU row.

Reproduce:

```bash
python3 -B scripts/bench_gp_classification_sample_weights.py \
  --fortml ../fortml \
  --output results/gp_classification_sample_weights.csv
```

The CUDA row remains `unavailable` until the weighted covariance and Laplace
state execute as a resident graph; no hidden host fallback is permitted.
