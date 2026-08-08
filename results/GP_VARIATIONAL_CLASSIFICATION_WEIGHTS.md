# Weighted variational GP classification correctness gate

This lane records the weighted Bernoulli and one-vs-rest variational-GP
objective contract. The independent NumPy fixture uses a fixed normal table,
nonuniform row weights, and a dense RBF inducing solve. It checks uniform
likelihood scaling (the KL is not scaled) and the packed variational gradient
against central finite differences. The FortML gate independently exercises
weighted binary and OVR ELBOs, directional products, malformed-weight
refusals, CPU dispatch, and typed CUDA refusal.

The CSV contains independent-oracle, CPU public-contract, and explicit CUDA
capability rows. Gate wall time is not a model-throughput measurement, and no
GPU performance claim is inferred from the CPU row.

Reproduce:

```bash
python3 -B scripts/bench_gp_variational_classification_weights.py \
  --fortml ../fortml \
  --output results/gp_variational_classification_weights.csv
```

The CUDA row remains `unavailable` until the weighted inducing solve,
likelihood, and reduction execute as a resident graph; a host fallback would
make the device contract misleading.
