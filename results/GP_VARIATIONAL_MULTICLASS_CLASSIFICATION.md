# Variational GP multiclass prediction benchmark

`bench_gp_variational_multiclass_classification.py` independently checks
one-vs-rest Bernoulli-link probabilities, simplex normalization, and the
packed parameter JVP by central finite differences.  The FortML release test
adds inducing-point ELBO/gradient/JVP checks, sorted arbitrary integer labels,
unknown-label refusal, and CPU/CUDA dispatch behavior.

Run:

```bash
python3 -B scripts/bench_gp_variational_multiclass_classification.py \
  --fortml ../fortml \
  --output results/gp_variational_multiclass_classification.csv
```

The multiclass inducing-point graph has no resident CUDA implementation yet,
so the CUDA row is explicitly `unavailable`.
