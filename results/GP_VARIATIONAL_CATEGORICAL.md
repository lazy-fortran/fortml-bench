# Coupled categorical variational GP benchmark

`bench_gp_variational_categorical.py` checks the coupled categorical
variational-GP contract against an independent NumPy oracle. The oracle uses
the variance-corrected logits and checks the simplex and its directional JVP
with central differences. The FortML release test adds the categorical ELBO
gradient, packed and query-input JVP/VJP products, FortOpt fitting, sorted
labels, and typed CUDA refusals.

Run the lane from the benchmark repository:

```bash
FO_SCAN_FALLBACK=regex python3 -B scripts/bench_gp_variational_categorical.py \
  --fortml ../fortml \
  --output results/gp_variational_categorical.csv
```

The CSV records the FortML and benchmark revisions, compiler flags, NumPy
oracle errors, and a transfer boundary row. The coupled inducing graph has no
resident CUDA implementation yet, so CUDA is recorded as `unavailable` with
the typed `FORTNUM_NOT_IMPLEMENTED` refusal.
