# Random-forest permutation importance

`bench_random_forest_permutation.py` checks the fixed-state accuracy-decrease
diagnostic on a seeded 64-tree classifier. The independent NumPy oracle
replays the same Park--Miller Fisher--Yates stream, so both mean importance
and population repeat dispersion are compared before the Fortran timing row
is accepted.

Run:

```bash
python -B scripts/bench_random_forest_permutation.py \
  --fortml ../fortml --output results/random_forest_permutation.csv
```

The clean 2026-08-08 record uses source `d7f6cc4` and benchmark generation
revision `a0cf1bd`. It has four rows: fit, importance, dispersion, and the
typed CUDA-unavailable contract. The importance and dispersion oracle errors
are both zero. CUDA is unavailable because no resident permutation kernel is
linked; the refusal preserves every supplied output buffer.
