# Multiclass XGBoost validation and early stopping

This release lane independently replays the exact depth-one OVR
logistic Newton updates for arbitrary labels `[-8, 2, 11]`.  It
checks weighted normalized multiclass log-loss, deterministic
patience/min-delta metadata, restored best-prefix staged probabilities,
the typed CUDA refusal, and transactional unknown-label rejection.

| metric | observed | independent oracle |
|---|---:|---:|
| best iteration | 1 | 1 |
| requested estimators | 5 | 5 |
| retained estimators | 1 | 1 |
| weighted validation log-loss | 9.4309471214668528e-01 | 9.4309471214668528e-01 |
| staged probability max error | 0.0000000000000000e+00 | <= 3e-12 |

All CPU rows agree with the independent oracle. CUDA is recorded as
`unavailable` with the typed resident-tree refusal; no host fallback is
included in the timing result.

Reproduce:

```bash
python -B scripts/bench_xgboost_multiclass_validation.py \
  --fortml ../fortml --output results/xgboost_multiclass_validation.csv \
  --report results/XGBOOST_MULTICLASS_VALIDATION.md
```

FortML revision: `08f35af5d5cbb539f8244f138bbd7bbac9ab6259`
Benchmark revision: `c6e82ae2327454799261b65f8e357eb9c09b32b7`
Python 3.14.6, NumPy 2.5.1
