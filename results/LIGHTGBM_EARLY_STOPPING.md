# LightGBM validation and early stopping

This correctness-gated release lane independently replays the one-feature leaf-wise Newton recurrence for regression and binary logistic objectives. Patience is two rounds; both ensemble-retention policies and malformed validation/CUDA refusals are checked.

| objective | policy | best iteration | retained estimators | early stopped | best validation loss |
|---|---|---:|---:|---:|---:|
| regression | restore_best | 1 | 1 | 1 | 4.0500000000000000e+01 |
| regression | retain_all | 1 | 3 | 1 | 4.0500000000000000e+01 |
| binary | restore_best | 1 | 1 | 1 | 1.3132616875182228e+00 |
| binary | retain_all | 1 | 3 | 1 | 1.3132616875182228e+00 |

Independent NumPy losses and the release app agree for every row. The app also emits `lgbm_early_invalid_validation,1` and `lgbm_early_cuda,3`; these typed contracts are required.

This is CPU correctness evidence, not resident CUDA performance evidence: LightGBM histogram prediction retains its explicit CUDA refusal.

Reproduce with:

```bash
python -B scripts/bench_lightgbm_early_stopping.py \
  --fortml ../fortml --output results/LIGHTGBM_EARLY_STOPPING.md
```

FortML revision: `f10dde5358ded9e05c144662653498b5f5e35219`
Benchmark revision: `0a448200e9e12d869fe9132a81cf221e46de6079+dirty`
Python 3.14.6, NumPy 2.5.1
