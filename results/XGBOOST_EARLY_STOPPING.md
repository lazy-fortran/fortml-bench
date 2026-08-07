# XGBoost validation and early stopping

This release-app lane independently replays exact depth-one Newton
updates for squared, binary logistic, and squared-log objectives.
Validation loss is evaluated after every stage; patience is two
consecutive non-improving rounds. Both `restore_best` policies are
checked, and malformed validation data must return
`FORTNUM_DOMAIN_ERROR` (code 1).

| objective | policy | best iteration | retained estimators | early stopped | best validation loss |
|---|---|---:|---:|---:|---:|
| squared | restore_best | 1 | 1 | 1 | 4.0500000000000000e+01 |
| squared | retain_all | 1 | 3 | 1 | 4.0500000000000000e+01 |
| logistic | restore_best | 1 | 1 | 1 | 1.3132616875182228e+00 |
| logistic | retain_all | 1 | 3 | 1 | 1.3132616875182228e+00 |
| squared_log | restore_best | 2 | 2 | 1 | 2.2945694984880318e+00 |
| squared_log | retain_all | 2 | 4 | 1 | 2.2945694984880318e+00 |

The independent oracle and release app agree for every row. The
release app also emits `xgb_early_invalid_validation,1`; the benchmark
fails if that typed refusal changes. This is CPU correctness evidence,
not resident CUDA performance evidence: XGBoost tree prediction still
uses FortML's explicit CUDA refusal contract.

Reproduce:

```bash
python -B scripts/bench_xgboost_early_stopping.py \
  --fortml ../fortml --output results/XGBOOST_EARLY_STOPPING.md
```

FortML revision: `d9d09e52737defc4642b46a2ef3da1602361909b`
Benchmark revision: `9e6f87af160c955c2ca7ceb4f8cebcce87879b48`
Python 3.14.6, NumPy 2.5.1
