# XGBoost row and feature sampling

This release-fixture lane checks one depth-one squared-error tree with
deterministic, without-replacement row sampling (`subsample=0.5`) and feature
sampling (`colsample_bytree=0.5`). The NumPy oracle independently advances the
same Park--Miller stream, enumerates every retained-feature threshold, evaluates
the regularized second-order gain, and predicts every input row. The release
fixture and oracle agree to machine precision; seed `12345` selects feature 1.

| phase | device | status | max absolute error | timing |
|---|---|---|---:|---:|
| fit/predict | CPU | pass | 0 | see CSV |
| prediction capability | CUDA | unavailable | — | — |

The CUDA row is a typed `FORTNUM_NOT_IMPLEMENTED` refusal. Tree growth and
prediction have no resident CUDA kernel in this release; the CPU result is not
relabeled as accelerator evidence.

Run:

```bash
python -B scripts/bench_xgboost_sampling.py \
  --fortml ../fortml --output results/xgboost_sampling.csv
```

Raw data: [`xgboost_sampling.csv`](xgboost_sampling.csv).
