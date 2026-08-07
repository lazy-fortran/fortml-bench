# Robust XGBoost-style objectives

This lane checks the weighted dense-tree implementation of Huber and pinball
(quantile) objectives. The fixture is a four-row, one-feature, one-tree
Newton problem with an independent hand-derived oracle for the base margin and
both leaf corrections. It records exact CPU fit/prediction timings and a typed
CUDA-unavailable row; no host fallback is counted as GPU work.

Run:

```bash
python -B scripts/bench_xgboost_robust.py \
  --fortml ../fortml --output results/xgboost_robust.csv
```

Raw data: [`xgboost_robust.csv`](xgboost_robust.csv).
