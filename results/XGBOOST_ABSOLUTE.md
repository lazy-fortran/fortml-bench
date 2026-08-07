# Absolute-deviation XGBoost objective

This lane checks FortML’s `absolute`/`reg:absoluteerror` XGBoost-style
objective against an independent one-tree Newton oracle. The fixture uses a
weighted-median identity-link base margin and a positive Hessian floor for the
L1 subgradient. CPU fit and prediction timings are recorded separately; the
CUDA row is an explicit capability refusal until resident tree kernels exist.

```bash
.venv/bin/python -B scripts/bench_xgboost_absolute.py \
  --fortml ../fortml --output results/xgboost_absolute.csv
```

Raw data: [`xgboost_absolute.csv`](xgboost_absolute.csv).
