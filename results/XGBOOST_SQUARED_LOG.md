# Squared-log XGBoost objective

This lane checks FortML's `reg:squaredlogerror` (RMSLE) objective. The
four-row, one-tree fixture is solved independently in NumPy in the
`log1p(target)` coordinate: it checks the geometric base margin, analytic
gradient and positive-clipped Hessian, both leaf corrections, and the guarded
`expm1` inverse link. The release app must pass this gate before its exact
depth-two CPU timings and weighted-histogram parity row are retained.

The CUDA row is explicitly `unavailable`. FortML has no resident squared-log
tree kernel, so `FORTNUM_NOT_IMPLEMENTED` is recorded and CPU work is never
presented as GPU performance.

Reproduce:

```bash
python -B scripts/bench_xgboost_squared_log.py \
  --fortml ../fortml --output results/xgboost_squared_log.csv
```

Raw data: [`xgboost_squared_log.csv`](xgboost_squared_log.csv).
