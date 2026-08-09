# XGBoost resident CUDA policy

This lane exercises the finite numeric XGBoost `gbtree` path and its explicit
device boundaries. Numeric native CUDA prediction is admitted only when it
matches the independent CPU prediction. Categorical partitions,
missing-default routing, ranking, and DART return `FORTNUM_NOT_IMPLEMENTED`
before changing caller output. Those rows are capability evidence rather than
GPU timings. The resident additive-tree ABI keeps model arrays on the device
and exposes transfer counters for query-only steady-state accounting.

Run:

```sh
python3 scripts/bench_xgboost_cuda.py --fortml ../fortml
```
