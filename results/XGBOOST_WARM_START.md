# XGBoost warm-start continuation

The raw record is [`xgboost_warm_start.csv`](xgboost_warm_start.csv), generated
by [`scripts/bench_xgboost_warm_start.py`](../scripts/bench_xgboost_warm_start.py).
The release executable is `fortml_bench_xgboost_warm_start`.

The fixture fits two depth-one squared-loss trees, continues the fitted prefix
to four trees, and compares the fourth staged margin with a fresh four-tree
fit and an independent NumPy Newton-stump replay. The release app also checks
transactional refusals for a non-increasing target, changed learning rate, and
an unfitted source. The CUDA row is an explicit unavailable capability record
because warm-start continuation has no resident CUDA entry point.

Run:

```bash
python -B scripts/bench_xgboost_warm_start.py \
  --fortml ../fortml --output results/xgboost_warm_start.csv
```
