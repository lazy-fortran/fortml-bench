# Poisson XGBoost objective

The Poisson lane covers the log-link Newton objective in `xgboost_t`:

`gradient = exp(margin) - target`, `hessian = exp(margin)`.

The release app measures exact CPU depth-two fitting and prediction on a
deterministic 256-row count fixture, plus the weighted-quantile histogram
policy. Before timings are retained, the app's four-row one-tree fixture is
checked against an independent NumPy oracle: the base mean is 5 and the
expected means are `5 exp(-0.8)` on the left and `5 exp(0.8)` on the right.
The recorded maximum error is below `2e-15`.

The CUDA row is intentionally `unavailable`: no resident tree kernel is
linked, so FortML returns `FORTNUM_NOT_IMPLEMENTED`. CPU timings are not
presented as GPU performance.

Reproduce:

```bash
.venv/bin/python -B scripts/bench_xgboost_poisson.py \
  --fortml ../fortml --output results/xgboost_poisson.csv
```

Raw data: [`xgboost_poisson.csv`](xgboost_poisson.csv).
