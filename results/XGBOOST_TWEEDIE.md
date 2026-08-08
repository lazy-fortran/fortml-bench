# Tweedie XGBoost objective

This lane covers FortML's bounded XGBoost `reg:tweedie` objective for the
compound-Poisson variance-power interval `1 < p < 2`. Margins are log means;
the inverse link is `exp(margin)`. The release app reports a deterministic
four-row, one-tree Newton fixture and exact/histogram CPU timings on a larger
nonnegative-target fixture.

Before timings are retained, `bench_xgboost_tweedie.py` reconstructs the
gradient
`-y*exp((1-p)*margin) + exp((2-p)*margin)` and Hessian
`y*(p-1)*exp((1-p)*margin) + (2-p)*exp((2-p)*margin)` independently in NumPy.
The app's maximum prediction error against that oracle is gated at
`3e-13`. The same objective value is exposed by `xgb_tweedie_loss`; invalid
powers, negative targets, and nonfinite products are refusal cases in the
FortML test suite.

The CUDA row is explicitly `unavailable`: no resident tree kernel is linked,
so `FORTNUM_NOT_IMPLEMENTED` is recorded and CPU work is never relabeled as
GPU performance.

Reproduce:

```bash
python -B scripts/bench_xgboost_tweedie.py \
  --fortml ../fortml --output results/xgboost_tweedie.csv
```

Raw data: [`xgboost_tweedie.csv`](xgboost_tweedie.csv).
