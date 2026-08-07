# Trainable exact-GP constant and linear means

This lane fits exact RBF GPs with a trainable constant mean and a trainable
intercept-plus-linear mean. An independent NumPy covariance oracle checks
packed parameters, posterior means and variances, log marginal likelihood,
analytic hyperparameter gradients, and mean-only HVP directions. Both CPU rows
agree to the recorded tolerance before timing is retained.

| workload | device | status | max absolute error | timing |
|---|---|---|---:|---:|
| constant mean | CPU | pass | see CSV | see CSV |
| linear mean | CPU | pass | see CSV | see CSV |
| exact-GP mean capability | CUDA | unavailable | — | — |

The CUDA row is the typed `FORTNUM_NOT_IMPLEMENTED` contract: exact GP
factorization and trainable mean fitting have no resident CUDA implementation
in this release. Host results are not relabeled as GPU evidence.

Run:

```bash
python -B scripts/bench_gp_mean.py \
  --fortml ../fortml --output results/gp_mean.csv
```

Raw data: [`gp_mean.csv`](gp_mean.csv).
