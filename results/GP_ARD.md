# ARD RBF kernel and exact-GP products

This correctness-gated lane exercises the anisotropic squared-exponential
kernel with one positive length scale per feature. The release fixture checks
the dense value matrix, both input gradients, the mixed input Hessian,
parameter JVP/VJP/HVP products, and an exact GP posterior, log marginal
likelihood, and packed hyperparameter gradient. A direct NumPy implementation
of the covariance and analytic products is the independent oracle; host rows
are retained only after every value agrees to the recorded tolerance.

| workload | device | status | max absolute error | timing |
|---|---|---:|---:|---:|
| ARD kernel value and derivative products | CPU | pass | see CSV | see CSV |
| exact GP ARD fit/predict/hypergradient | CPU | pass | see CSV | see CSV |
| resident ARD covariance capability | CUDA | unavailable | — | — |

The CUDA row is the typed `FORTNUM_NOT_IMPLEMENTED` contract. Resident CUDA
covariance currently supports isotropic leaves only; the CPU result is not
relabeled as accelerator evidence.

Run:

```bash
python -B scripts/bench_gp_ard.py \
  --fortml ../fortml --output results/gp_ard.csv
```

Raw data: [`gp_ard.csv`](gp_ard.csv).
