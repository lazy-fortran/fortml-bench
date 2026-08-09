# Variational multiclass GP log-probability benchmark

`bench_gp_variational_multiclass_log_proba.py` checks a stable one-vs-rest
log-probability contract. Its NumPy fixture uses branch-stable log-sigmoids,
row-wise log-sum-exp normalization, and a central finite difference for the
packed-link JVP. It also checks a tail case whose positive probabilities would
underflow before normalization.

The FortML gate runs
`test_gp_variational_multiclass_log_proba`. That test covers logistic and
probit links, packed parameter JVP and VJP products, fixed-state input JVP and
VJP products, CPU dispatch, and typed CUDA refusals. The release app records a
CPU prediction timing and verifies the probability simplex after exponentiating
the returned logs.

Run:

```bash
python3 -B scripts/bench_gp_variational_multiclass_log_proba.py \
  --fortml ../fortml \
  --output results/gp_variational_multiclass_log_proba.csv
```

The CUDA row is `unavailable` because the OVR inducing solves and log-sum-exp
reduction do not have resident CUDA kernels yet.
