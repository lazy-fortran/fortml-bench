# Differentiable binary MLP classifier

This release-fixture lane exercises the one-logit binary classifier on a
deterministic two-feature fixture. It performs one full-batch Adam update with
the production Xavier/trigonometric initializer, then checks classes,
probabilities, labels, the weighted BCE value and gradient, and the exact
parameter Hessian-vector product. An independent NumPy oracle reproduces every
operation; the recorded CPU rows are accepted only when all outputs agree to
within `3e-11`.

| phase | device | status | max absolute error | timing |
|---|---|---|---:|---:|
| fit/predict | CPU | pass | see CSV | see CSV |
| derivatives | CPU | pass | see CSV | — |
| probability capability | CUDA | unavailable | — | — |

The CUDA row is a typed `FORTNUM_NOT_IMPLEMENTED` refusal: no resident MLP
classifier graph is linked in this release. Host derivative results are not
reported as GPU execution.

Run:

```bash
python -B scripts/bench_mlp_binary_classifier.py \
  --fortml ../fortml --output results/mlp_binary_classifier.csv
```

Raw data: [`mlp_binary_classifier.csv`](mlp_binary_classifier.csv).
