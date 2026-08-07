# Differentiable multilabel MLP classifier

This release-fixture lane exercises two independent sigmoid heads over a
deterministic two-feature fixture. Each head performs one full-batch Adam
update. The independent NumPy oracle checks the concatenated parameters,
probabilities, binary indicators, summed BCE objective, gradient, and exact
parameter Hessian-vector product. Recorded CPU rows are accepted only when all
outputs agree to within `3e-11`.

| phase | device | status | max absolute error | timing |
|---|---|---|---:|---:|
| fit/predict | CPU | pass | see CSV | see CSV |
| derivatives | CPU | pass | see CSV | — |
| probability capability | CUDA | unavailable | — | — |

The CUDA row is a typed `FORTNUM_NOT_IMPLEMENTED` refusal: no resident
multilabel MLP graph is linked in this release. Host derivative results are
not reported as GPU execution.

Run:

```bash
python -B scripts/bench_mlp_multilabel_classifier.py \
  --fortml ../fortml --output results/mlp_multilabel_classifier.csv
```

Raw data: [`mlp_multilabel_classifier.csv`](mlp_multilabel_classifier.csv).
