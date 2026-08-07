# Weighted ordinal logistic classification

This lane covers `ordinal_logistic_classifier_t` on 192 samples, four
continuous features, and four arbitrary ordered integer labels (`-9`, `4`,
`13`, `42`). A deterministic positive sample-weight vector is used in the
fit. The model is a single linear score with three cumulative-logit cut points.

The independent SciPy oracle optimizes the same weighted negative
log-likelihood with L2 regularization using unconstrained positive threshold
increments. Every FortML probability, prediction, and stored class label is
checked before retaining timings. The recorded CSV contains five rows: one
SciPy oracle row, two FortML CPU rows (fit and prediction), and two explicit
CUDA capability rows. The FortML probability error is `1.51e-7`, below the
`3e-5` release gate; ordinal accuracy is `0.625` on this deliberately noisy
fixture.

Reproduce with:

```bash
.venv/bin/python -B scripts/bench_ordinal_logistic.py \
    --fortml ../fortml --output results/ordinal_logistic.csv
```

The Fortran unit test independently checks cumulative-logit probabilities,
ordered argmax labels, input and packed-parameter JVP finite differences,
both VJP adjoint identities, weighted fit metadata, and CPU/CUDA device
contracts. CUDA rows are `unavailable`: no resident ordinal kernel is linked,
and host execution is never relabeled as accelerator evidence.
