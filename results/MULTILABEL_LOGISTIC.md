# Multilabel logistic classification

This lane covers FortML's dense `multilabel_logistic_classifier_t`.  The
fixture has 192 samples, four continuous features, and three zero/one target
columns.  Each output is fitted by an independent L2-regularized logistic
head.  The NumPy implementation is an independent damped-Newton oracle for
the same averaged objective; it checks every positive probability and every
hard indicator before retaining timings.  scikit-learn's
`MultiOutputClassifier` is included as contextual CPU evidence.

Run it with:

```bash
python -B scripts/bench_multilabel_logistic.py \
  --fortml ../fortml --output results/multilabel_logistic.csv
```

The checked release record contains two NumPy-oracle rows, two FortML rows,
two contextual scikit-learn rows, and six explicit CUDA capability rows.  The
FortML probabilities agree with the independent oracle to `1.45e-7`; this is
below the `3e-5` release gate.  CUDA rows are `unavailable` because no
resident multi-head kernel is linked.  No host fallback is timed or described
as GPU execution.

The derivative contract is covered by the FortML unit test: fixed-fit input
and packed-parameter JVP/VJP products use finite-difference and adjoint
oracles.  Integer targets and thresholded hard predictions remain discrete.
