# Independent multilabel Laplace-GP hyperparameter optimization

This lane checks the packed per-label fixed-state objective for
`gp_multilabel_classification_t`. Two weighted binary Laplace heads use the
same fitted fixture, then each head receives its own RBF
`[log variance, log lengthscale]` block. The NumPy oracle independently solves
the candidate prior systems and compares the negative mode posterior, analytic
gradient, directional JVP, VJP, and central finite difference.

The fixture has 10 samples, two indicator columns, one feature, jitter
`1e-7`, and row weights `[1,.9,1.1,1,.8,1.2,1,1.1,.9,1]`. The initial packed
vector is `[log(1.3),log(.75),log(1.3),log(.75)]`. FortOpt L-BFGS-B uses
uniform bounds `[-1,1]`, 1000 iterations, 80 line-search evaluations, and
gradient, objective, and step tolerances `5e-3`, `1e-6`, and `1e-6`.

The release probe reports objective `10.269388551423564`, JVP
`0.0352806469372524`, and maximum NumPy product error below `3e-12`. The
optimizer reaches objective `9.080724252903265` in eight iterations with
status `FORTNUM_OK`. CUDA is an explicit unavailable capability because the
Laplace factors and independent reduction are not resident.

Run from the benchmark repository with:

```bash
python -B scripts/bench_gp_multilabel_independent_optimizer.py \
  --fortml ../fortml --output results/gp_multilabel_independent_optimizer.csv
```

Raw rows are in
[`gp_multilabel_independent_optimizer.csv`](gp_multilabel_independent_optimizer.csv).
