# Finite-feature GP/NTK last-layer initializer

This lane evaluates `mlp_last_layer_gp_initializer_t` on a deterministic
`256 x 8` input fixture with a `16`-unit `tanh` hidden layer, two linear
outputs, and regularization `0.1`. The independent NumPy oracle reproduces
FortML's documented deterministic hidden-layer seed, forms the augmented
feature matrix, and solves the regularized normal equations. The expected
posterior-mean MSE is `7.538037387199728e-02`; the analytic regularization JVP
is checked separately by `test_mlp_last_layer_gp` against a central difference.

Run from this repository:

```bash
python3 scripts/bench_mlp_last_layer_gp.py \
  --fortml ../fortml --output results/mlp_last_layer_gp.csv
```

`results/mlp_last_layer_gp.csv` records the oracle timing and provenance. The
captured snapshot intentionally uses `--skip-fortml` because the dependency
checkout was being rebuilt; rerunning without that option is required to retain
the release-app CPU row. The CUDA row is `unavailable`: resident feature-map
and normal-equation kernels are not implemented, and no host timing is
relabeled as GPU evidence.

This is a finite-width last-layer posterior mean, not an exact NNGP/NTK or
infinite-width equivalence. Full NNGP covariance propagation and structure-
preserving GP-initialized networks remain separate roadmap work.
