# Finite-feature GP posterior variance

This lane extends the finite-feature MLP last-layer kernel-ridge initializer
with the exact posterior predictive variance diagonal
`diag(Z (Z^T Z + lambda I)^(-1) Z^T)` at unit observation-noise scale and its
analytic regularization JVP. The hidden feature map, fixture, and positive
regularization `lambda=0.1` match `MLP_LAST_LAYER_GP.md`.

The independent NumPy harness in
`scripts/bench_mlp_last_layer_gp_posterior.py` forms the precision matrix and
solves it twice for the variance JVP. The release app reports fit, prediction,
variance timings, posterior-mean MSE, and variance/JVP checksums. The CPU row
must agree with NumPy within `3e-11`; the CUDA row is explicitly unavailable
because resident feature-map and precision kernels are not linked.

Run:

```bash
python3 scripts/bench_mlp_last_layer_gp_posterior.py \
  --fortml ../fortml --output results/mlp_last_layer_gp_posterior.csv
```

This contract is deliberately finite-width. It does not claim exact NNGP or
infinite-width NTK covariance propagation, sampled posterior weights, or
physics-preserving initialization.
