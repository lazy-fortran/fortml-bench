# Structure-aware MLP GP initializer

This lane measures `fortml_mlp_structure_gp` on a deterministic `256 x 8`
input batch, a 16-unit `tanh` hidden layer, and two affine outputs. The
initializer fits the finite-feature kernel-ridge posterior and verifies that
the packed hidden parameter prefix remains unchanged.

Run it from this repository with:

```bash
python3 -B scripts/bench_mlp_structure_gp.py \
  --fortml ../fortml --output results/mlp_structure_gp.csv
```

The NumPy row is an independent dense solve of
`(Z^T Z + lambda I) C = Z^T Y`, where `Z` is the deterministic hidden feature
map with an intercept. The FortML row is retained only when its posterior MSE
is within `2e-12` of that oracle and its hidden-parameter delta is within
`2e-13`. The current CSV records a `2.78e-17` MSE difference and zero hidden
delta on the CPU release app.

The CUDA row is typed `unavailable`. Resident structure-aware GP/MLP kernels
are not implemented, so no host feature-map or solve timing is relabeled as a
GPU measurement.
