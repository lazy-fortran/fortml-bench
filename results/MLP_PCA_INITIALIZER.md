# PCA-seeded linear MLP benchmark

This lane measures `mlp_t%initialize_from_pca` on a deterministic `512 x 16`
fixture with eight retained components. NumPy's centered thin SVD is the
independent oracle: it applies the same rank-truncated projection and inverse
map and gates the FortML timing by reconstruction RMSE. The initializer is a
finite linear/PCA optimum, not an NNGP, NTK, or GP-posterior equivalence.

Run from this repository:

```bash
python3 scripts/bench_mlp_pca_initializer.py \
  --fortml ../fortml --output results/mlp_pca_initializer.csv
```

The CSV records Python/compiler and source revisions, the independent oracle,
the CPU release-app timing, and a typed CUDA-unavailable row. No host timing
is relabeled as GPU execution until a resident dense MLP/PCA lowering exists.
