# Exact GP and MLP workloads

Run date: 2026-08-06. The raw record is
[`model_workloads.csv`](model_workloads.csv). Times below are milliseconds per
operation. Setup and release-build time are excluded.

This is a historical device-comparison snapshot. Its raw rows deliberately
retain the FortML and benchmark dirty-tree revisions that were present on
2026-08-06. The clean 2026-08-07 release evidence for the new classifier,
GaussianNB, logistic-objective, and host-only CUDA boundaries is in
[`FEATURES.md`](FEATURES.md), [`CLASSIFICATION_MODELS.md`](CLASSIFICATION_MODELS.md),
and [`XGBOOST.md`](XGBOOST.md). Do not mix the historical timings with those
current release rows without matching source and toolchain revisions.

| workload | phase | FortML, gfortran CPU | FortML, nvfortran CPU | PyTorch/GPyTorch CPU | PyTorch/GPyTorch CUDA |
|---|---:|---:|---:|---:|---:|
| exact GP | fit | 0.796 | 0.585 | 0.393 | 0.693 |
| exact GP | predict | 0.576 | 0.503 | 0.492 | 0.808 |
| MLP | forward | 0.248 | 0.241 | 0.124 | 0.0544 |
| MLP | VJP | 0.405 | 0.832 | 0.331 | 0.359 |

These are deliberately small workloads. CUDA launch overhead is visible: it
helps the MLP forward pass, but not exact GP fit, exact GP prediction, or the
MLP reverse product. FortML's exact GP and MLP application paths are host-only,
so their CUDA rows are recorded as `unsupported`, not inferred from CPU data.

The 30-percent CPU gate compares the faster of the two FortML compiler lanes
with the matched Python CPU lane for each complete call:

| workload and phase | fastest FortML / reference | within 30 percent |
|---|---:|:---:|
| exact GP fit | 1.49 | no |
| exact GP predict | 1.02 | yes |
| MLP forward | 1.94 | no |
| MLP VJP | 1.23 | yes |

The gate therefore passes for exact-GP prediction and the MLP VJP, but not for
exact-GP fitting or the MLP forward pass. It is not generalized to CUDA,
where FortML has no matching public path.

## Matched work

The exact GP has 128 training points, four features, two output columns, and
32 prediction points. Every backend uses an RBF covariance with variance 1.4,
lengthscale 0.9, observation-noise variance 0.08, jitter `1e-10`, and float64
arithmetic. `fit` constructs the dense training covariance, factors it, and
solves for both target columns. `predict` constructs the training-test and
test-test covariances, applies the stored factor to all 32 cross-covariance
columns, and returns the complete posterior mean and latent variance.

The MLP is a 16-32-4 tanh network evaluated on a batch of 512. The 676
parameters, inputs, and output cotangent are deterministic and identical in
both implementations. `forward` returns the complete 512-by-4 prediction.
`VJP` includes the forward work needed by the reverse pass and returns both the
676 parameter adjoints and the 512-by-16 input adjoints. It is a training
product, not an optimizer step or a training epoch.

Before any timing, a separate NumPy implementation checks the full GP mean and
variance or the full MLP prediction and both adjoint arrays. The largest
recorded FortML error is below `3.3e-14`. The largest PyTorch MLP error is below
`4.0e-14`. GPyTorch's largest posterior error is `6.1e-9`. The training
covariance is ill-conditioned. The GP fit is therefore checked through its
posterior rather than by comparing unstable solve coordinates.

## Run conditions and reproduction

The CPU lane is pinned to one core of an AMD Ryzen 9 5950X. The GPU lane uses
an NVIDIA GeForce RTX 5060 Ti with driver 610.43.03. The record contains GNU
Fortran 16.1.1 with `-O3 -march=native`, nvfortran 26.5 with
`-O3 -mp=multicore`, Python 3.14.6, NumPy 2.5.1, PyTorch 2.13.0+cu130, and
GPyTorch 1.15.2. It also stores source revisions, the dirty-tree diff hash,
driver and application hashes, affinity, repetitions, warmups, peak memory,
executable size, and build time.

Run the benchmark and regenerate the plots with:

```sh
.venv/bin/python -B scripts/bench_model_workloads.py \
    --output results/model_workloads.csv
.venv/bin/python -B scripts/plot_model_workloads.py \
    --input results/model_workloads.csv --output-dir results
```

The harness switches a shared FortML build tree between compilers. Do not run
another `fo` build concurrently. The figures are
[`exact_gp_workloads.png`](exact_gp_workloads.png) and
[`mlp_workloads.png`](mlp_workloads.png).
