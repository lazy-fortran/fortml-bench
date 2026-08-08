# Pinned exact-GP reference

This lane compares FortML's exact Gaussian-process regression with
scikit-learn's `GaussianProcessRegressor` and, when the optional dependencies
are installed, GPyTorch. The model is deliberately pinned rather than fitted:
all three implementations receive the same RBF length scale, signal
variance, observation noise, and deterministic inputs. The result therefore
measures Cholesky, cross-covariance, triangular-solve, and reduction work
instead of comparing three hyperparameter optimizers.

The independent Python side reconstructs the training and query arrays from
the same closed forms used by the Fortran app. It rejects the result before
timing if the predictive mean or variance sums disagree beyond the relative
`1e-8` gate. Fit and prediction are timed separately because they have
different scaling in the training and query sizes.

Run the lane with:

```bash
python -B scripts/bench_gp_vs_reference.py \
  --fortml ../fortml --output fixtures/gp_vs_reference.json
```

The committed fixture uses 400 training points, 4,000 queries, and eight
features with length scale `0.8`, signal variance `1.4`, and noise variance
`0.04`. In the recorded run FortML and scikit-learn agree to better than
`1e-14` in the reported sums. FortML takes `8.11 ms` to fit and `42.27 ms` to
predict; scikit-learn takes `9.84 ms` and `51.74 ms`, respectively (1.21x and
1.22x relative speedups). The fixture records GPyTorch as unavailable for that
environment. Installing the optional PyTorch/GPyTorch stack adds its rows
without changing the correctness gate or the pinned model.

This is a CPU reference lane. GPU measurements remain in the matched
PyTorch/KeOps/GPyTorch workloads, and no unavailable GPU or optional-package
row is counted as a performance pass.
