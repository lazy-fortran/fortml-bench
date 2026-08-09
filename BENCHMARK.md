# Benchmark lanes

## Basis-composed linear regression

The basis-linear lane covers polynomial, cubic B-spline, and Fourier feature
maps with a fitted multi-output linear model.  It checks prediction, JVP, and
VJP products against independent NumPy finite-difference and adjoint oracles,
then records FortML release timings and the explicit CUDA capability boundary.

```bash
python -B scripts/bench_basis_linear_regression.py \
  --fortml ../fortml --output results/basis_linear_regression.csv
```

See [`results/BASIS_LINEAR_REGRESSION.md`](results/BASIS_LINEAR_REGRESSION.md)
for the fixture and reproducibility details.

## Sequential basis device dispatch

This lane checks explicit CPU dispatch for a sequential polynomial/Fourier
basis composition and the typed CUDA refusal for transform, JVP, VJP, and HVP.
An independent NumPy oracle covers the mixed input/log-frequency JVP and VJP
adjoint identity.

```bash
python -B scripts/bench_basis_sequential_device.py \
  --fortml ../fortml --output results/basis_sequential_device.csv
```

See [`results/BASIS_SEQUENTIAL_DEVICE.md`](results/BASIS_SEQUENTIAL_DEVICE.md)
for the fixture and refusal contract.

## Accumulated SGD momentum hyperproducts

This lane checks deterministic contiguous microbatch accumulation for the
fixed SGD/Nesterov trajectory objective. The Python oracle compares value,
gradient, JVP, VJP, and affine HVP products and records the typed CUDA
boundary.

```bash
python -B scripts/bench_sgd_momentum_hypergradient.py \
  --fortml ../fortml --output results/sgd_momentum_hypergradient_accumulation.csv
```

See
[`results/SGD_MOMENTUM_HYPERGRADIENT_ACCUMULATION.md`](results/SGD_MOMENTUM_HYPERGRADIENT_ACCUMULATION.md).

## Adagrad affine Hessian-vector products

This lane checks fixed full-batch Adagrad value/gradient/JVP products and the
exact outer HVP recurrence for a one-layer affine MLP.  The independent NumPy
oracle uses central differences of the trajectory gradient and records the
typed nonlinear-network and CUDA boundaries.

```bash
python -B scripts/bench_adagrad_hypergradient.py \
  --fortml ../fortml --output results/adagrad_hypergradient_hvp.csv
```

See [`results/ADAGRAD_HYPERGRADIENT_HVP.md`](results/ADAGRAD_HYPERGRADIENT_HVP.md)
for the fixture, tolerances, and timings. The current affine HVP row has
maximum oracle error `4.54e-7`. Nonlinear multi-layer outer HVPs and resident
Adagrad derivative state remain typed refusal rows.

## Weighted random-forest regression

This lane replays the seeded weighted CART bootstrap ensemble in NumPy. It
checks scalar and multi-output predictions, staged prefixes, bootstrap
inclusion, split-frequency importance, fixed-state products, and the typed
CUDA refusal.

```bash
python -B scripts/bench_random_forest_regression.py \
  --fortml ../fortml --output results/random_forest_regression.csv
```

See [`results/RANDOM_FOREST_REGRESSION.md`](results/RANDOM_FOREST_REGRESSION.md).

## Robust Poisson Gaussian process products

This lane checks the Poisson log-rate likelihood and fixed-mode robust-GP
posterior products against an independent NumPy oracle. It records likelihood
and posterior HVP errors, query JVP/VJP checks, FortOpt fitting, and the typed
CUDA refusal.

```bash
python -B scripts/bench_robust_gp_poisson_products.py \
  --fortml ../fortml --output results/robust_gp_poisson_products.csv
```

See [`results/ROBUST_GP_POISSON_PRODUCTS.md`](results/ROBUST_GP_POISSON_PRODUCTS.md).

## Multiclass Laplace-GP log probabilities

This lane checks sorted-label one-vs-rest `predict_log_proba`, input and packed
kernel-parameter products, and the typed CUDA boundary against an independent
NumPy normalization oracle.

```bash
python -B scripts/bench_gp_multiclass_log_proba.py \
  --fortml ../fortml --output results/gp_multiclass_log_proba.csv
```

See [`results/GP_MULTICLASS_LOG_PROBA.md`](results/GP_MULTICLASS_LOG_PROBA.md)
for the fixture and reproducibility details.

## Multilabel Laplace-GP log probabilities

This lane checks independent positive-label `predict_log_proba`, query-input
and packed per-label products, the packed shared-kernel JVP/VJP reduction,
threshold metadata, and output-preserving typed CUDA refusals against an
independent NumPy logarithm/adjoint oracle.

```bash
python -B scripts/bench_gp_multilabel_log_proba.py \
  --fortml ../fortml --output results/gp_multilabel_log_proba.csv
```

See [`results/GP_MULTILABEL_LOG_PROBA.md`](results/GP_MULTILABEL_LOG_PROBA.md)
for the fixture and reproducibility details.

## MLP automatic loss scaling gradient contract

This lane checks the independent growth/backoff recurrence together with the
trainer's allocation-free scale/check/unscale gradient products. The release
app verifies exact finite round trips, detects a scale-induced IEEE overflow,
and refuses the corresponding optimizer commit. FP32 and CUDA remain explicit
typed capability boundaries until resident master-weight kernels are released.

```bash
python -B scripts/bench_mlp_loss_scaling.py \
  --fortml ../fortml --output results/mlp_loss_scaling.csv \
  --report results/MLP_LOSS_SCALING.md
```

See [`results/MLP_LOSS_SCALING.md`](results/MLP_LOSS_SCALING.md) for the
independent NumPy oracle and provenance.

## Independent multilabel Laplace-GP hyperparameter optimization

This lane checks one independent RBF kernel-log block per multilabel head. The
NumPy oracle compares fixed-state objective value, gradient, JVP, VJP, and a
central directional finite difference, then records bounded FortOpt
L-BFGS-B and the typed CUDA capability boundary.

```bash
python -B scripts/bench_gp_multilabel_independent_optimizer.py \
  --fortml ../fortml --output results/gp_multilabel_independent_optimizer.csv
```

See
[`results/GP_MULTILABEL_INDEPENDENT_OPTIMIZER.md`](results/GP_MULTILABEL_INDEPENDENT_OPTIMIZER.md)
for the fixture and reproducibility details.

## Multiclass XGBoost log probabilities

This lane checks stable sorted-label one-vs-rest `predict_log_proba`, input and
packed leaf-coordinate JVP/VJP products, and explicit CPU/CUDA dispatch. The
independent NumPy oracle exercises a tail that would underflow under
`log(predict_proba)` and checks the normalized simplex.

```bash
python -B scripts/bench_xgboost_multiclass_log_proba.py \
  --fortml ../fortml --output results/xgboost_multiclass_log_proba.csv \
  --report results/XGBOOST_MULTICLASS_LOG_PROBA.md
```

See [`results/XGBOOST_MULTICLASS_LOG_PROBA.md`](results/XGBOOST_MULTICLASS_LOG_PROBA.md)
for the fixture, derivative errors, and typed CUDA refusal.

## Multiclass LightGBM log probabilities

This lane checks stable sorted-label one-vs-rest `predict_log_proba`, input and
packed leaf-coordinate JVP/VJP products, and explicit CPU/CUDA dispatch. The
independent NumPy oracle exercises a tail that would underflow under
`log(predict_proba)` and checks the normalized simplex.

```bash
python -B scripts/bench_lightgbm_multiclass_log_proba.py \
  --fortml ../fortml --output results/lightgbm_multiclass_log_proba.csv \
  --report results/LIGHTGBM_MULTICLASS_LOG_PROBA.md
```

See [`results/LIGHTGBM_MULTICLASS_LOG_PROBA.md`](results/LIGHTGBM_MULTICLASS_LOG_PROBA.md)
for the fixture, derivative errors, and typed CUDA refusal.
