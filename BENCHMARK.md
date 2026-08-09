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
for the fixture, tolerances, and timings.

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
