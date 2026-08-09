# Benchmark lanes

## Trainer validation direction and checkpoint replay

This lane checks the model-agnostic trainer's patience and best-state
restoration for both loss metrics (minimize, the default) and score metrics
(`validation_higher_is_better`). The independent NumPy oracle checks the
known-answer trajectories and the release test checks schema-6 checkpoint
continuation plus the transactional callback-presence refusal.

```bash
python -B scripts/bench_trainer_validation.py \
  --fortml ../fortml --output results/trainer_validation.csv
```

See [`results/TRAINER_VALIDATION.md`](results/TRAINER_VALIDATION.md) for the
fixture and reproducibility details.

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

## Fixed-latent Student-t GP likelihood products

This lane checks the normalized Student-t observation density over stable
`[log(scale),log(nu)]` coordinates. An independent NumPy scalar oracle checks
value, gradient, JVP, VJP, and directional HVP products; the release probe also
passes the negative-log-likelihood callback to FortOpt and records objective
decrease. CUDA is an explicit typed refusal until resident latent batches and
special functions are linked.

```bash
python -B scripts/bench_gp_student_t_likelihood.py \
  --fortml ../fortml --output results/gp_student_t_likelihood.csv \
  --report results/GP_STUDENT_T_LIKELIHOOD.md
```

See [`results/GP_STUDENT_T_LIKELIHOOD.md`](results/GP_STUDENT_T_LIKELIHOOD.md)
for the fixture, tolerances, provenance, and refusal contract.

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
and refuses the corresponding optimizer commit. Its FP32 rows compare the
binary64 master trajectory and schema-11 checkpoint metadata with an
independent NumPy float32-boundary recurrence. FP16, BF16, and CUDA remain
explicit typed capability boundaries.

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

## Separable Hamiltonian leapfrog JVP

This lane checks the exact parameter/state tangent of one separable
velocity-Verlet step, the primal-map equivalence, and the typed refusal for a
general nonseparable Hamiltonian. An independent NumPy harmonic-oscillator
oracle checks the state tangent and canonical symplectic form before the
Fortran gate.

```bash
python -B scripts/bench_symplectic_leapfrog_jvp.py \
  --fortml ../fortml --output results/symplectic_leapfrog_jvp.csv
```

See [`results/SYMPLECTIC_LEAPFROG_JVP.md`](results/SYMPLECTIC_LEAPFROG_JVP.md)
for the contract, provenance, and explicit CUDA boundary.

## Semantic basis feature labels

This lane checks optional semantic output names on `basis_map_t` and their
qualified propagation through horizontal, sequential, and column-selecting
pipelines. An independent NumPy fixture assembles polynomial and Fourier
features directly; the Fortran gate checks duplicate-name transactionality and
that labels leave values and packed layouts unchanged.

```bash
python -B scripts/bench_basis_feature_names.py \
  --fortml ../fortml --output results/basis_feature_names.csv \
  --report results/BASIS_FEATURE_NAMES.md
```

See [`results/BASIS_FEATURE_NAMES.md`](results/BASIS_FEATURE_NAMES.md) for
the fixture and provenance.

## Matérn-5/2 second-derivative GP hyperproducts

This lane checks the bounded one-dimensional exact GP with mixed value,
first-derivative, and second-derivative observations. The independent NumPy
oracle assembles the order-four Matérn-5/2 covariance, central-differences the
prediction and query functional, and checks likelihood gradients and HVPs over
the packed variance, lengthscale, and noise coordinates. The release app adds
CPU timings for prediction, input JVP/VJP, and both hyperproducts. Selected
CUDA is recorded as a typed refusal until resident derivative covariance and
factorization kernels are linked.

```bash
python -B scripts/bench_second_derivative_gp_matern52_hyperparameters.py \
  --fortml ../fortml \
  --output results/second_derivative_gp_matern52_hyperparameters.csv \
  --report results/SECOND_DERIVATIVE_GP_MATERN52_HYPERPARAMETERS.md
```

See
[`results/SECOND_DERIVATIVE_GP_MATERN52_HYPERPARAMETERS.md`](results/SECOND_DERIVATIVE_GP_MATERN52_HYPERPARAMETERS.md)
for the independent oracle, provenance, tolerances, and timing rows.

## Uniform empirical quantile transformer

This lane checks the fitted per-feature order-statistic map, inverse
interpolation, endpoint clamping, and the fixed-segment input JVP against an
independent NumPy oracle. It records the CPU release timing and the explicit
typed CUDA boundary. Normal-output quantiles and power transforms remain
separate roadmap contracts.

Run:

    python -B scripts/bench_quantile_transformer.py \
      --fortml ../fortml --output results/quantile_transformer.csv \
      --report results/QUANTILE_TRANSFORMER.md

See results/QUANTILE_TRANSFORMER.md for the fixture, oracle errors, and
provenance.

## Resident CUDA dense MLP chain

This lane checks a three-layer dense chain against an independent NumPy
recurrence. It covers value, packed input/parameter JVP, packed input/parameter
VJP, a central directional finite difference, and the reverse-mode adjoint
identity. The Fortran gate checks ordinary-build typed refusal and sentinel
preservation. Native CUDA is executed only when `nvcc` and a CUDA device are
available; otherwise the CSV records an explicit typed refusal rather than a
CPU GPU claim.

```bash
python -B scripts/bench_cuda_mlp_chain.py \
  --fortml ../fortml --output results/cuda_mlp_chain.csv \
  --report results/CUDA_MLP_CHAIN.md
```

See [`results/CUDA_MLP_CHAIN.md`](results/CUDA_MLP_CHAIN.md) for the oracle,
transfer/residency contract, tolerances, and provenance.

## Power transformer

This lane compares fixed-lambda Yeo--Johnson and Box--Cox transforms against
independent NumPy branch oracles. It checks transform checksums, inverse
reconstruction, and the explicit resident-CUDA boundary. The release app uses
the same 256-row fixture and records CPU elapsed time and fitted lambda state.

```bash
python -B scripts/bench_power_transformer.py \
  --fortml ../fortml --output results/power_transformer.csv \
  --report results/POWER_TRANSFORMER.md
```

See [`results/POWER_TRANSFORMER.md`](results/POWER_TRANSFORMER.md) for oracle
errors, provenance, and the typed device row.

## LightGBM query-weighted rank:pairwise

This lane checks `lightgbm_t%fit_ranking` against a direct NumPy pair loop.
The oracle covers minimum endpoint row weighting, query isolation, the
two-row Newton leaf solution, deterministic replay, malformed/singleton
query refusal, and the explicit CUDA boundary. Ranking margins are raw-link
values; no probability or hidden host fallback is inferred.

```bash
python -B scripts/bench_lightgbm_ranking.py \
  --fortml ../fortml --output results/lightgbm_ranking.csv
```

See [`results/LIGHTGBM_RANKING.md`](results/LIGHTGBM_RANKING.md) for the
fixture, oracle values, provenance, and timing rows.

## Native ordered ordinal GP likelihood

This lane checks the backend-independent ordered-logit and ordered-probit
likelihood value, latent and cut-point JVP/VJP products, and analytic HVP
against an independent NumPy CDF/PDF oracle. It also checks malformed threshold
rollback and the explicit CPU-only capability boundary.

```bash
python -B scripts/bench_gp_ordinal_likelihood.py \
  --fortml ../fortml --output results/gp_ordinal_likelihood.csv
```

See [`results/GP_ORDINAL_LIKELIHOOD.md`](results/GP_ORDINAL_LIKELIHOOD.md) for
the fixture, checksum errors, timings, and pinned provenance.

## MLP trainable parameter state

This lane compares a named dense-MLP freeze block with an independent NumPy
forward/reverse recurrence. It freezes `layer_1.weight`, checks that the
packed deployment value is unchanged and that frozen VJP/JVP coordinates are
zero, then re-enables the block and checks the analytic JVP. The Fortran
behavioral oracle covers unknown-path transactionality. The CUDA row is an
explicit unavailable boundary because resident optimizer routing for this
metadata path is not claimed.

```bash
python -B scripts/bench_mlp_trainable_state.py \
  --fortml ../fortml --output results/mlp_trainable_state.csv \
  --report results/MLP_TRAINABLE_STATE.md
```

See [`results/MLP_TRAINABLE_STATE.md`](results/MLP_TRAINABLE_STATE.md) for
the independent oracle, release output, tolerances, and provenance.

## Exact GP posterior covariance

This lane compares `gp_regression_t%predict_covariance` with an independent
NumPy dense solve for an RBF exact GP. It checks the full latent posterior
matrix, symmetry, and agreement with the marginal variance path. CPU dispatch
is the reference; selected CUDA returns the typed `FORTNUM_NOT_IMPLEMENTED`
boundary until resident covariance and Cholesky kernels are linked.

```bash
python -B scripts/bench_gp_posterior_covariance.py \
  --fortml ../fortml --output results/gp_posterior_covariance.csv \
  --report results/GP_POSTERIOR_COVARIANCE.md
```

See [`results/GP_POSTERIOR_COVARIANCE.md`](results/GP_POSTERIOR_COVARIANCE.md)
for the fixture, oracle checks, timings, and pinned provenance.

## Transactional basis-pipeline cloning

This lane checks the host-side clone/reset seam for a fitted polynomial-plus-
radial basis pipeline. An independent NumPy oracle verifies equal copied
outputs and that changing a clone's radial centre does not mutate the source.
The release test also checks transactional invalid-source behavior, CPU device
dispatch, and a typed CUDA refusal that leaves the destination unchanged.

```bash
python -B scripts/bench_basis_pipeline_clone.py \
  --fortml ../fortml --output results/basis_pipeline_clone.csv
```

See [`results/BASIS_PIPELINE_CLONE.md`](results/BASIS_PIPELINE_CLONE.md) for
the fixture, independent oracle, timing, and device-boundary evidence.

## Polynomial-kernel binary SVM

This lane compares the dense finite-basis polynomial SVM with an independent
SciPy L-BFGS-B weighted squared-hinge solve. It gates sorted arbitrary labels,
degree/gamma/coef0 metadata, score and prediction checks, and the release
timing rows. The FortML behavioral test additionally checks analytic JVP/VJP
products, malformed-input rollback, CPU dispatch, and the typed CUDA refusal.

```bash
python -B scripts/bench_polynomial_svm.py \
  --fortml ../fortml --output results/polynomial_svm.csv
```

See [`results/POLYNOMIAL_SVM.md`](results/POLYNOMIAL_SVM.md) for the fixture,
oracle, tolerances, and pinned provenance.
