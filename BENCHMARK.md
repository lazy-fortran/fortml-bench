# Benchmark lanes

## Stable linear-classifier log probabilities

This lane checks saturation-safe binary log-sigmoid and multinomial
log-softmax values, then runs the independent logistic and softmax JVP/VJP
tests. The fixture includes logits of `+/-1200`, so a separate `log(p)` path
would expose underflow. CUDA is recorded as a typed refusal until resident
classifier reductions are linked.

```bash
python -B scripts/bench_linear_log_proba.py \
  --fortml ../fortml --output results/linear_log_proba.csv \
  --report results/LINEAR_LOG_PROBA.md
```

See [`results/LINEAR_LOG_PROBA.md`](results/LINEAR_LOG_PROBA.md) for the
independent oracle and provenance.

## Trainer validation direction and checkpoint replay

This lane checks the model-agnostic trainer's patience and best-state
restoration for both loss metrics (minimize, the default) and score metrics
(`validation_higher_is_better`). The independent NumPy oracle checks the
known-answer trajectories and the release test checks schema-8 checkpoint
continuation plus the transactional callback-presence refusal.

```bash
python -B scripts/bench_trainer_validation.py \
  --fortml ../fortml --output results/trainer_validation.csv
```

See [`results/TRAINER_VALIDATION.md`](results/TRAINER_VALIDATION.md) for the
fixture and reproducibility details.

## Resumable trainer partial-fit contract

This lane compares one six-update Adam trajectory with two warm-start chunks
(2+4) against an independent NumPy diagonal-quadratic recurrence. The
Fortran behavioral oracle also checks checkpoint continuation, transactional
over-budget requests, and CPU/CUDA dispatch. CUDA is an explicit typed
refusal because the generic trainer has no resident objective or optimizer
state.

```bash
python -B scripts/bench_trainer_partial_fit.py \
  --fortml ../fortml --output results/trainer_partial_fit.csv \
  --report results/TRAINER_PARTIAL_FIT.md
```

See [`results/TRAINER_PARTIAL_FIT.md`](results/TRAINER_PARTIAL_FIT.md) for
the independent oracle, provenance, and refusal contract.

## Generic trainer fit diagnostics

This lane checks the production `trainer_state_t` diagnostics introduced by
the schema-8 checkpoint contract. An independent quadratic oracle gates the
bounded FortOpt L-BFGS-B optimum, while the release app records optimizer
iterations, line-search evaluations, curvature updates, and successful fit
calls. A callback-stopped Adam run checks that streaming optimizers expose zero
for the L-BFGS-B-specific counters. The generic trainer remains host-owned;
CUDA is not claimed without a resident objective and optimizer state.

```bash
python -B scripts/bench_trainer_fit_diagnostics.py \
  --fortml ../fortml --output results/trainer_fit_diagnostics.csv \
  --report results/TRAINER_FIT_DIAGNOSTICS.md
```

See [`results/TRAINER_FIT_DIAGNOSTICS.md`](results/TRAINER_FIT_DIAGNOSTICS.md)
for the independent oracle, diagnostics, and provenance.

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

## Exhaustive small-cardinality categorical XGBoost

This lane checks the `categorical_policy="partition"` CPU path against an
independent NumPy enumeration of every canonical nontrivial subset.  The
four-code fixture verifies the Newton leaf values, selected three-node tree,
prediction checksum, and repeated prediction timing.  The CUDA row remains an
explicit unavailable/typed-refusal boundary because no resident categorical
tree kernel is linked.

```bash
python -B scripts/bench_xgboost_categorical_partition.py \
  --fortml ../fortml --output results/xgboost_categorical_partition.csv
```

See [`results/XGBOOST_CATEGORICAL_PARTITION.md`](results/XGBOOST_CATEGORICAL_PARTITION.md)
for the oracle, refusal contract, and pinned provenance.

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

## Resident CUDA boosted-tree plan

This lane checks a fixed two-tree additive ensemble against an independent
NumPy leaf-walk oracle. It covers base score, learning rate, per-tree scales,
strict split routing, learned NaN defaults, and zero fixed-routing input JVPs.
The ordinary Fortran gate checks invalid-device handling, typed
`FORTNUM_NOT_IMPLEMENTED` refusal, and sentinel preservation. Native CUDA runs
only when `nvcc` and a CUDA device are available; unavailable hardware is
recorded explicitly rather than timed as a host fallback.

```bash
python -B scripts/bench_cuda_boosted_tree.py \
  --fortml ../fortml --output results/cuda_boosted_tree.csv \
  --report results/CUDA_BOOSTED_TREE.md
```

See [`results/CUDA_BOOSTED_TREE.md`](results/CUDA_BOOSTED_TREE.md) for the
independent oracle, resident model contract, boundary behavior, and pinned
provenance.

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

## Exact GP posterior covariance and hyperparameter products

This lane compares `gp_regression_t%predict_covariance` with an independent
NumPy dense solve for an RBF exact GP. It checks the full latent posterior
matrix, symmetry, and agreement with the marginal variance path. It also
checks full-matrix kernel/log-noise JVP and VJP products against independent
NumPy implicit-solve products, including the Frobenius cotangent reduction.
CPU dispatch is the reference; selected CUDA returns the typed
`FORTNUM_NOT_IMPLEMENTED` boundary for value and derivative products until
resident covariance and Cholesky kernels are linked.

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

## Variational multiclass GP log probabilities

This lane checks stable one-vs-rest logistic and probit log probabilities,
row-wise log-sum-exp normalization, packed variational-state JVP/VJP products,
fixed-state input products, and the typed CUDA refusal. An independent NumPy
oracle covers central-difference agreement and an extreme tail that would
underflow in probability space.

```bash
python -B scripts/bench_gp_variational_multiclass_log_proba.py \
  --fortml ../fortml --output results/gp_variational_multiclass_log_proba.csv
```

See [`results/GP_VARIATIONAL_MULTICLASS_LOG_PROBA.md`](results/GP_VARIATIONAL_MULTICLASS_LOG_PROBA.md)
for the oracle, release timing, and provenance.

## Resident CUDA dense MSE training

This lane checks a resident dense affine MSE update with device-resident
parameters, batches, gradients, and optimizer state. The native CUDA oracle
covers SGD, Adam, and AdamW against a NumPy recurrence and compute-sanitizer
checks report zero memory errors. Ordinary builds retain a typed
`FORTNUM_NOT_IMPLEMENTED` refusal rather than falling back to the host.

```bash
python -B scripts/bench_cuda_dense_training.py \
  --fortml ../fortml --output results/cuda_dense_training.csv
```

See [`results/CUDA_DENSE_TRAINING.md`](results/CUDA_DENSE_TRAINING.md) for
native/sanitizer results, transfer counters, and refusal behavior.

## Successive-halving hyperparameter search

This lane checks deterministic seeded candidate generation, multi-fidelity rung
pruning, evaluation-budget accounting, and fixed-resource FortOpt L-BFGS-B
refinement against an independent quadratic oracle. CUDA is a typed refusal
until a resident objective callback ABI is available.

```bash
python -B scripts/bench_hyperparameter_successive_halving.py \
  --fortml ../fortml --output results/hyperparameter_successive_halving.csv
```

See [`results/HYPERPARAMETER_SUCCESSIVE_HALVING.md`](results/HYPERPARAMETER_SUCCESSIVE_HALVING.md)
for the fixture, rung diagnostics, and provenance.

## Ordinal GP log probabilities

This lane checks stable ordered predictive log probabilities, packed
parameter/query JVP/VJP products, and finite-difference/adjoint agreement.
The CPU path is the reference; selected CUDA returns the typed refusal until
the ordinal covariance graph is resident.

```bash
python -B scripts/bench_gp_ordinal_log_proba.py \
  --fortml ../fortml --output results/gp_ordinal_log_proba.csv
```

See [`results/GP_ORDINAL_LOG_PROBA.md`](results/GP_ORDINAL_LOG_PROBA.md) for
the independent oracle and pinned provenance.

## Generic trainer value clipping

This lane compares per-coordinate gradient-value clipping with an independent
NumPy quadratic update and checks the persisted diagnostic counter and schema-8
checkpoint round trip. The generic trainer remains host-owned, so CUDA is a
typed unavailable row.

```bash
python -B scripts/bench_trainer_value_clipping.py \
  --fortml ../fortml --output results/trainer_value_clipping.csv \
  --report results/TRAINER_VALUE_CLIPPING.md
```

See [`results/TRAINER_VALUE_CLIPPING.md`](results/TRAINER_VALUE_CLIPPING.md)
for the exact update and provenance.

## Multi-output tree validation metadata

This lane checks per-output best iteration, validation loss, and early-stop
metadata for the XGBoost-style and LightGBM-style adapters against an
independent two-leaf Newton-stump oracle. The resident tree state is not yet
linked, so CUDA is a typed refusal.

```bash
python -B scripts/bench_xgboost_multioutput_validation_metadata.py \
  --fortml ../fortml \
  --output results/xgboost_multioutput_validation_metadata.csv \
  --report results/XGBOOST_MULTIOUTPUT_VALIDATION_METADATA.md
```

See [`results/XGBOOST_MULTIOUTPUT_VALIDATION_METADATA.md`](results/XGBOOST_MULTIOUTPUT_VALIDATION_METADATA.md)
for the scalar oracle and pinned provenance.

## Five-coordinate mini-batch Adam hypergradients

This lane checks the deterministic coupled-L2 Adam trajectory over
`[log_lr, log_l2, logit_beta1, logit_beta2, log_epsilon]`. An independent affine
replay gates exact derivatives, while a nonlinear fixture checks finite
differences, JVP/VJP duality, FortOpt L-BFGS-B convergence, and the typed CUDA
boundary.

```bash
python -B scripts/bench_mlp_minibatch_adam_hypergradient.py \
  --fortml ../fortml --output results/mlp_minibatch_adam_hypergradient.csv
```

See [`results/MLP_MINIBATCH_ADAM_HYPERGRADIENT.md`](results/MLP_MINIBATCH_ADAM_HYPERGRADIENT.md)
for the 21-row oracle and provenance.

## Fixed-latent ordinal-GP cut-point calibration

This lane checks weighted ordered-probit/logistic cut-point calibration with a
strict location-plus-log-gap transform, transactional FortOpt L-BFGS-B state,
analytic gradient/HVP products, and prediction threshold JVP/VJP products.
CUDA is a typed refusal until the ordinal graph is resident.

```bash
python -B scripts/bench_gp_ordinal_cutpoints.py \
  --fortml ../fortml --output results/gp_ordinal_cutpoints.csv
```

See [`results/GP_ORDINAL_CUTPOINTS.md`](results/GP_ORDINAL_CUTPOINTS.md) for
the independent SciPy/NumPy convergence oracle and provenance.

## Weighted boosted-tree partial dependence and ICE

This lane checks weighted partial-dependence and individual conditional
expectation curves for XGBoost-style and LightGBM-style trees, including
transformed predictions and raw margins. Weight aggregation is overflow-safe;
invalid grids are transactional and CUDA is an explicit refusal.

```bash
python -B scripts/bench_boosted_partial_dependence.py \
  --fortml ../fortml --output results/boosted_partial_dependence.csv
```

See [`results/BOOSTED_PARTIAL_DEPENDENCE.md`](results/BOOSTED_PARTIAL_DEPENDENCE.md)
for the hand-computed/NumPy oracle and provenance.

## Learned basis fan-in

This lane exercises two named same-shape basis branches combined by learned
mixing weights. The independent NumPy fixture checks value, input/parameter
JVP and VJP products. The Fortran gate adds HVP, metadata, transactional
rollback, CPU dispatch, and output-preserving CUDA/OpenACC refusals. The
release run uses the GNU compiler explicitly because the NVFortran runtime
fixture remains an open compatibility boundary.

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
python -B scripts/bench_basis_blend_pipeline.py \
  --fortml ../fortml --output results/basis_blend_pipeline.csv
```

See [`results/BASIS_BLEND_PIPELINE.md`](results/BASIS_BLEND_PIPELINE.md) for
the oracle and timing. The CSV was generated at clean benchmark revision
`7ff48db` and recorded in `48008d4`.

## Fixed-full-batch SGD clipping hypergradients

This lane checks exact active/inactive global-norm clipping sensitivities over
log learning rate, log L2, and log clip norm. The independent NumPy recurrence
is compared with the production MLP path, FortOpt L-BFGS-B, and typed kink,
outer-HVP, and CUDA boundaries.

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
python -B scripts/bench_mlp_clip_hypergradient.py \
  --fortml ../fortml --output results/mlp_clip_hypergradient.csv
```

See [`MLP_CLIP_HYPERGRADIENT.md`](MLP_CLIP_HYPERGRADIENT.md) and
[`results/mlp_clip_hypergradient.csv`](results/mlp_clip_hypergradient.csv)
for the 15-row oracle. The CSV was generated at clean benchmark revision
`8683d98` and recorded in `cb681b3`.

## Weighted Gamma GP likelihood products

This lane checks the positive-target Gamma density over latent log means and a
transformed log-shape coordinate. The independent NumPy/SciPy oracle covers
value, gradient, JVP, VJP, and directional HVP products. A bounded FortOpt
fit is compared with SciPy and CUDA is recorded as a typed refusal.

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
python -B scripts/bench_gp_gamma_likelihood.py \
  --fortml ../fortml --output results/gp_gamma_likelihood.csv \
  --report results/GP_GAMMA_LIKELIHOOD.md
```

See [`results/GP_GAMMA_LIKELIHOOD.md`](results/GP_GAMMA_LIKELIHOOD.md) for
the maximum product error (`1.399e-6`) and fit error (`4.274e-9`). The CSV was
generated at clean benchmark revision `566fbc1` and recorded in `86c4bd9`.

## OVR logistic partial-fit state

This lane checks sorted arbitrary labels, deferred class completion,
transactional malformed-batch rollback, deterministic replay, fixed-state
products, and the explicit CUDA boundary for the shared OVR logistic state.
The Python state machine is independent of the Fortran implementation.

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
python -B scripts/bench_ovr_logistic_partial_fit.py \
  --fortml ../fortml --output results/ovr_logistic_partial_fit.csv \
  --report results/OVR_LOGISTIC_PARTIAL_FIT.md
```

See [`results/OVR_LOGISTIC_PARTIAL_FIT.md`](results/OVR_LOGISTIC_PARTIAL_FIT.md)
for the replay error (zero), behavioral gate, and typed CUDA refusal. The
clean evidence is pinned by benchmark commit `ea13ff1` and FortML revision
`124a9b4`.

## Weighted RMSprop trajectory hypergradients

This lane checks a centered RMSprop trajectory with non-uniform training and
validation weights, exact value/gradient/JVP/VJP products, an affine outer HVP,
FortOpt L-BFGS-B callbacks, and the typed CUDA refusal. An independent NumPy
recurrence is the behavioral oracle.

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
python -B scripts/bench_mlp_rmsprop_weighted_hypergradient.py \
  --fortml ../fortml --output results/mlp_rmsprop_weighted_hypergradient.csv \
  --report results/MLP_RMSPROP_WEIGHTED_HYPERGRADIENT.md
```

See [`results/MLP_RMSPROP_WEIGHTED_HYPERGRADIENT.md`](results/MLP_RMSPROP_WEIGHTED_HYPERGRADIENT.md)
for the maximum oracle error (`3.48e-9`) and CPU timings. The clean evidence
is pinned by benchmark commit `c140039` and FortML revision `124a9b4`.

## ICM multi-output GP likelihood products

This lane assembles a dense intrinsic-coregionalization covariance and checks
likelihood JVP, hyperparameter gradient, and directional HVP products against
independent NumPy central differences. The focused Fortran oracle also checks
all packed coordinates, transactional rollback, metadata, and the typed CUDA
boundary.

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
python -B scripts/bench_multi_output_gp_hypergradients.py \
  --fortml ../fortml --output results/multi_output_gp_hypergradients.csv \
  --report results/MULTI_OUTPUT_GP_HYPERGRADIENTS.md
```

See [`results/MULTI_OUTPUT_GP_HYPERGRADIENTS.md`](results/MULTI_OUTPUT_GP_HYPERGRADIENTS.md)
for the gradient error (`2.20e-8`), HVP error (`3.17e-6`), timings, and typed
CUDA refusal. The clean evidence is pinned by benchmark commit `59a66fc` (the
oracle correction is `ca1bbf1`) and FortML revision `124a9b4`.

## Grouped NDCG ranking metric

This lane checks the standalone tree-ranking reduction for arbitrary positive
query IDs, deterministic score ties, per-query cutoffs, weighted exponential
gains, and the typed CUDA boundary. A Python DCG reduction is the independent
behavioral oracle.

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
python -B scripts/bench_ranking_metrics.py \
  --fortml ../fortml --output results/ranking_metrics.csv
```

See [`results/RANKING_METRICS.md`](results/RANKING_METRICS.md) for the
matching value (`0.8295009024012067`), zero maximum error, CPU timing, and
CUDA status 3 refusal. The clean evidence is pinned by benchmark commit
`70f25ab` and FortML revision `dfa3a92`.

## Laplace GP-classifier implicit prediction JVP

This lane differentiates latent means, latent variances, and logistic or probit
probabilities through the converged weighted Laplace mode. Independent NumPy
refits and central parameter differences certify all 18 output tangents. The
CUDA path is an explicit typed refusal because the implicit Laplace graph is
not resident.

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
python -B scripts/bench_gp_classification_implicit_prediction.py \
  --fortml ../fortml --output results/gp_classification_implicit_prediction.csv
```

See [`results/GP_CLASSIFICATION_IMPLICIT_PREDICTION.md`](results/GP_CLASSIFICATION_IMPLICIT_PREDICTION.md)
for the maximum logistic error (`3.58e-10`), probit error (`9.61e-10`), CPU
timings, and CUDA refusal. The clean evidence is pinned by benchmark commit
`0327e94` and FortML revision `9c1dbdf`.

## MLP optimizer-group checkpoint identity

This lane checks that named optimizer groups survive formatted checkpoint
round-trips and that resume rejects identity drift even when ranges and
multipliers match. The CUDA row records the typed refusal for the non-resident
grouped hypergradient path.

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
python -B scripts/bench_mlp_optimizer_group_registry.py \
  --fortml ../fortml --output results/mlp_optimizer_group_registry.csv \
  --report results/MLP_OPTIMIZER_GROUP_REGISTRY.md
```

See [`results/MLP_OPTIMIZER_GROUP_REGISTRY.md`](results/MLP_OPTIMIZER_GROUP_REGISTRY.md)
for the round-trip, name-drift, and device rows. The clean evidence is pinned
by benchmark commit `6874dc9` and FortML revision `c50511c`.

## Resident numeric XGBoost dispatch

This lane routes numeric fixed-topology XGBoost trees through the resident
CUDA plan when native support is available. It checks a NumPy leaf-walk oracle,
CPU dispatch parity, learned missing-value routing, split-boundary behavior,
and the explicit categorical/unavailable refusal. Timing rows separate the
ordinary plan gate from the native resident execution.

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
python -B scripts/bench_cuda_boosted_tree.py \
  --fortml ../fortml --output results/cuda_boosted_tree.csv \
  --report results/CUDA_BOOSTED_TREE.md
```

See [`results/CUDA_BOOSTED_TREE.md`](results/CUDA_BOOSTED_TREE.md) for the
zero-error CPU oracle, native CUDA row, and typed boundary. The clean evidence
is pinned by benchmark commit `186d186` and FortML revision `c50511c`.

## Versioned result schema

Release CSVs use the required provenance and correctness fields described in
[`RESULT_SCHEMA.md`](RESULT_SCHEMA.md). Validate the latest lanes with:

```bash
python -B scripts/validate_result_schema.py --all
```

The `--all` audit reports historical pre-v1 rows as migrations. Release rows
must pass without `--allow-dirty`.
