# fortml-bench roadmap

This repository owns reproducible cross-engine evidence for fortml. A result
requires an independent oracle, matched mathematical work, recorded toolchain
metadata, and a committed raw record.

## Status and handoff

Last updated 2026-08-09. The scalable-GP study of Liu et al. (IEEE TNNLS <!-- slop-ok -->
31(11):4405-4423, 2020) is complete and is the centrepiece of the GP evidence
here. Read `results/SCALABLE_GP.md` first: it states what "matching the paper"
can mean (the review publishes no numeric result tables), which half of each
result is a reproduced behaviour and which is a measured complexity order, and
the conclusion at n = 131,072.

Headline: on the review's 1-D fixture at 131,072 points the exact matrix-free
solve on a GPU takes 374 s against FITC's 0.708 s for the same accuracy
(6.99e-5 against 6.76e-5). A dense exact GP was not attempted at that size:
the preflight allocation is about 137 GB. SKI with its grid scaled to `n/8`
matches that accuracy in 19 MiB.

The completion artifacts are:

- [`results/MODEL_WORKLOADS.md`](results/MODEL_WORKLOADS.md), with raw
  gfortran, nvfortran, PyTorch, and GPyTorch rows in
  [`model_workloads.csv`](results/model_workloads.csv) and separate exact-GP
  and MLP plots.
- [`results/GP_FEATURES.md`](results/GP_FEATURES.md), with stochastic spectral,
  derivative, multi-output, and variational rows in
  [`gp_features.csv`](results/gp_features.csv) and its comparison plot.
- [`results/SCALABLE_GP.md`](results/SCALABLE_GP.md), including corrected GRBCM
  evidence in
  [`scalable_gp_grbcm_corrected.csv`](results/scalable_gp_grbcm_corrected.csv),
  contiguous/clustered evidence in
  [`scalable_gp_clustered.csv`](results/scalable_gp_clustered.csv), and current
  multidimensional-SKI evidence in
  [`scalable_gp_dimension_current.csv`](results/scalable_gp_dimension_current.csv).

The corrected GRBCM record replaces every older GRBCM claim. The partition
record covers all seven local methods over five expert counts. The SKI record
contains valid `d = 1, 2, 4` rows and an explicit grid-budget refusal at
`d = 8`. It does not hide that configuration as a missing measurement.

The 30-percent gate remains workload-specific. Among the new model calls,
exact-GP prediction and the MLP VJP pass. Exact-GP fitting and MLP forward do
not. Among the new GP-feature calls, multi-output and variational calls pass.
Log determinant, predictive variance, and the derivative call do not. The
individual ratios and timed-call definitions are in the two reports above.

Two traps that produced wrong numbers in this repository and will again:

* The default `fo` profile is `-O0 -fcheck=all`. Any timing taken without
  `--flag "-O3 -funroll-loops"` is a debug-build timing. The first scalable-GP <!-- slop-ok -->
  sweep was one and had to be discarded.
* `fo` shares `build/fo/bin` between compilers, so a device run can execute a
  host binary. Rebuild immediately before a device measurement and let nothing
  else build in between. `scripts/` does this.

Reproduce:

```sh
python3 scripts/bench_scalable_gp.py --output results/scalable_gp_large_current.csv \
    --sweep samples --values 8192 16384 32768 65536 131072 \
    --repetitions 1 --expert-size 1024 --threads 1
python3 scripts/plot_scalable_gp.py --input results/scalable_gp_large_current.csv \
    --prefix results/scalable_gp_large_current --metric train_seconds
bash scripts/fetch_reference_implementations.sh
```

## Research directions

A useful follow-on target would have structure that a rank-64 inducing summary
cannot capture. That would separate the exact and approximate lanes and test
whether the 131k conclusion generalizes beyond the current fixture. This is a
research extension, not unfinished work in the present benchmark scope.

The checklist below is complete for the defined scope. Research directions are
not completion gates.

## Workloads

- [x] Complete matched RBF MVM runs on CPU and GPU for Fortran, dense PyTorch,
  KeOps, and GPyTorch with KeOps.
- [x] Add size sweeps that report runtime, the dense OOM boundary, and
  CPU/GPU scaling plots.
- [x] Add matched matrix-free CG solves with the same float64 tolerance,
  iteration cap, diagonal shift, unpreconditioned recurrence, and true-residual
  stopping check for dense PyTorch, KeOps, GPyTorch-KeOps, and nvfortran
  FortML.
- [x] Add stochastic log-determinant and predictive-variance products.
- [x] Add exact small-GP training and prediction comparisons.
- [x] Add a pinned exact-GP reference lane against scikit-learn and optional
  GPyTorch. The shared closed-form fixture checks predictive mean and variance
  before timing fit and prediction; unavailable optional dependencies are
  recorded explicitly in `fixtures/gp_vs_reference.json`, with the protocol
  in `results/GP_REFERENCE.md`.
- [x] Add regular-grid Toeplitz/Kronecker evidence with independent dense or
  structured oracles and resident OpenACC scaling records.
- [x] Add compact-support sparse workloads using `fortsparse`, with stored
  nonzero/storage diagnostics, independent row-wise oracles, and matched
  float64 CPU/GPU measurements against dense PyTorch and KeOps.
- [x] Add a KeOps-style static RBF-plus-constant matrix-free lane with a
  blocked independent oracle, native CUDA option, and high-N CPU/GPU scaling
  evidence against KeOps, GPyTorch-KeOps, and dense PyTorch.
- [x] Add derivative-observation and derivative-prediction workloads.
- [x] Add the locally-periodic mixed-observation derivative-GP lane. An
  independent dense NumPy Cholesky oracle checks value/first-derivative
  covariance blocks, parameter JVPs, posterior moments, and query-input JVPs,
  including a coincident query row. The raw record in
  `results/derivative_gp_local_periodic.csv` retains the explicit CUDA
  refusal; the protocol is documented in
  `results/DERIVATIVE_GP_LOCAL_PERIODIC.md`.
- [x] Add multi-output and variational GP workloads.
- [x] Add the batched multi-output GP shape contract. The independent NumPy
  oracle assembles output-major intrinsic-coregionalization covariance and
  checks batched posterior means, input JVPs, and VJP scalar duality. The raw
  record is `results/multi_output_gp_batch.csv`; CPU shape/device gates and an
  explicit resident-CUDA refusal are documented in
  [`results/MULTI_OUTPUT_GP_BATCH.md`](results/MULTI_OUTPUT_GP_BATCH.md).
- [x] Add exact multi-output prior-covariance parameter JVP/VJP products over
  kernel/log-noise/coregionalization coordinates. The independent NumPy
  Kronecker derivative oracle checks every packed direction, the release app
  checks full-cotangent adjoint duality, and the raw record retains an explicit
  CUDA refusal in `results/multi_output_gp_products.csv`.
- [x] Add periodic and rational-quadratic kernel value, input-derivative, and
  logarithmic parameter JVP/VJP/HVP checks with release CPU timings and
  explicit CUDA capability-refusal rows. See
  [`results/KERNEL_CATALOG.md`](results/KERNEL_CATALOG.md).
- [x] Add mixed value/first-derivative periodic and rational-quadratic GP
  query-input JVP/VJP workloads. The independent covariance/posterior oracle
  gates every CPU row; the resident derivative-GP CUDA graph remains an
  explicit typed refusal. See [`results/DERIVATIVE_GP.md`](results/DERIVATIVE_GP.md).
- [x] Add dense derivative-GP posterior covariance parameter JVP/VJP
  workloads for periodic, rational-quadratic, cosine, and polynomial kernels.
  The benchmark finite-differences every packed log-kernel/log-noise coordinate
  through an independent NumPy Cholesky oracle, checks the release app's
  exact CPU products, and records typed CUDA refusal rows.
- [x] Extend the derivative-GP benchmark to the GPyTorch-compatible
  spectral-mixture kernel. The independent dense oracle covers mixed
  value/first-derivative blocks, query JVP/VJP, posterior covariance, and
  packed parameter JVP/VJP products; the release app records exact CPU rows
  and the resident CUDA graph remains an explicit typed refusal.
- [x] Add the polynomial mixed-observation hyperparameter HVP lane. The
  independent NumPy likelihood-gradient finite difference checks all four
  packed log-kernel coordinates and log noise, including the degree-one limit;
  the release record has CPU timings and an explicit resident-CUDA refusal.
- [x] Add the RBF mixed-observation order-three lane. The independent dense
  NumPy oracle covers derivative covariance blocks through total order six,
  posterior moments, query input JVP/VJP products, and the packed likelihood
  HVP. The correctness-gated release app records CPU rows and a typed CUDA
  refusal in `results/second_derivative_gp_rbf_order3.csv`; the protocol is
  documented in [`results/SECOND_DERIVATIVE_GP_RBF_ORDER3.md`](results/SECOND_DERIVATIVE_GP_RBF_ORDER3.md).

## Evidence

- [x] Record gfortran and nvfortran compiler reports for the Fortran kernels.
- [x] Keep the standalone nvfortran RBF benchmark synchronized with the
  FortAD-generated RBF and Matérn product modules and FortSym-generated primal
  leaf. The direct CPU/GPU builds and independent pairwise oracle pass on the
  cluster.
- [x] Record PyTorch, GPyTorch, KeOps, CUDA, driver, and GPU revisions.
- [x] Generate comparison plots from committed CSV data.
- [x] Upload the first released plot to Slopbox and record its URL.
- [x] Establish the within-30-percent comparison against the best matched
  competitor for the first RBF MVM workload, precision, and device.
- [x] Record the composite lane through the high-N slope regime and retain
  dense PyTorch capacity failures as explicit non-timing rows.
- [x] Publish a machine-readable CSV record and reproducibility script.

The within-30-percent target is a measurement gate. It is never inferred from
a different workload, precision, device, or residency policy.

## Classification and scientific-ML extensions

- [x] Add a deterministic binary logistic workload for FortML and
  scikit-learn. The fixture stores arbitrary integer labels, the NumPy-generated
  score labels, probability normalization, accuracy, fit time, and prediction
  time in `results/classification_workloads.csv`.
- [x] Add positive binary temperature scaling beside Platt sigmoid and weighted
  PAVA isotonic calibration. The independent NumPy inverse-temperature Newton
  oracle checks all 256 labels, predictions, and probability columns before
  retaining CPU timings in `results/probability_calibration.csv`; all three
  CUDA capability rows remain explicit refusals until a resident kernel exists.
- [x] Add a multiclass softmax-temperature calibration lane with sorted integer
  classes, stable complete probability/prediction arrays, fitted-temperature
  agreement against an independent NumPy weighted-NLL Newton oracle, and an
  explicit CUDA refusal. The raw record is
  `results/multiclass_probability_calibration.csv`; Platt/isotonic multiclass
  policies remain typed unsupported contracts.
- [x] Add a deterministic MLP-training lane with an independent NumPy Adam
  oracle, full loss/prediction checks, and release-build fit timings.
- [x] Add release-build momentum-SGD and Nesterov MLP-training workloads with
  independent NumPy recurrences, complete prediction/loss arrays, and strict
  fit timing records.
- [x] Add a release-build AdamW MLP-training workload with an independent
  NumPy first/second-moment and decoupled-weight-decay oracle.  The raw record
  is `results/adamw_training.csv`; an absent FortML release app remains an
  explicit `unavailable` target row.
- [x] Add an independent five-parameter AdamW beta-logit hypergradient lane
  covering the validation value, all five packed components, and a directional
  JVP. The raw record is `results/adamw_beta_hypergradient.csv`; its strict
  complete-array release app now passes the NumPy oracle.
- [x] Add a strict typed MLP schedule lane covering constant, linear warm-up,
  cosine, warm-up-plus-cosine, exponential decay, and one-cycle schedules. The
  independent NumPy formulas central-difference every used continuous schedule
  product, including one-cycle peak/final-rate tangents; the FortML release app
  emits all 180 values before its resident scalar timing is retained in
  `results/mlp_schedules.csv`. This gate does not claim neural training,
  OpenACC, or CUDA performance.
- [x] Add a correctness-gated one-cycle schedule trajectory hypergradient
  workload. The independent affine NumPy replay checks value, all four
  logarithmic base/L2/peak/final derivatives, and a directional JVP before
  retaining CPU timing; CUDA and outer-HVP rows are explicit capability
  refusals in `results/mlp_one_cycle_hypergradient.csv`.
- [x] Add a correctness-gated metric-aware plateau schedule lane. Independent
  Python transitions cover minimizing and maximizing metrics, patience and
  `min_delta` boundaries, compounded reductions, exact base/factor products,
  and discrete comparison-state zeros. The FortML app emits 132 values for
  each mode before timing, and the CSV records two explicit CUDA capability
  refusals in `results/mlp_plateau_schedule.csv`; the protocol is documented
  in `results/MLP_PLATEAU_SCHEDULE.md`.
- [x] Add the metric-aware plateau trainer lane. An independent NumPy
  recurrence checks the final learning rate and update count, while the
  Fortran trainer test also covers minimizing/maximizing transitions,
  malformed state, checkpoint round trips, diagnostics, and interrupted versus
  uninterrupted trajectories. `results/mlp_plateau_training.csv` records
  zero oracle error and an explicit typed CUDA-unavailable row; the protocol is
  documented in `results/MLP_PLATEAU_TRAINING.md`.
- [x] Add a dense MLP activation lane for linear, `tanh`, ReLU, GELU, SiLU,
  ELU, softplus, leaky ReLU, sigmoid, and Mish. Ten independent NumPy checksum
  rows and ten FortML CPU timings are retained alongside ten explicit CUDA
  refusal rows in `results/mlp_activations.csv`; no host activation timing is
  relabeled as device evidence.
- [x] Add an independent Adagrad accumulated-square and checkpoint/resume
  recurrence lane. The raw record is `results/adagrad.csv`; the missing
  dedicated FortML release app remains an explicit `unavailable` row rather
  than borrowing an Adam/AdamW timing.
- [x] Add independent FortOpt RMSprop and centered/momentum MLP-training
  recurrences with release-app rows for both paths. The raw record is
  `results/rmsprop.csv`; direct parameter norms and final MLP losses are
  checked before timing.
- [x] Add deterministic dense k-nearest-neighbor classification with sorted
  arbitrary integer labels, stable distance ties, uniform and inverse-distance
  votes, and complete probability/prediction checksums. The raw record is
  `results/knn.csv`; discrete input JVP/VJP refusals remain explicit.
- [x] Add dense scalar closed-radius nearest-neighbor regression with uniform
  or inverse-distance sample-weighted averaging and an explicit
  empty-neighborhood value. The independent NumPy prediction oracle and typed
  CUDA refusal are defined by `scripts/bench_radius_neighbors_regression.py`
  and documented in `results/RADIUS_NEIGHBORS_REGRESSION.md`; the CSV is
  generated only after the FortML release app builds successfully.
- [x] Add dense multi-output closed-radius nearest-neighbor regression with
  shared uniform neighborhoods, vector-valued outliers, complete prediction
  arrays, and an independent NumPy oracle. The raw record is
  `results/radius_neighbors_multioutput.csv`; CUDA remains a typed unavailable
  row until a resident radius-search reduction is linked. The protocol is
  documented in `results/RADIUS_NEIGHBORS_MULTIOUTPUT.md`.
- [x] Add weighted primal linear SVM classification with arbitrary binary
  integer labels, feature-only L2 regularization, complete signed-margin
  checks, and a typed CUDA refusal. The raw record is
  `results/linear_svm.csv`; kernel, one-class, ranking, and SVR variants remain
  separate work packages.
- [x] Add the dense finite-basis RBF-SVM workload with weighted squared-hinge
  SciPy/NumPy score and label checks, arbitrary class ordering, and explicit
  CUDA capability refusal. Its FortML behavioral gate also checks fixed-state
  input/parameter JVP/VJP products, CPU dispatch, and typed CUDA derivative
  refusals. The raw record is `results/rbf_svm.csv`.
- [x] Add a bounded dense RBF one-class SVM correctness lane with an independent
  NumPy capped-simplex dual/score/label oracle and CPU oracle fit/predict
  timings. The release app is checked against every support weight, offset,
  score, and label before retaining FortML CPU timings; typed CUDA refusal is
  recorded explicitly in `results/one_class_svm.csv`. Active-set/hyperparameter
  derivatives and resident CUDA remain separate FortML work packages.
- [x] Add independent BCE, stable softmax/log-softmax, weighted softmax
  cross-entropy, multiclass focal-softmax, weighted-MSE, Huber, and focal BCE
  value/JVP/HVP workloads, including the weighted-MSE MLP objective path.
  NumPy value, central-difference, and adjoint checks cover the new rows;
  exact Huber kinks, true-class focal underflow, and resident CUDA loss/MLP
  kernels remain explicit refusal boundaries. The raw record is
  `results/neural_losses.csv`.
- [x] Add a correctness-gated pairwise contrastive metric-learning benchmark.
  The independent NumPy Euclidean replay checks weighted value, JVP, VJP, and
  HVP checksums for paired embeddings before retaining CPU timings; zero-
  distance/margin derivative boundaries and the typed CUDA value refusal are
  recorded in `results/contrastive_loss.csv` and
  `results/CONTRASTIVE_LOSS.md`.
- [x] Add a dense multilabel-indicator logistic lane with independent
  per-output Newton oracle checks, complete positive-probability and hard
  indicator outputs, contextual scikit-learn timing, and explicit CUDA
  capability-refusal rows. Sparse targets, ordinal outcomes, and resident
  multi-head CUDA kernels remain separate work packages.
- [x] Add a sequential classifier-chain logistic lane with arbitrary integer
  labels, observed-label training features, smooth probability-chain
  prediction, and an independent NumPy replay of the packed fitted heads. The
  raw record is `results/classifier_chain.csv`; resident CUDA remains an
  explicit typed refusal.
- [x] Add a fixed full-batch MLP SGD hypergradient workload over log learning
  rate and log L2, including a central finite-difference value/gradient/JVP
  oracle and explicit FortML release-target refusal rows.  The raw record is
  `results/mlp_hypergradient.csv`; Adam/momentum/CUDA trajectory derivatives
  are outside this contract.
- [x] Add a fixed full-batch RMSprop hypergradient workload over log learning
  rate, log L2, decay, log epsilon, and momentum. The independent NumPy lane
  central-differences every packed component and a directional JVP for both
  centered and uncentered state, while the FortML release app emits a checked
  value/gradient/JVP timing. The raw record is
  `results/rmsprop_hypergradient.csv`; CUDA remains an explicit refusal until
  optimizer state and MLP HVPs are resident.
- [x] Add a fixed full-batch Adagrad hypergradient workload over log learning
  rate, log L2, and log epsilon. The independent NumPy lane central-differences
  every packed component and a directional JVP for the accumulated-square
  trajectory; the FortML release app emits a checked complete-array CPU timing.
  The raw record is `results/adagrad_hypergradient.csv`, with compiler and
  revision provenance plus typed CUDA refusals until resident state derivatives
  are linked.
- [x] Add a fixed full-batch SGD momentum hypergradient workload over log
  learning rate, log regularization, and momentum. The independent NumPy
  trajectory checks value, all three gradient components, and a directional JVP
  before retaining the FortML timing; the raw record is
  `results/sgd_momentum_hypergradient.csv`, with compiler/source provenance and
  typed CUDA refusals.
- [x] Extend the fixed full-batch SGD momentum lane with exact outer HVP
  products on the one-layer affine branch. A nested central-difference NumPy
  oracle checks all three mixed hyperparameter components, and the release app
  checks value/gradient/JVP/HVP arrays before timing both CPU products. The raw
  record is `results/sgd_momentum_hypergradient_hvp.csv`; nonlinear/multilayer
  and CUDA paths remain explicit typed refusals.
- [x] Add a fixed seeded mini-batch SGD hypergradient workload over log
  learning rate and log regularization. The independent NumPy lane reproduces
  the Park–Miller shuffle cursor and checks value, both gradient components,
  and a directional JVP before retaining FortML timing; CUDA remains an
  explicit refusal until resident batch-cursor derivatives exist.
- [x] Add a fixed seeded mini-batch coupled-L2 Adam hypergradient workload
  over log learning rate and log regularization. The independent NumPy lane
  replays first/second moments and bias correction, checks the complete
  value/gradient/JVP array before retaining FortML timing, and records typed
  CUDA refusals. The raw record is
  `results/mlp_minibatch_adam_hypergradient.csv`; resident Adam trajectory
  derivatives remain an explicit follow-up contract.
- [x] Add a fixed full-batch Lion hypergradient workload over log learning
  rate, log regularization, and beta logits. The independent NumPy lane
  central-differences all four packed coordinates and a directional JVP away
  from sign boundaries; the FortML release app must match the complete array
  before its CPU timing is retained. Near-zero sign branches and CUDA remain
  explicit typed refusals. The raw record is
  `results/mlp_lion_hypergradient.csv`.
- [x] Add the optimizer-group trajectory workload over log learning rate, log
  regularization, and one log multiplier per contiguous group. Its independent
  NumPy oracle checks post-SGD scaling, all four gradients, and a directional
  JVP before retaining FortML timing; overlap validation and CUDA are explicit
  refusals. The raw record is
  `results/mlp_optimizer_group_hypergradient.csv`.
- [x] Add a calibrated neural classifier lane with an independent fixture,
  sorted-class, probability-simplex, finite-bound, and prediction-domain
  oracle. The raw record is `results/mlp_calibrated_classifier.csv`; CPU fit
  and prediction timings are retained only after the contract passes, while
  CUDA remains an explicit refusal.
- [x] Add mean, median, and constant differentiable simple-imputer workloads
  with independent statistic, transform, JVP, and VJP checks. Host-only device
  boundaries are retained as explicit capability notes.
- [x] Add dense missing-indicator workloads for both `all` and `missing-only`
  policies, with complete transform and zero JVP/VJP array checks, release CPU
  timings, independent NumPy NaN-mask oracles, and explicit CUDA refusal rows.
  Sparse CSR/CSC views and resident indicator kernels remain separate work.
- [x] Add a sparse-safe CSC standard-scaler lane with implicit-zero means and
  population scales, complete transform/inverse/JVP/VJP value checks against
  an independent dense NumPy expansion, release CPU timings, and a typed CUDA
  refusal row. The raw record is `results/sparse_preprocessing.csv` and the
  protocol is documented in `results/SPARSE_PREPROCESSING.md`; CSR conversion,
  sparse categorical/indicator views, and resident kernels remain open.
- [x] Add a composable polynomial/Fourier basis-pipeline lane with independent
  transform, JVP, and VJP checks and timings.
- [x] Add the differentiable Chebyshev first-kind basis lane. The NumPy
  recurrence independently checks value, input JVP, input VJP, and scalar
  contraction HVP reductions for a 4,096-by-3 degree-eight fixture; the raw
  rows are `results/chebyshev_basis.csv`, with CPU timings and a typed CUDA
  refusal documented in `results/CHEBYSHEV_BASIS.md`.
- [x] Add the dense pipeline input-schema lane. The independent names/count
  oracle checks matching validation plus transactional duplicate/mismatch
  refusal, while the Fortran app records repeated CPU validation and an
  explicit CUDA metadata refusal in `results/pipeline_schema.csv` and
  `results/PIPELINE_SCHEMA.md`.
- [x] Add the typed device-dispatch lane for the column-selecting basis union.
  CPU transform/JVP/VJP/HVP calls are compared with the host products and an
  independent NumPy feature oracle; CUDA remains an explicit refusal with
  sentinel-preservation checks in `results/column_pipeline_device.csv`.
- [x] Add the fitted basis-linear regression composition lane with an
  independent NumPy least-squares and chained JVP oracle.
- [x] Add deterministic CART-stump and residual-stump boosting lanes with an
  exhaustive NumPy split oracle and matched scikit-learn depth-1 reference.
- [x] Record explicit host-only/CUDA refusal rows for GaussianNB, the MLP
  trainer, the logistic objective, and the classifier release app. A missing
  FortML device path is never replaced with a relabeled CPU timing. Resident
  PyTorch CUDA rows remain independent evidence.
- [x] Add resident CUDA correctness gates for kNN prediction, the direct
  no-autodiff RMSprop state kernel, and dense-affine value/JVP/VJP across all eight
  MLP activations. The independent NumPy fixtures and native gate results are
  recorded in `results/device_contracts.csv`; this lane has no timing claim.
- [x] Extend the resident dense-affine gate with one full-batch tanh MSE update,
  an independent NumPy loss/gradient/parameter oracle, and transfer/residency
  counter checks. The same CSV records the MSE-update row. A missing CUDA
  toolchain or device remains an explicit skipped row.
- [x] Add a resident CUDA AdamW state correctness gate with an independent
  NumPy seven-step moment/bias-correction/decoupled-decay oracle. The raw
  record is `results/cuda_adamw.csv`; compile-inclusive gate wall time is not
  presented as resident kernel performance.
- [x] Add a resident CUDA Adagrad state correctness gate with an independent
  NumPy eight-step accumulated-square recurrence oracle. The raw record is
  `results/cuda_adagrad.csv`; compile-inclusive gate wall time is not presented
  as resident kernel performance, and a missing CUDA toolchain/device remains
  an explicit `unavailable` row.
- [x] Add the joint basis-pipeline training correctness lane. The raw record is
  `results/basis_pipeline_training.csv`, with independent value/JVP/HVP and
  CUDA-refusal checks documented in `results/BASIS_PIPELINE_TRAINING.md`.
- [x] Add the named basis fan-out/fan-in DAG correctness lane. The raw record is
  `results/basis_fanout_pipeline.csv`; `test_basis_fanout_pipeline` supplies
  independent feature, metadata, JVP/VJP, HVP, CPU-dispatch, and typed-CUDA
  refusal checks. Residual, conditional, cyclic, and resident branch execution
  remain separate open contracts.
- [x] Add the named residual-sum basis DAG correctness lane. The raw record is
  `results/basis_residual_pipeline.csv`; `test_basis_residual_pipeline` and
  the independent NumPy fixture check branch-summed value/JVP/VJP/HVP products,
  metadata, and output-preserving typed CUDA refusals. Conditional, cyclic,
  and resident branch execution remain separate open contracts.
- [x] Add the fitted horizontal basis-pipeline persistence lane. The raw record
  is `results/pipeline_persistence.csv`; an independent NumPy reconstruction
  checks feature and metadata counts before the Fortran release app's
  versioned text save/load round trip. The source test additionally checks
  value/JVP/VJP/HVP equivalence, names, offsets, malformed-input transactionality,
  and the typed CUDA serialization refusal. Sequential/residual/linear-estimator
  schemas and resident-device checkpoints remain separate contracts.
- [x] Add the production Lion trainer lane. The independent NumPy recurrence
  checks decoupled weight decay, clipping, EMA, validation, and uninterrupted
  versus checkpointed text-resume trajectories. The raw record is
  `results/lion_training.csv`; resident Lion state remains an explicit CUDA
  unavailable row.
- [x] Add correctness-gated records for the model-agnostic objective trainer
  and XGBoost additive tree contributions. The raw record is
  `results/training_core.csv`; the rows are correctness wall times, not
  throughput claims, and the independent NumPy/Fortran oracles remain explicit.
- [x] Add fitted XGBoost ensemble prefix slicing. The independent NumPy staged
  and full-ensemble replay checks prefix predictions, preserves the fitted
  objective/routing state, and records the zero-length-prefix refusal in
  `results/xgboost_slice.csv` and `results/XGBOOST_SLICE.md`.
- [x] Add the generic trainer portable text checkpoint/resume lane. Independent
  NumPy Adam and Lion state continuation oracles are paired with the
  `test_trainer` malformed/truncated/extra-record gate in
  `results/trainer_checkpoint.csv`; host-resident state has an explicit CUDA
  refusal row for both optimizer workloads.
- [x] Add the deterministic unfactored Adafactor trainer/MLP lane. The
  independent NumPy squared-gradient/update-RMS oracle is paired with
  `test_trainer` and `test_mlp_adafactor` recurrence/checkpoint gates in
  `results/adafactor.csv`; the packed-vector API does not claim matrix-factored
  state, and the CPU-only CUDA boundary remains an explicit unavailable row.
- [x] Add layout-aware matrix-factored Adafactor checkpoint migration. The
  independent row/column/vector NumPy recurrence is paired with native and
  formatted schema-8 continuation gates in `results/adafactor_factored.csv`;
  resident CUDA remains an explicit unavailable row. The protocol is
  documented in `results/ADAFACTOR_FACTORED.md`.
- [x] Add the deterministic AMSGrad trainer/MLP lane. The independent NumPy
  bias-corrected max-second-moment oracle is paired with `test_mlp_amsgrad`'s
  in-memory and formatted checkpoint continuation gates in `results/amsgrad.csv`;
  CPU rows compare complete parameter norms and MLP losses, while resident
  AMSGrad CUDA state remains an explicit unavailable row.
- [x] Add the deterministic RAdam trainer/MLP lane. The independent NumPy
  rho-threshold recurrence oracle is paired with `test_mlp_radam`'s exact
  in-memory and formatted checkpoint continuation gate in `results/radam.csv`;
  CPU rows compare complete parameter norms and MLP losses, while resident
  RAdam CUDA state remains an explicit unavailable row. Optimizer-trajectory
  hypergradients and FortOpt RAdam adapters remain open.
- [x] Add the exact fixed full-batch unfactored Adafactor trajectory
  hypergradient lane. The independent NumPy recurrence checks the objective,
  all five packed hyperparameter derivatives, JVP, VJP, bounded L-BFGS-B path,
  and the typed CUDA refusal in `results/adafactor_hypergradient.csv` and
  `results/ADAFACTOR_HYPERGRADIENT.md`.
- [x] Add the exact fixed full-batch AMSGrad trajectory hypergradient lane.
  The independent NumPy recurrence checks validation value, all five packed
  log/logit derivatives, a directional JVP, and the max-second-moment active
  set. The release app is retained only after the complete array agrees, and
  `results/amsgrad_hypergradient.csv` records CPU timing plus typed max-tie,
  denominator, and CUDA refusals. The protocol is documented in
  `results/AMSGRAD_HYPERGRADIENT.md`.
- [x] Add the weighted binary MLP objective adapter and bounded L-BFGS-B lane.
  The independent value/JVP/HVP finite-difference oracle and FortOpt contract
  are recorded in `results/mlp_binary_objective.csv`; no resident CUDA graph is
  implied.
- [x] Add the weighted multiclass MLP objective adapter and bounded L-BFGS-B
  lane. The independent softmax value/JVP/HVP oracle and FortOpt contract are
  recorded in `results/mlp_classifier_objective.csv`; no resident CUDA graph
  is implied.
- [x] Add fixed-input packed-parameter probability JVP/VJP products for the
  multiclass MLP classifier. The independent tanh/softmax replay checks all 21
  packed coordinates, central differences, and tangent/cotangent duality; the
  raw record is `results/mlp_classifier_parameter_products.csv` with a typed
  resident-CUDA refusal.
- [x] Add the composable physics-residual objective lane. Four weighted
  affine residual slots and a nonlinear reverse-over-forward HVP are checked
  with independent value/gradient/JVP/VJP products; providers without an HVP
  callback retain the typed refusal, and the callback-based CUDA boundary
  remains explicit in `results/physics_objective.csv`.
- [x] Add the bounded PINN training-adapter lane. The independent
  manufactured four-slot fixture checks value/gradient/JVP/VJP, nonlinear HVP,
  FortOpt L-BFGS-B fitting, malformed shapes, and typed CUDA refusal in
  `results/pinn.csv`.
- [x] Add the general nonseparable Hamiltonian lane. An independent analytic
  canonical-field/Jacobian oracle is paired with the FortML full-state
  derivative, adjoint, separable-symplectic, and typed split-integrator-refusal
  gate in `results/hamiltonian_general.csv`; CUDA/OpenACC remains explicitly
  unavailable until a resident implicit-integrator graph exists.
- [x] Add the canonical symplectic-form residual lane. An independent
  velocity-Verlet Jacobian oracle checks the defect, value/JVP/VJP products,
  reusable physics-constraint bridge, and typed CUDA refusal in
  `results/symplectic_residual.csv`.
- [ ] Add resident CUDA/OpenACC timing rows for kNN search, RMSprop
  optimizer/trainer state, AdamW trainer state, staged XGBoost diagnostics, and GP classification
  hyperparameter training. Native CUDA kNN and a direct RMSprop state-kernel
  oracle now exist in the FortML checkout, but these benchmark rows still need
  matched transfer-inclusive and resident timings. Until those rows pass, CPU
  timings remain provisional and device refusals must stay explicit.
- [x] Add fitted standard/min-max scaler, binary Laplace GP logistic/probit,
  and one-vs-rest multiclass Laplace-GP lanes with independent NumPy transform
  and Newton posterior oracles. The raw record is
  `results/classification_extensions.csv`.
- [x] Add binary Laplace-GP log-probability products. The independent NumPy
  fixture checks stable log/probability round trips and central-difference
  input/parameter products, while the release app checks the public contract
  and typed CUDA refusal. The raw record is
  `results/gp_classification_log_proba.csv`, with protocol in
  `results/GP_CLASSIFICATION_LOG_PROBA.md`.
- [x] Add a generic hyperparameter-search lane for deterministic bounded
  Cartesian grids, seeded random candidates, and FortOpt L-BFGS-B over one
  analytic quadratic objective. The 125-grid and 128-sample random lanes have
  independent NumPy checks; the CUDA row is an explicit refusal until search
  state and objective kernels are resident. See
  [`results/HYPERPARAMETER_SEARCH.md`](results/HYPERPARAMETER_SEARCH.md).
- [x] Add bounded binary and shared-kernel one-vs-rest GP classification
  hyperparameter-training rows with an independent NumPy mode/envelope-gradient
  oracle. This records the mode-log-posterior adapter, not full Laplace
  evidence; the raw record is `results/gp_classification_training.csv`.
- [x] Add weighted binary and one-vs-rest Laplace-GP fitting with finite,
  nonnegative row weights, zero-weight curvature suppression, weighted mode
  log posteriors, and weighted kernel envelope gradients. The independent
  NumPy recurrence and refitted-gradient gate are in
  `results/gp_classification_sample_weights.csv`; resident CUDA remains an
  explicit unavailable row.
- [x] Add an inducing-point Bernoulli variational-GP classification lane with
  an independent NumPy ELBO/packed-gradient finite-difference fixture and the
  FortML seeded-Monte-Carlo/parameter-and-query-JVP-VJP/minibatch/L-BFGS-B/refusal test gate. The raw record is
  `results/gp_variational_classification.csv`; CUDA remains a typed refusal
  until the inducing solve and likelihood reduction are resident.
- [x] Add one-vs-rest multiclass variational-GP prediction and parameter-JVP
  evidence. Sorted arbitrary labels, packed per-class ELBO/gradient/JVP, simplex
  normalization, and CUDA refusal are gated in
  `results/gp_variational_multiclass_classification.csv`.
- [x] Add weighted Bernoulli and one-vs-rest variational-GP objectives. The
  independent NumPy oracle checks uniform likelihood-only scaling and a
  nonuniform packed-gradient finite difference; the FortML gate checks shared
  OVR row weights, malformed-weight refusals, CPU dispatch, and typed CUDA
  refusal. The raw record is
  `results/gp_variational_classification_weights.csv`; resident weighted
  inducing solves remain unavailable.
- [x] Add the coupled categorical variational-GP likelihood-temperature lane.
  The independent NumPy oracle recomputes the positive softmax temperature,
  probability JVP, fixed-cotangent probability HVP, and ELBO derivative/HVP
  from emitted latent moments; the raw record is
  `results/gp_categorical_likelihood.csv` and the protocol is documented in
  `results/GP_CATEGORICAL_LIKELIHOOD.md`. The CUDA JVP/HVP product rows remain
  explicit typed refusals until the inducing reduction is resident.
- [x] Add the sparse-GP Gaussian-likelihood log-noise product lane. The
  independent dense NumPy oracle assembles the inducing ELBO and checks the
  transformed log-noise gradient and HVP against central differences. The
  FortML gate checks fixed-state JVP/VJP/HVP products, transactional malformed
  and overflow refusals, and typed CPU/CUDA dispatch. The raw record is
  `results/gp_sparse_likelihood_noise.csv`, with protocol in
  `results/GP_SPARSE_LIKELIHOOD_NOISE.md`; the resident inducing CUDA path
  remains an explicit unavailable row.
- [x] Add the shared binary GP likelihood value/JVP/VJP lane for logistic and
  probit signed margins, including a stable negative-tail oracle and an
  independent adjoint check. The raw record is `results/gp_likelihood.csv`;
  complete scalar FortML release-app output is retained only after the same
  oracle check; absent compiler/app output is explicit `unavailable`, and no
  host timing is presented as GPU evidence.
- [x] Record explicit CUDA capability-refusal rows for the weighted
  elastic-net, OVO logistic, typed schedule, and GP likelihood lanes. Each
  row is `unavailable` with no timing and names the corresponding
  `device_supported(CUDA)=false` boundary; CPU results are never relabeled as
  accelerator evidence. The raw records are `results/elastic_net.csv`,
  `results/ovo_logistic.csv`, `results/mlp_schedules.csv`, and
  `results/gp_likelihood.csv`.
- [x] Add an exact depth-limited recursive second-order boosting lane with independent NumPy
  gradient/Hessian, regularized leaf-weight, split-gain, squared-objective, and
  logistic-objective checks. Record a dedicated FortML workload and an explicit
  optional-XGBoost contextual/refusal row in `results/xgboost_workloads.csv`.
- [x] Extend that lane to one-vs-rest multiclass XGBoost probabilities over
  sorted arbitrary labels. The NumPy oracle rebuilds one exact binary booster
  per class, checks row normalization and argmax labels, and records fit and
  prediction timings in the same raw CSV.
- [x] Extend the exact XGBoost lane with an independent six-sample NaN fixture:
  `missing_policy="learn"` scores both default directions, checks every
  prediction and split gain, and records fit/predict timings. The same CSV
  retains an explicit native-CUDA histogram refusal and LightGBM leaf-wise/
  EFB refusal rather than conflating CPU histogram growth with those
  policies.
- [x] Add correctness-gated weighted CPU histogram workloads for regression,
  binary logistic, and one-vs-rest multiclass XGBoost. The release app uses
  `tree_method="hist"`, `max_bin=2`, and nonuniform sample weights; an
  independent NumPy weighted-quantile oracle checks base scores,
  gradient/Hessian reductions, the selected cut, predictions, and OVR
  normalization in `results/xgboost_workloads.csv`.
- [x] Add correctness-gated binary and multiclass staged XGBoost diagnostics,
  raw multiclass margins, and normalized gain feature importance. The release
  app exports first/final stage checksums and the raw records are retained in
  `results/xgboost_workloads.csv`.
- [x] Add one-file text persistence for `xgboost_multiclass_t`. The independent
  stable-sigmoid and probability-simplex oracle checks arbitrary integer class
  labels, class metadata, round-trip equality, and malformed/truncated refusal;
  the raw record is `results/xgboost_multiclass_persistence.csv` and the
  protocol is documented in `results/XGBOOST_MULTICLASS_PERSISTENCE.md`.
- [x] Add weighted validation and common-prefix early stopping for
  `xgboost_multiclass_t`. `scripts/bench_xgboost_multiclass_validation.py`
  independently replays all depth-one OVR Newton stages, weighted normalized
  multiclass log-loss, best-prefix restoration, arbitrary validation labels,
  and transactional unknown-label refusal. The release app records requested,
  retained, best-round, loss, and patience metadata plus the typed CUDA tree
  refusal in `results/xgboost_multiclass_validation.csv`; the protocol is
  documented in [`results/XGBOOST_MULTICLASS_VALIDATION.md`](results/XGBOOST_MULTICLASS_VALIDATION.md).
- [x] Add a correctness-gated squared-log (RMSLE) XGBoost lane. The independent
  NumPy oracle solves the transformed-coordinate one-split Newton fixture,
  while the release app records exact CPU fit/predict, weighted-histogram
  diagnostics, and a typed CUDA refusal in
  `results/xgboost_squared_log.csv`.
- [x] Add a fixed-shape Gamma log-link XGBoost lane. The independent NumPy
  oracle checks the positive-target one-split gradient/Hessian and inverse-link
  prediction, while the release app records exact CPU fit/predict,
  weighted-histogram diagnostics, and an explicit CUDA refusal in
  `results/xgboost_gamma.csv`.
- [x] Add the `rank:pairwise` XGBoost lane. An independent pairwise logistic
  loss/gradient/Hessian oracle is paired with FortML fit ordering, query
  isolation, and singleton-query refusal in `results/xgboost_ranking.csv`.
- [x] Add an absolute-deviation XGBoost lane. The independent one-tree NumPy
  oracle checks the weighted-median identity-link base margin, sign
  subgradient, positive Hessian-floor leaf corrections, CPU fit/predict
  timings, and the explicit CUDA refusal in `results/xgboost_absolute.csv`.
- [x] Add matched multinomial softmax regression and multiclass neural
  classifier lanes. `scripts/bench_classification_models.py` uses independent
  NumPy damped-Newton and full-batch Adam oracles, records scikit-learn and
  optional resident PyTorch context timings, and emits explicit FortML
  target/dependency refusals. A FortML pass
  additionally requires complete probability/prediction arrays and the
  release-app timing protocol documented in
  `results/CLASSIFICATION_MODELS.md`. The current raw record is
  `results/classification_models.csv`.
- [x] Add a relaxed Bernoulli Naive Bayes lane with an independent NumPy
  likelihood/log-softmax/input-JVP oracle, scikit-learn context rows, and
  matched FortML release-app rows (with an explicit refusal if the target is
  absent). The raw record is `results/bernoulli_naive_bayes.csv` and the
  protocol is documented in `results/BERNOULLI_NB.md`.
- [x] Add a differentiable Multinomial Naive Bayes lane with independent
  smoothed token-mass, log-softmax, prediction, and input-JVP oracles, a
  scikit-learn context row, and complete FortML release-app output arrays. The
  raw record is `results/multinomial_naive_bayes.csv` and the protocol is
  documented in `results/MULTINOMIAL_NB.md`.
- [x] Add a differentiable Complement Naive Bayes lane with independent
  complement-count, positive-weight, log-softmax, prediction, and input-JVP
  oracles.  The scikit-learn multiclass prior-intercept difference is retained
  as contextual evidence, and missing FortML targets remain explicit refusal
  rows.  The raw record is `results/complement_naive_bayes.csv` and the
  protocol is documented in `results/COMPLEMENT_NB.md`.
- [x] Add an integer one-hot encoder lane with independent sorted-category,
  packed-offset, missing/unknown-policy, and complete dense-transform checks.
  Categorical JVP/VJP are explicit refusals because integer categories have no
  canonical tangent space.  The raw record is `results/one_hot_encoder.csv`
  and the protocol is documented in `results/ONE_HOT_ENCODER.md`.
- [x] Add a grouped K-fold validation lane with an independent stable
  largest-first packing oracle, complete test-index and group-isolation checks,
  release-app split timing, and an explicit CUDA capability refusal. The raw
  record is `results/group_kfold.csv`; the protocol is documented in
  `results/GROUP_KFOLD.md`.
- [x] Add a chronological time-series validation lane with an independent
  NumPy blocked-window oracle, expanding/rolling training semantics, explicit
  gap rows, complete one-based train/test assignment checks, scorer-orientation
  metadata, release-app timing, and a typed CUDA capability refusal. The raw
  record is `results/time_series_split.csv`; the protocol is documented in
  `results/TIME_SERIES_SPLIT.md`. Repeated K-fold, Monte Carlo scoring, and
  generic model cloning remain separate open workflows.
- [x] Add a centered dense PCA lane with an independent NumPy thin-SVD oracle,
  scikit-learn full-SVD context, deterministic sign/rank checks, and the
  FortML release-app orthonormality/timing protocol. The raw record is
  `results/pca.csv`; complete FortML fitted-array export remains explicitly
  open in `results/PCA.md`.
- [x] Add a deterministic dense k-means lane with an independent seeded Lloyd
  oracle, final-inertia gate, release fit/transform timings, and an explicit
  CUDA refusal. The raw record is `results/kmeans.csv`; the protocol is
  documented in `results/KMEANS.md`.
- [x] Add a dense robust-scaler lane with an independent NumPy
  linear-interpolation median/IQR oracle, complete transform/inverse/JVP
  checks, release transform/JVP checksum gates, and an explicit CUDA refusal.
  The raw record is `results/robust_scaler.csv`; the protocol is documented in
  `results/ROBUST_SCALER.md`.
- [x] Add a weighted multi-output ridge lane with an independent NumPy
  closed-form oracle, vector/matrix prediction, and packed coefficient/input
  JVP/VJP checks. The raw record is `results/ridge.csv`; a complete-array
  FortML release app remains an explicit unavailable boundary until it is
  added.
- [x] Add a weighted multi-output elastic-net lane with an independent NumPy
  coordinate-descent oracle, complete coefficients/predictions, and packed
  coefficient/input JVP/VJP checks. The raw record is `results/elastic_net.csv`
  and the strict release-app protocol is documented in
  `results/ELASTIC_NET.md`.
- [x] Add a composable sequential MLP module-tree lane with an independent
  NumPy two-stage value/JVP/VJP oracle and differentiated-VJP HVP check. The
  raw record is `results/mlp_chain.csv`, with separate predict/JVP/VJP/HVP
  timings and an explicit CUDA refusal until a resident fused chain kernel is
  available. The protocol is documented in `results/MLP_CHAIN.md`.
- [x] Add the finite-feature GP/NTK last-layer initializer lane. The independent
  NumPy oracle reproduces the deterministic hidden MLP feature map and solves
  the augmented kernel-ridge normal equations; the raw record is
  `results/mlp_last_layer_gp.csv`, with a typed CUDA-unavailable row and an
  explicit finite-width approximation boundary. The clean release-app CPU row
  matches the NumPy posterior MSE to `2.78e-17` on source `c2d9c4c`; the
  protocol and provenance are documented in `results/MLP_LAST_LAYER_GP.md`.
- [ ] Add matched GPyTorch variational-likelihood and calibrated-likelihood
  comparisons. FortML's CPU inducing-point Bernoulli ELBO, packed gradient,
  JVP, and CUDA refusal are covered by
  `results/gp_variational_classification.csv`; full likelihood catalogs,
  natural-gradient optimization, and resident GPU training remain open.
- [x] Add a GPyTorch-compatible spectral-mixture kernel lane with independent
  NumPy value, input-product, parameter JVP/VJP/HVP, and packed-metadata
  oracles. `results/spectral_mixture.csv` records the two-mixture CPU release
  app and typed resident-CUDA refusal; the protocol is documented in
  `results/SPECTRAL_MIXTURE.md`.
- [x] Add the periodic derivative-GP mixed-observation HVP lane. The
  independent NumPy dense oracle central-differences the likelihood gradient
  for all packed logarithmic kernel/noise coordinates; the FortML app checks
  the analytic radial fourth-input product and records CPU timing plus the
  typed resident-CUDA refusal in `results/derivative_gp_periodic_hvp.csv`.
  The protocol is documented in `results/DERIVATIVE_GP_PERIODIC_HVP.md`.
- [x] Add the locally-periodic exact-GP lane with separate vectorized and
  scalar-loop covariance oracles, exact posterior moments, input and
  logarithmic parameter products, and an explicit static-operator/CUDA
  refusal. The raw record is `results/local_periodic_gp.csv`; the protocol is
  documented in `results/LOCAL_PERIODIC_GP.md`.
- [ ] Add the remaining histogram/CART feature matrix: weighted missing-bin
  workloads, class weights, monotonic constraints, early stopping, feature
  importance, categorical inputs, and native GPU histograms. CPU weighted
  binary/regression/multiclass histogram growth is now covered by the release
  lane; the CUDA histogram gate remains open.
- [x] Add the bounded LightGBM-style leaf-wise lane. `scripts/bench_lightgbm.py`
  checks an independent six-row weighted-Newton oracle, while the FortML app
  records weighted regression, binary-logistic, deterministic best-leaf CPU
  timings, and a typed CUDA refusal in `results/lightgbm_leafwise.csv`.
  EFB, categorical statistics, distributed workers, and resident GPU
  histograms remain open.
- [x] Add the LightGBM GOSS lane. `scripts/bench_lightgbm_goss.py` independently
  replays top-gradient retention, hash-ranked other-row sampling, and the
  `(1-top_rate)/other_rate` gradient/Hessian correction. The raw
  `results/lightgbm_goss.csv` records CPU fit/predict and seed-replay checks,
  malformed-rate refusal, and typed CUDA refusal.
- [x] Add the bounded LightGBM DART/dropout lane. `scripts/bench_lightgbm_dart.py`
  independently replays seeded prior-tree dropout and `1/(k+1)` tree
  normalisation. `results/lightgbm_dart.csv` records the CPU prediction oracle,
  deterministic scales, schema-3 persistence, warm-start parity, malformed-rate
  refusal, and typed CUDA refusal; fit-time dropout derivatives remain an
  explicit discrete-fit boundary.
- [x] Add fixed-structure XGBoost/LightGBM leaf-coordinate JVP/VJP products.
  `scripts/bench_tree_leaf_products.py` checks packed base-plus-leaf routing,
  forward/reverse contractions, and the independent two-leaf NumPy oracle;
  `results/tree_leaf_products.csv` records zero CPU error and explicit
  unavailable resident-CUDA rows. Split routing and fit-time derivatives stay
  separate discrete boundaries.
- [x] Add transactional multi-output XGBoost and LightGBM regression adapters.
  `scripts/bench_xgboost_multioutput.py` computes an independent NumPy
  two-output one-tree Newton-stump oracle and checks matrix predictions,
  staged margins, concatenated fixed-leaf JVP/VJP products, metadata,
  malformed-fit transactionality, and typed CUDA refusals in
  `results/xgboost_multioutput.csv`. Resident multi-output GPU kernels and
  distributed output sharding remain open.
- [x] Add bounded seeded XGBoost DART/dropout boosting. `booster="dart"`
  selects prior trees through a compiler-independent hash stream and persists
  per-tree `1/(k+1)` normalisation in schema-5 snapshots. The release app and
  `scripts/bench_xgboost_dart.py` independently replay a depth-one NumPy
  Newton oracle and gate staged margins, additive contributions, deterministic
  replay, warm-start suffixes, malformed-control refusals, and the typed CUDA
  boundary in `results/xgboost_dart.csv`; fit-time dropout remains a discrete
  split boundary.
- [ ] Add a matched full XGBoost lane when the optional dependency and a pinned
  release are available. The current exact depth-limited and weighted CPU
  histogram FortML lanes are recorded in `results/xgboost_workloads.csv`; both
  have independent NumPy correctness gates. Categorical, constraint, and native
  GPU histogram comparisons remain separate work. The CPU `rank:pairwise`
  objective now has its own independent gate in `results/xgboost_ranking.csv`;
  matched external ranking comparisons remain open. A stump benchmark is not
 called XGBoost.
- [x] Add the XGBoost interaction-constraint lane. The release app fits
  unconstrained and separated-group depth-two trees, while
  `scripts/bench_xgboost_interaction.py` independently reconstructs the
  group-mean/path-mask oracle and checks the complete prediction vector,
  diagnostics, and typed CUDA refusal. The raw record is
  `results/xgboost_interaction.csv`; the CPU rows pass with zero error.
- [x] Add the bounded ordered-gradient categorical XGBoost lane. The release
  app fits one integer-coded four-category feature with an independent
  two-group NumPy partition oracle; `scripts/bench_xgboost_categorical.py`
  checks the complete prediction vector and fit diagnostic, records repeated
  CPU prediction timing, and emits an explicit CUDA-unavailable row. Raw
  evidence is in `results/xgboost_categorical.csv` and the protocol is
  documented in [`results/XGBOOST_CATEGORICAL.md`](results/XGBOOST_CATEGORICAL.md).
- [x] Add binary XGBoost classifier log-probability products. The independent
  fixture checks stable `exp(predict_log_proba)` round trips, while the release
  app records the direct log-space error, staged consistency, and typed CUDA
  refusal in `results/xgboost_classifier_log_proba.csv`; the protocol is
  documented in [`results/XGBOOST_CLASSIFIER_LOG_PROBA.md`](results/XGBOOST_CLASSIFIER_LOG_PROBA.md).
- [x] Add the binary AdaBoost lane. The release app fits a weighted depth-one
  CART learner, and `scripts/bench_adaboost_classifier.py` independently
  reconstructs the learner error, alpha, signed margin, probabilities, and
  labels. The raw record is `results/adaboost_classifier.csv`; the CUDA row is
  an explicit typed refusal.
- [x] Add the multiclass SAMME lane. The release app fits a deterministic
  weighted-CART stump over sorted arbitrary integer labels, and
  `scripts/bench_adaboost_samme.py` independently reconstructs the
  random-guessing-bound error, `log((1-error)/error)+log(K-1)` stage weight,
  weighted-vote margins, stabilized softmax, and labels. The raw record is
  `results/adaboost_samme.csv`; the CUDA row is an explicit typed refusal.
- [x] Add the multiclass SAMME.R probability-update lane. The release app fits
  a deterministic weighted-CART stump with clipped centred log-probability
  updates, unit stage weights, and the normalized geometric probability
  ensemble. `scripts/bench_adaboost_samme_r.py` independently replays the
  clipped NumPy oracle, checks probabilities and labels, and records the typed
  CUDA refusal in `results/adaboost_samme_r.csv`; the protocol is documented in
  `results/ADABOOST_SAMME_R.md`.
- [x] Add the seeded bagging classifier lane. The release app fits a
  bootstrap CART ensemble, while `scripts/bench_bagging_classifier.py`
  independently checks the six cluster labels and probability simplex. The
  raw record is `results/bagging_classifier.csv`; the CUDA row is an explicit
  typed refusal until a resident ensemble kernel is linked.
- [x] Add the random-forest OOB lane. The release app retains one bootstrap
  inclusion column per tree and checks transactional OOB decision probabilities,
  OOB accuracy, complete coverage, and the typed CUDA refusal. The Python gate
  independently reconstructs the threshold-label fixture and rejects any
  hidden in-bag fallback. Raw evidence is `results/random_forest_oob.csv` and
  the protocol is documented in `results/RANDOM_FOREST_OOB.md`.
- [ ] Extend the existing physics-informed, Hamiltonian, and symplectic
  workloads with manufactured-PDE, Lagrangian, posterior-calibration, and
  long-horizon trajectory studies. The canonical harmonic-oscillator residual
  and Jacobian-defect gate is recorded above.
- [ ] Add physics-consistent GP, Ghosttasking, Monge-GP, and GP-initialized
  finite-network lanes when their public equations and reference data are
  pinned. Private project results must be accompanied by a reproducible data
  generator and versioned artifact.

Every new lane keeps the independent behavioral oracle ahead of timing. A
missing compiler, GPU, package, equation, or reference dataset produces a
parseable refusal row rather than an omitted result.
