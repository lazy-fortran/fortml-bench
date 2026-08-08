# fortml-bench

`fortml-bench` holds the expensive performance comparisons for
[fortml](https://github.com/lazy-fortran/fortml). It measures the Fortran
implementation against dense PyTorch, GPyTorch, and KeOps on matched inputs,
precisions, devices, and stopping rules.

The library repositories keep unit tests and small smoke workloads. This
repository owns cross-engine environments, large sweeps, compiler reports,
peak-memory measurements, raw CSV records, plots, and shareable plot URLs.

## First workload

The initial workload is an RBF kernel matrix-vector product

\[
 y_i = \sigma_n^2 v_i + \sigma_f^2 \sum_j
 \exp\left(-\frac{\lVert x_i-x_j\rVert^2}{2\ell^2}\right)v_j.
\]

The direct NumPy implementation is the independent correctness oracle. Dense
PyTorch, KeOps, GPyTorch with its KeOps kernel, and the Fortran operator are
competitors. Correctness is checked before timing.

The benchmark records resident and transfer-inclusive device timings. First
call setup time is recorded separately because KeOps may compile a generated
kernel on its first invocation.

## Run

Create the reference environment and run the CPU/GPU suite:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
./scripts/run_suite.sh
```

The Fortran checkout is selected with `FORTML_DIR`. It defaults to the sibling
checkout `../fortml`. The scripts record both source commits.

```bash
FORTML_DIR=../fortml ./scripts/run_suite.sh
```

Results are written under `results/`. The plot command writes a PNG and prints
its local path. The checked-in first run is summarized in
[`results/README.md`](results/README.md). Its raw CSV and plot are next to
that note.

## Complete model and GP-feature calls

The small exact-GP and MLP harness separates fit, prediction, forward, and
reverse-product timings. It checks complete outputs with independent NumPy
implementations before timing and records unsupported FortML CUDA rows
explicitly:

```bash
.venv/bin/python -B scripts/bench_model_workloads.py \
    --output results/model_workloads.csv
.venv/bin/python -B scripts/plot_model_workloads.py \
    --input results/model_workloads.csv --output-dir results
```

The method-level GP harness covers stochastic log determinants, predictive
variance, derivative observations, multi-output regression, and a variational
ELBO plus prediction:

```bash
.venv/bin/python -B scripts/bench_gp_features.py \
    --output results/gp_features.csv --plot results/gp_features.png
```

Both commands use the shared `../fortml/build/fo` directory. Run them
serially, with no other `fo` build active. Workload definitions, validity
boundaries, plots, and the recorded machine results are in
[`results/MODEL_WORKLOADS.md`](results/MODEL_WORKLOADS.md) and
[`results/GP_FEATURES.md`](results/GP_FEATURES.md).

The dedicated multi-output product lane adds an independent dense
intrinsic-coregionalization oracle for output-major posterior means, query
input JVPs, packed kernel/noise/coregionalization parameter JVP/VJPs, and the
explicit CUDA refusal:

```bash
python -B scripts/bench_multi_output_gp_products.py \
  --fortml ../fortml --output results/multi_output_gp_products.csv
```

See [`results/MULTI_OUTPUT_GP_PRODUCTS.md`](results/MULTI_OUTPUT_GP_PRODUCTS.md).

The batched multi-output product lane checks independent query-set stacking,
fixed-fit input JVP/VJP products, CPU dispatch, malformed-shape validation,
and the typed CUDA boundary against a direct NumPy intrinsic-coregionalization
oracle:

```bash
python -B scripts/bench_multi_output_gp_batch.py \
  --fortml ../fortml --output results/multi_output_gp_batch.csv
```

See [`results/MULTI_OUTPUT_GP_BATCH.md`](results/MULTI_OUTPUT_GP_BATCH.md).

The exact-GP mean lane checks trainable constant and intercept-plus-linear
means, packed coefficients, posterior predictions, log marginal likelihood,
and analytic mean hypergradients/HVPs against an independent NumPy covariance
oracle. Exact-GP CUDA remains an explicit typed unavailable row:

```bash
python -B scripts/bench_gp_mean.py \
  --fortml ../fortml --output results/gp_mean.csv
```

See [`results/GP_MEAN.md`](results/GP_MEAN.md).

The Student-t process lane checks the large-degree-of-freedom exact-GP limit
and the defining data-dependent covariance contrast against an independent
NumPy Cholesky oracle. It also records the typed refusal at `nu <= 2`, where a
finite covariance does not exist:

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_student_t_process.py \
  --fortml ../fortml --output results/student_t_process.csv
```

See [`results/STUDENT_T_PROCESS.md`](results/STUDENT_T_PROCESS.md).

The supplied-noise heteroskedastic GP lane checks the constant-noise reduction
to an ordinary GP, the quiet/noisy posterior contrast, and positive log-noise
interpolation against an independent NumPy diagonal-noise oracle. A zero
observation variance is recorded as an explicit typed refusal:

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_heteroskedastic_gp.py \
  --fortml ../fortml --output results/heteroskedastic_gp.csv
```

See [`results/HETEROSKEDASTIC_GP.md`](results/HETEROSKEDASTIC_GP.md).

The robust observation-model GP lane checks Poisson mode stationarity and
positive log-rate responses, then contrasts a Student-t fit with a Gaussian
fit after one large outlier. Malformed counts, likelihoods, and degrees of
freedom remain explicit refusal rows:

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_robust_gp.py \
  --fortml ../fortml --output results/robust_gp.csv
```

See [`results/ROBUST_GP.md`](results/ROBUST_GP.md).

The bounded second-derivative GP lane checks mixed value/gradient/Hessian
observations for scalar one-dimensional RBF and Matérn-5/2 references. An
independent NumPy covariance oracle covers posterior moments, latent joint
covariance, query JVP finite differences, VJP duality, and the typed
CUDA/non-RBF/order/coincidence refusal boundaries:

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_second_derivative_gp.py \
  --fortml ../fortml --output results/second_derivative_gp.csv
```

See [`results/SECOND_DERIVATIVE_GP.md`](results/SECOND_DERIVATIVE_GP.md).

The locally-periodic GP lane checks a four-parameter logarithmic covariance,
coincident-safe input products, parameter JVP/VJP/HVP products, and exact-GP
posterior moments against separate vectorized and scalar-loop NumPy oracles.
The FortML behavioral gate also records the static-operator/CUDA refusal:

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_local_periodic_gp.py \
  --fortml ../fortml --output results/local_periodic_gp.csv
```

See [`results/LOCAL_PERIODIC_GP.md`](results/LOCAL_PERIODIC_GP.md).

The change-point GP lane checks a gated RBF-plus-constant covariance, exact-GP
posterior moments, and packed parameter JVP/VJP/HVP products against an
independent NumPy covariance and central-difference oracle. The Fortran gate
covers input gradients and mixed Hessians. Resident CUDA covariance remains a
typed refusal:

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_change_point_gp.py \
  --fortml ../fortml --output results/change_point_gp.csv
```

See [`results/CHANGE_POINT_GP.md`](results/CHANGE_POINT_GP.md).

The production Lion trainer lane checks the stateful CPU recurrence, decoupled
weight decay, clipping, EMA, validation, and uninterrupted versus checkpointed
text-resume trajectories against an independent NumPy oracle. Resident Lion
state is currently an explicit CUDA-unavailable row:

```bash
python3 -B scripts/bench_lion_training.py \
  --fortml ../fortml --output results/lion_training.csv
```

See [`results/LION_TRAINING.md`](results/LION_TRAINING.md).

The dense k-means lane checks deterministic seeded Lloyd fit, final inertia,
and transform timing against an independent NumPy implementation. CUDA is a
typed unavailable row until resident clustering state is linked:

```bash
python -B scripts/bench_kmeans.py --fortml ../fortml \
  --output results/kmeans.csv
```

See [`results/KMEANS.md`](results/KMEANS.md).

The dense robust-scaler lane checks median/IQR fitting, transform, inverse
transform, and the input JVP against an independent NumPy linear-interpolation
oracle.  The FortML release app's transform and JVP checksums gate its CPU
timing, while CUDA remains an explicit typed-unavailable row:

```bash
python -B scripts/bench_robust_scaler.py --fortml ../fortml \
  --output results/robust_scaler.csv
```

See [`results/ROBUST_SCALER.md`](results/ROBUST_SCALER.md).

The ARD-GP lane checks an anisotropic RBF covariance with one length scale per
feature, input gradients and mixed Hessians, analytic parameter JVP/VJP/HVP
products, and exact-GP posterior and hyperparameter-gradient products against
an independent NumPy oracle. Exact-GP ARD CUDA remains an explicit typed
unavailable row until a resident anisotropic covariance kernel is linked:

```bash
python -B scripts/bench_gp_ard.py \
  --fortml ../fortml --output results/gp_ard.csv
```

See [`results/GP_ARD.md`](results/GP_ARD.md).

The kernel-catalog lane checks periodic, rational-quadratic, cosine, and
polynomial covariance leaves, their input derivatives, and logarithmic
parameter JVP/VJP/HVP products against independent NumPy formulas. It records
host timings and explicit CUDA capability refusals until resident kernels are
linked:

```bash
python -B scripts/bench_kernel_catalog.py \
  --fortml ../fortml --output results/kernel_catalog.csv
```

See [`results/KERNEL_CATALOG.md`](results/KERNEL_CATALOG.md).

The spectral-mixture lane checks the GPyTorch-compatible stationary mixture
formula, compositional packed metadata, dense value and input products, and
parameter JVP/VJP/HVP products against an independent NumPy oracle.  It records
FortML CPU timings and an explicit resident-CUDA refusal:

```bash
python -B scripts/bench_spectral_mixture.py \
  --fortml ../fortml --output results/spectral_mixture.csv
```

See [`results/SPECTRAL_MIXTURE.md`](results/SPECTRAL_MIXTURE.md).

The weighted discriminant-analysis lane checks arbitrary integer-label LDA and
QDA fits, stabilized class probabilities, predictions, and fixed-state input
JVPs against an independent NumPy Gaussian-discriminant oracle. It also records
the fitted moments and explicit CUDA refusals:

```bash
python -B scripts/bench_discriminant_analysis.py \
  --fortml ../fortml --output results/discriminant_analysis.csv
```

See [`results/DISCRIMINANT_ANALYSIS.md`](results/DISCRIMINANT_ANALYSIS.md).

The deterministic random-forest lane exercises the seeded bootstrap CART
ensemble on a separated three-class fixture. A direct NumPy threshold oracle
checks all six query labels and the probability-simplex contract without a
scikit-learn dependency:

```bash
python -B scripts/bench_random_forest.py \
  --fortml ../fortml --output results/random_forest.csv
```

See [`results/RANDOM_FOREST.md`](results/RANDOM_FOREST.md). The CSV records
CPU fit/predict timings and explicit CUDA capability and plan-creation
refusals until a resident tree-ensemble kernel is available. The plan ABI
version and fitted shape are checked without allocating or copying host trees.

The Extra-Trees lane exercises the seeded randomized-threshold classifier on
the same separated fixture. A direct NumPy class-rule oracle checks all query
labels and probability normalization; the CUDA row is a typed refusal until a
resident no-autodiff tree kernel is linked:

```bash
python -B scripts/bench_extra_trees.py \
  --fortml ../fortml --output results/extra_trees.csv
```

See [`results/EXTRA_TREES.md`](results/EXTRA_TREES.md).

The grouped MLP lane checks named per-parameter log-L2 regularization,
including value, JVP, gradient norm, and mixed HVP norm, against a closed-form
NumPy linear-ridge oracle. Its CUDA row is explicit `unavailable` because a
resident MLP derivative graph is not yet linked:

```bash
python -B scripts/bench_mlp_grouped_training.py \
  --fortml ../fortml --output results/mlp_grouped_training.csv
```

See [`results/MLP_GROUPED_TRAINING.md`](results/MLP_GROUPED_TRAINING.md).

The shared feature workload now includes an independent central-difference
of-VJP oracle and timing row for analytic basis-pipeline HVPs:

```bash
python -B scripts/bench_features.py \
  --fortml ../fortml --output results/features_workloads.csv
```

See [`results/FEATURES.md`](results/FEATURES.md); the HVP row is CPU evidence
for the polynomial/Fourier pipeline contract, not a GPU claim.

The derivative-GP lane checks exact query-input JVP/VJP products and dense
posterior covariance parameter JVP/VJP products for mixed value/first-derivative
periodic, rational-quadratic, cosine, polynomial, spectral-mixture, and ARD-RBF
GPs against an independent NumPy covariance and posterior finite-difference
oracle. The polynomial and ARD-RBF lanes also check mixed-observation
hyperparameter HVPs through the Cholesky likelihood, including the degree-one
limit and every ARD log lengthscale. CPU timings are retained only
after the checks pass; CUDA is an explicit typed refusal until the resident
derivative-GP graph is linked:

```bash
python -B scripts/bench_derivative_gp.py \
  --fortml ../fortml --output results/derivative_gp.csv
```

See [`results/DERIVATIVE_GP.md`](results/DERIVATIVE_GP.md).

The binary classification lane compares the FortML logistic estimator with
scikit-learn on a deterministic two-label fixture. It checks full predicted
labels and probabilities with independent NumPy accuracy, log-loss, and
confusion-matrix calculations cross-checked by `sklearn.metrics` before timing:

```bash
.venv/bin/python -B scripts/bench_classification.py \
    --fortml ../fortml --output results/classification_workloads.csv
```

The workload definition and validity boundary are in
[`results/CLASSIFICATION.md`](results/CLASSIFICATION.md).

The binary MLP lane checks one-logit Adam training, arbitrary integer class
labels, probabilities, BCE gradients, and parameter HVPs against an independent
NumPy oracle. The CUDA row remains an explicit typed refusal until a resident
MLP classifier graph is linked:

```bash
python -B scripts/bench_mlp_binary_classifier.py \
  --fortml ../fortml --output results/mlp_binary_classifier.csv
```

See [`results/MLP_BINARY_CLASSIFIER.md`](results/MLP_BINARY_CLASSIFIER.md).

The multiclass MLP objective lane checks weighted softmax cross-entropy,
parameter/L2 value and gradient products, exact nonlinear HVPs, FortOpt
callback routing, and bounded L-BFGS-B against an independent NumPy affine
logits oracle. CUDA remains an explicit typed refusal until resident
multiclass objective state is linked:

```bash
python3 -B scripts/bench_mlp_classifier_objective.py \
  --fortml ../fortml --output results/mlp_classifier_objective.csv
```

See [`results/MLP_CLASSIFIER_OBJECTIVE.md`](results/MLP_CLASSIFIER_OBJECTIVE.md).

The multilabel MLP lane checks two independent sigmoid heads, concatenated
parameters, multilabel probabilities and indicators, BCE gradients, parameter
HVPs, and probability JVPs against an independent NumPy oracle. CUDA remains
an explicit typed refusal until a resident multilabel MLP graph is linked:

```bash
python -B scripts/bench_mlp_multilabel_classifier.py \
  --fortml ../fortml --output results/mlp_multilabel_classifier.csv
```

See [`results/MLP_MULTILABEL_CLASSIFIER.md`](results/MLP_MULTILABEL_CLASSIFIER.md).

The extended classification lane checks fitted standard/min-max scalers and
binary Laplace GP logistic/probit inference against independent NumPy solves:

```bash
.venv/bin/python -B scripts/bench_classification_extensions.py \
  --fortml ../fortml --output results/classification_extensions.csv
```

See [`results/CLASSIFICATION_EXTENSIONS.md`](results/CLASSIFICATION_EXTENSIONS.md).
The lane now includes one-vs-rest multiclass Laplace GP probabilities. Variational
GP classification remains a separate benchmark contract.

The one-vs-one logistic lane fits six deterministic pair estimators on four
arbitrary integer classes. Its independent NumPy/Newton oracle checks the
complete pair-vote probabilities, labels, and normalization before timing:

```bash
.venv/bin/python -B scripts/bench_ovo_logistic.py \
  --fortml ../fortml --output results/ovo_logistic.csv
```

See [`results/OVO_LOGISTIC.md`](results/OVO_LOGISTIC.md). The CSV includes
explicit CUDA capability-refusal rows; scikit-learn is contextual because its
pairwise probability coupling policy differs from FortML's declared vote map.

The dense multilabel-indicator lane fits one independent logistic head per
zero/one target column and checks the complete positive-probability matrix and
hard indicator matrix against an independent NumPy Newton oracle:

```bash
.venv/bin/python -B scripts/bench_multilabel_logistic.py \
  --fortml ../fortml --output results/multilabel_logistic.csv
```

See [`results/MULTILABEL_LOGISTIC.md`](results/MULTILABEL_LOGISTIC.md). CUDA
capability rows are explicit refusals until a resident multi-head kernel is
linked; host timings are never relabeled as accelerator evidence.

The multilabel metrics lane checks micro, macro, and samples precision, recall,
F1, weighted F-beta (`beta=2`), Jaccard, and Hamming loss, together with `>=`
probability thresholding, against independent NumPy TP/FP/FN and
intersection/union/error oracles. It records the typed CUDA refusal until
resident multilabel reduction kernels are available:

```bash
python -B scripts/bench_multilabel_metrics.py \
  --fortml ../fortml --output results/multilabel_metrics.csv
```

See [`results/MULTILABEL_METRICS.md`](results/MULTILABEL_METRICS.md).

The ROC-AUC lane separately checks binary and one-vs-rest multiclass ROC AUC
with arbitrary integer labels and half-credit ties, and binary/OVR PR AUC with
average-precision threshold-group semantics, against independent NumPy
oracles. Its CUDA rows remain explicit refusals until resident ranking/reduction
kernels are linked:

```bash
python -B scripts/bench_roc_auc.py \
  --fortml ../fortml --output results/roc_auc.csv
```

See [`results/ROC_AUC.md`](results/ROC_AUC.md).

The radius-neighbor lane checks a closed Euclidean-radius classifier with
inverse-distance votes, sample weights, arbitrary labels, and an explicit
outlier policy against a complete NumPy oracle:

```bash
.venv/bin/python -B scripts/bench_radius_neighbors.py \
    --fortml ../fortml --output results/radius_neighbors.csv
```

See [`results/RADIUS_NEIGHBORS.md`](results/RADIUS_NEIGHBORS.md). CUDA is an
explicit unavailable capability row until a resident radius-search kernel is
linked.

The scalar radius-neighbor regression lane checks uniform or inverse-distance
sample-weighted averaging, an explicit empty-neighborhood value, and all
scalar predictions against an independent NumPy oracle:

```bash
.venv/bin/python -B scripts/bench_radius_neighbors_regression.py \
    --fortml ../fortml --output results/radius_neighbors_regression.csv
```

See [`results/RADIUS_NEIGHBORS_REGRESSION.md`](results/RADIUS_NEIGHBORS_REGRESSION.md).
CUDA remains an explicit unavailable capability row until a resident
radius-search reduction is linked.

The multi-output radius-neighbor regression lane checks shared uniform
neighborhood membership, vector-valued outliers, all two target columns, and
the release app's summary against an independent NumPy oracle:

```bash
.venv/bin/python -B scripts/bench_radius_neighbors_multioutput.py \
    --fortml ../fortml --output results/radius_neighbors_multioutput.csv
```

See [`results/RADIUS_NEIGHBORS_MULTIOUTPUT.md`](results/RADIUS_NEIGHBORS_MULTIOUTPUT.md).
CUDA remains an explicit unavailable capability row until a resident
radius-search reduction is linked.

The dense RBF one-class SVM lane checks a capped-simplex dual fit, KKT offset,
signed scores, and `+1`/`-1` anomaly labels against an independent NumPy oracle,
then records NumPy and checked FortML CPU fit/predict timings.  The CUDA row
remains an explicit typed-refusal record until a resident release kernel is
linked:

```bash
.venv/bin/python -B scripts/bench_one_class_svm.py \
  --fortml ../fortml --output results/one_class_svm.csv
```

See [`results/ONE_CLASS_SVM.md`](results/ONE_CLASS_SVM.md).  Host timings are
never relabeled as GPU evidence.

The linear-margin lane checks weighted primal squared-hinge SVM fitting,
arbitrary integer classes, labels, and signed decision margins against an
independent NumPy/SciPy L-BFGS-B oracle:

```bash
python -B scripts/bench_linear_svm.py \
    --fortml ../fortml --output results/linear_svm.csv
```

See [`results/LINEAR_SVM.md`](results/LINEAR_SVM.md). CUDA is an explicit
unavailable capability row until a resident linear-SVM kernel is linked.

The dense finite-basis RBF-SVM lane checks the weighted squared-hinge RKHS
score/label map against an independent SciPy L-BFGS-B solve with arbitrary
integer classes. Its FortML gate also checks fixed-state input/parameter JVPs
and VJPs, CPU dispatch, and typed CUDA derivative refusals:

```bash
python -B scripts/bench_rbf_svm.py \
    --fortml ../fortml --output results/rbf_svm.csv
```

See [`results/RBF_SVM.md`](results/RBF_SVM.md). CUDA value and derivative rows
remain explicit unavailable capability records until resident kernels are linked.

The linear-regression margin lane checks weighted dense primal SVR fitting for
arbitrary real targets, epsilon-insensitive prediction, packed affine
parameters, and an independent NumPy/SciPy L-BFGS-B oracle:

```bash
python -B scripts/bench_linear_svr.py \
    --fortml ../fortml --output results/linear_svr.csv
```

See [`results/LINEAR_SVR.md`](results/LINEAR_SVR.md). CUDA is an explicit
unavailable capability row until a resident linear-SVR kernel is linked.

The differentiable neural-loss lane checks BCE/logistic, softmax
cross-entropy, weighted MSE, Huber, MAE, focal BCE-with-logits, Gaussian NLL,
and Poisson/count NLL products against independent NumPy value/derivative
formulas, and checks the weighted-MSE path used by the MLP objective:

```bash
python -B scripts/bench_neural_losses.py \
    --fortml ../fortml --output results/neural_losses.csv
```

See [`results/NEURAL_LOSSES.md`](results/NEURAL_LOSSES.md). CUDA is recorded
as an explicit unavailable capability until resident loss and MLP objective
kernels exist; no host fallback is timed.

The ordered-label lane fits a weighted cumulative-logit classifier with
strictly increasing cut points. It checks the complete probability matrix and
predicted labels against an independent SciPy L-BFGS-B oracle before timing:

```bash
.venv/bin/python -B scripts/bench_ordinal_logistic.py \
    --fortml ../fortml --output results/ordinal_logistic.csv
```

See [`results/ORDINAL_LOGISTIC.md`](results/ORDINAL_LOGISTIC.md). CUDA rows
are explicit `unavailable` capability records until a resident ordinal kernel
is linked; no host fallback is timed.

The grouped-validation lane checks deterministic largest-first group packing
on an uneven six-group fixture. Every FortML test assignment is compared with
an independent NumPy oracle and group isolation is checked before retaining
the CPU split-generation timing:

```bash
python -B scripts/bench_group_kfold.py \
    --fortml ../fortml --output results/group_kfold.csv
```

See [`results/GROUP_KFOLD.md`](results/GROUP_KFOLD.md). The CUDA row is an
explicit capability refusal because splitters own host index metadata only.

### Weighted softmax-training lane

The weighted softmax-training lane checks the packed FortOpt value, gradient,
and exact joint L2 HVP from `softmax_training_objective_t` against an
independent NumPy multinomial cross-entropy oracle. It also exercises bounded
FortOpt L-BFGS-B and records the release-app CPU fit plus a typed CUDA refusal:

```bash
python -B scripts/bench_softmax_training.py \
  --fortml ../fortml --output results/softmax_training.csv
```

The CSV retains componentwise value/gradient/HVP errors and the explicit
`FORTNUM_NOT_IMPLEMENTED` device record; no CPU result is relabeled as GPU
work.

The shared binary GP likelihood lane separately checks signed-margin logistic
and probit value/JVP/VJP products, including a stable negative-probit tail and
an independent adjoint oracle:

```bash
.venv/bin/python -B scripts/bench_gp_likelihood.py \
  --fortml ../fortml --output results/gp_likelihood.csv
```

See [`results/GP_LIKELIHOOD.md`](results/GP_LIKELIHOOD.md). The CSV retains
FortML rows only when the complete scalar release protocol agrees with the
NumPy oracle; missing compiler/app output is explicit `unavailable`. These
host likelihood timings do not imply end-to-end GPU GP performance.

The typed MLP schedule lane independently checks constant, warm-up, cosine,
warm-up-plus-cosine, exponential-decay, and one-cycle values and analytic
products (including one-cycle peak/final-rate tangents):

```bash
.venv/bin/python -B scripts/bench_mlp_schedules.py \
  --fortml ../fortml --output results/mlp_schedules.csv
```

See [`results/MLP_SCHEDULES.md`](results/MLP_SCHEDULES.md). CUDA schedule rows
are explicit `unavailable` capability records, not host timings.

The dense MLP activation lane checks the packed `8-32-4` forward path for
linear, `tanh`, ReLU, GELU, SiLU, ELU, softplus, leaky ReLU, sigmoid, and Mish
against an independent NumPy checksum oracle. Sigmoid uses a branch-stable
reference at extreme logits; sigmoid and Mish remain explicit CUDA capability
refusals until resident kernels are linked:

```bash
.venv/bin/python -B scripts/bench_mlp_activations.py \
    --fortml ../fortml --output results/mlp_activations.csv
```

See [`results/MLP_ACTIVATIONS.md`](results/MLP_ACTIVATIONS.md). Every CUDA
row is an explicit `unavailable` capability record until resident MLP
activation and dense-gradient kernels are linked.

The scheduled trajectory hypergradient lane checks a complete `3-8-1` MLP
training trajectory with exact reverse gradients and a directional JVP over
base-rate, L2, minimum-fraction, and decay-logit hyperparameters:

```bash
.venv/bin/python -B scripts/bench_mlp_schedule_hypergradient.py \
  --fortml ../fortml --output results/mlp_schedule_hypergradient.csv
```

See [`results/MLP_SCHEDULE_HYPERGRADIENT.md`](results/MLP_SCHEDULE_HYPERGRADIENT.md).
The lane records explicit CUDA and outer-hyper-HVP refusal rows rather than
silently falling back to host finite differences.

Binary probability calibration (positive temperature scaling, Platt sigmoid,
and weighted PAVA isotonic) is checked against an independent NumPy oracle:

```bash
.venv/bin/python -B scripts/bench_probability_calibration.py \
    --fortml ../fortml --output results/probability_calibration.csv
```

See [`results/PROBABILITY_CALIBRATION.md`](results/PROBABILITY_CALIBRATION.md).
CUDA capability rows are explicit refusals because no resident calibration
kernel is linked.

The calibration-aware binary cross-validation lane fits independent logistic
models on three stratified folds, calibrates held-out margins with positive
temperature scaling, then refits the deployment model on all rows. Its
independent NumPy oracle replays the packed deployment vector and checks the
out-of-fold diagnostics:

```bash
python3 -B scripts/bench_calibrated_logistic_cv.py \
    --fortml ../fortml --output results/calibrated_logistic_cv.csv
```

See [`results/CALIBRATED_LOGISTIC_CV.md`](results/CALIBRATED_LOGISTIC_CV.md).
The CUDA row is an explicit typed refusal until a resident logistic and
calibration graph is linked.

The multiclass calibration-aware cross-validation lane fits independent
softmax models on three stratified folds, calibrates held-out logits with one
positive temperature, then refits the deployment model.  A NumPy oracle
replays the packed coefficient/intercept/temperature vector and checks every
probability, prediction, and OOF diagnostic:

```bash
python3 -B scripts/bench_calibrated_softmax_cv.py \
    --fortml ../fortml --output results/calibrated_softmax_cv.csv
```

See [`results/CALIBRATED_SOFTMAX_CV.md`](results/CALIBRATED_SOFTMAX_CV.md).
The CUDA row remains an explicit typed refusal until the complete resident
softmax-plus-calibration graph is linked.

The multiclass calibration lane fits one positive softmax temperature on a
192-row, three-class logit fixture. It checks sorted integer classes, every
probability and prediction row, and the fitted temperature against an
independent NumPy weighted softmax-NLL Newton oracle before retaining timing:

```bash
python -B scripts/bench_multiclass_probability_calibration.py \
    --fortml ../fortml --output results/multiclass_probability_calibration.csv
```

See [`results/MULTICLASS_CALIBRATION.md`](results/MULTICLASS_CALIBRATION.md).
The CUDA row remains an explicit refusal until a resident calibration kernel
is linked.

The reliability-diagram lane checks weighted equal-width confidence bins,
including empty-bin zero conventions and deterministic first-maximum ties,
against an independent NumPy oracle before retaining timing:

```bash
python3 -B scripts/bench_reliability_diagram.py \
    --fortml ../fortml --output results/reliability_diagram.csv
```

See [`results/RELIABILITY_DIAGRAM.md`](results/RELIABILITY_DIAGRAM.md). The
CUDA row is an explicit capability refusal because no resident metric kernel
is linked.

The composable physics-residual lane checks four weighted affine PINN-style
terms plus an independent nonlinear reverse-over-forward HVP fixture. The
FortML gate covers the FortOpt adapter, malformed-input refusals, the exact
HVP callback path, and the typed refusal retained when a provider does not
register an HVP callback:

```bash
python3 -B scripts/bench_physics_objective.py \
    --fortml ../fortml --output results/physics_objective.csv
```

See [`results/PHYSICS_OBJECTIVE.md`](results/PHYSICS_OBJECTIVE.md). The
objective is callback-based and has no built-in resident CUDA dispatch, so
CUDA remains an explicit capability row until a resident adapter is linked.

The bounded PINN adapter lane wraps the same four-term objective and checks an
independent manufactured affine solution, nonlinear exact HVP, bounded
FortOpt L-BFGS-B fit, and typed CUDA refusal:

```bash
python3 -B scripts/bench_pinn.py \
    --fortml ../fortml --output results/pinn.csv
```

See [`results/PINN.md`](results/PINN.md). The CPU result is a correctness
gate, not a resident GPU performance claim.

The general nonseparable Hamiltonian lane checks an independent analytic
canonical vector-field/Jacobian oracle, full-state FortML JVP/VJP products, the
separable symplectic gate, and the typed refusal for applying split leapfrog to
general `H(q,p)`:

```bash
python3 -B scripts/bench_hamiltonian_general.py \
    --fortml ../fortml --output results/hamiltonian_general.csv
```

See [`results/HAMILTONIAN_GENERAL.md`](results/HAMILTONIAN_GENERAL.md). The
CUDA/OpenACC row remains explicitly unavailable until a resident model,
derivative, and implicit-integrator graph exists.

The matched multinomial softmax and multiclass neural-classifier lane uses an
independent NumPy damped-Newton/Adam oracle, scikit-learn and optional resident
PyTorch context rows, and an explicit FortML app protocol. Missing FortML
targets or optional dependencies are recorded as machine-readable
`unavailable` rows:

```bash
.venv/bin/python -B scripts/bench_classification_models.py \
    --fortml ../fortml --output results/classification_models.csv
```

See [`results/CLASSIFICATION_MODELS.md`](results/CLASSIFICATION_MODELS.md) for
the fixture, oracle tolerances, and required release-app records.

The relaxed Bernoulli Naive Bayes lane uses a complete NumPy likelihood,
log-softmax, and input-JVP oracle on sorted arbitrary labels.  It records a
scikit-learn `BernoulliNB` context row and matched FortML fit, predict, and
input-JVP rows through the release app (or an explicit target refusal when a
checkout predates that app):

```bash
.venv/bin/python -B scripts/bench_bernoulli_nb.py \
    --fortml ../fortml --output results/bernoulli_naive_bayes.csv
```

See [`results/BERNOULLI_NB.md`](results/BERNOULLI_NB.md) for the fixture,
oracle boundary, and release-app protocol.

The matched Multinomial Naive Bayes lane checks token-mass smoothing,
stable probabilities, predictions, and an input JVP against an independent
NumPy oracle.  It includes a contextual scikit-learn row and a strict FortML
release-app protocol:

```bash
.venv/bin/python -B scripts/bench_multinomial_nb.py \
    --fortml ../fortml --output results/multinomial_naive_bayes.csv
```

See [`results/MULTINOMIAL_NB.md`](results/MULTINOMIAL_NB.md) for the fixture,
oracle tolerances, and release-app records.

The Complement Naive Bayes lane independently checks complement counts,
positive feature weights, stable probabilities, predictions, and an input JVP.
It retains the scikit-learn prior-intercept difference as contextual evidence;
the checked-in FortML release app passes the complete oracle, while future
missing-target runs remain explicit unavailable rows:

```bash
.venv/bin/python -B scripts/bench_complement_nb.py \
  --fortml ../fortml --output results/complement_naive_bayes.csv
```

See [`results/COMPLEMENT_NB.md`](results/COMPLEMENT_NB.md) for the contract.

The integer one-hot lane checks sorted categories, packed offsets, missing and
unknown policies, and complete dense transforms against an independent NumPy
oracle and scikit-learn context.  Categorical JVP/VJP are explicit refusals:

```bash
.venv/bin/python -B scripts/bench_one_hot_encoder.py \
  --fortml ../fortml --output results/one_hot_encoder.csv
```

See [`results/ONE_HOT_ENCODER.md`](results/ONE_HOT_ENCODER.md) for the release
target protocol and derivative boundary.

The MLP training, composable polynomial/Fourier basis pipeline, deterministic
decision stump, depth-limited CART regression and classification, core
regression metrics, and residual-stump gradient-boosting lanes are in
[`results/FEATURES.md`](results/FEATURES.md). Run them with:

```bash
.venv/bin/python -B scripts/bench_features.py \
    --fortml ../fortml --output results/features_workloads.csv
```

The harness checks complete values against independent NumPy implementations
before timing. It adds a matched scikit-learn reference where applicable and
records explicit availability/refusal rows for optional PyTorch, JAX, and
XGBoost comparisons. It also records explicit FortML CUDA capability refusals
for host-only GaussianNB, MLP-training, and logistic-objective paths. These
rows contain no timing and never relabel a CPU run as device evidence.

The production trainer/preprocessing lane separately benchmarks full-batch MLP
momentum SGD and Nesterov updates plus the differentiable mean, median, and
constant simple imputer.  It checks complete predictions, losses, fitted
statistics, transforms, JVPs, and VJPs against independent NumPy formulas:

```bash
.venv/bin/python -B scripts/bench_training.py \
  --fortml ../fortml --output results/training_imputer.csv
```

See [`results/TRAINING_IMPUTER.md`](results/TRAINING_IMPUTER.md).  The lane is
CPU-only until FortML exposes a device-resident trainer/imputer path; no CPU
timing is relabeled as CUDA evidence.

The dense missing-indicator lane checks both `all` and `missing-only` fit-time
column policies against an independent NumPy NaN-mask oracle. It compares
every transform entry and exact zero JVP/VJP product, records release CPU
timings, and retains typed CUDA refusals:

```bash
python3 scripts/bench_missing_indicator.py \
  --fortml ../fortml --output results/missing_indicator.csv
```

See [`results/MISSING_INDICATOR.md`](results/MISSING_INDICATOR.md). Sparse
CSR/CSC views and resident device kernels remain separate work packages.

The sparse preprocessing lane checks sparse-safe CSC standard scaling against
an independent dense NumPy expansion. It compares transformed and inverse
stored values plus exact value JVP/VJP products, records the implicit-zero
statistics contract, and retains a typed CUDA refusal until a resident sparse
transform kernel is linked:

```bash
python3 scripts/bench_sparse_preprocessing.py \
  --fortml ../fortml --output results/sparse_preprocessing.csv
```

See [`results/SPARSE_PREPROCESSING.md`](results/SPARSE_PREPROCESSING.md).

The AdamW and fixed-trajectory MLP hypergradient lanes use separate release
contracts.  NumPy independently checks the AdamW moments/decoupled decay and
the validation objective, log-hyperparameter finite-difference gradient, and
directional JVP before any FortML timing is retained.  Missing release targets
are explicit `unavailable` rows:

```bash
.venv/bin/python -B scripts/bench_neural_training.py \
  --fortml ../fortml \
  --adamw-output results/adamw_training.csv \
  --hypergradient-output results/mlp_hypergradient.csv
```

See [`results/ADAMW_HYPERGRADIENT.md`](results/ADAMW_HYPERGRADIENT.md) for
the fixture, recurrence, finite-difference oracle, and release-app protocol.

The five-parameter AdamW beta-logit lane independently checks the full
trajectory value, all five hypergradient components, and a directional JVP.
FortML has no complete-array release app for this new objective yet, so its
rows remain explicit `unavailable` records rather than inferred timings:

```bash
.venv/bin/python -B scripts/bench_adamw_beta_hypergradient.py \
  --fortml ../fortml --output results/adamw_beta_hypergradient.csv
```

See [`results/ADAMW_BETA_HYPERGRADIENT.md`](results/ADAMW_BETA_HYPERGRADIENT.md).

The centered dense PCA lane compares the NumPy thin-SVD oracle with a
scikit-learn full-SVD context and the FortML release app. It checks centering,
deterministic sign alignment, rank ordering, and component orthonormality
before retaining timings:

```bash
.venv/bin/python -B scripts/bench_pca.py \
  --fortml ../fortml --output results/pca.csv
```

See [`results/PCA.md`](results/PCA.md). The current FortML release app exports
an orthonormality guard and fit timing; complete fitted-array export remains an
explicit follow-up boundary in the report.

The PCA-initialized tied linear-autoencoder lane uses the same centered
`512 x 16` fixture and rank eight. An independent NumPy thin-SVD reconstruction
RMSE is compared with `fortml_bench_linear_autoencoder`; CUDA is recorded as an
explicit refusal until a resident matrix-product lowering exists:

```bash
.venv/bin/python -B scripts/bench_linear_autoencoder.py \
  --fortml ../fortml --output results/linear_autoencoder.csv
```

See [`results/LINEAR_AUTOENCODER.md`](results/LINEAR_AUTOENCODER.md).

The weighted ridge lane checks the closed-form multi-output fit, vector and
matrix prediction, and packed coefficient/input JVP and VJP products against
an independent NumPy implementation. It times complete NumPy operations only
after finite-difference and adjoint checks pass. A complete-array FortML
release target is required before a FortML timing is retained; until then all
seven FortML operations are explicit `unavailable` rows:

```bash
.venv/bin/python -B scripts/bench_ridge.py \
  --fortml ../fortml --output results/ridge.csv
```

See [`results/RIDGE.md`](results/RIDGE.md) for the weighted fixture, strict
release protocol, and derivative boundary.

The weighted elastic-net lane uses the same complete-call contract for a
multi-output coordinate-descent fit with an L1/L2 penalty.  An independent
NumPy solver checks every fitted coefficient, prediction, packed-parameter
JVP, parameter VJP, and input VJP before retaining timings:

```bash
.venv/bin/python -B scripts/bench_elastic_net.py \
  --fortml ../fortml --output results/elastic_net.csv
```

See [`results/ELASTIC_NET.md`](results/ELASTIC_NET.md) for the fixture,
strict complete-array protocol, and fixed-fit derivative boundary.

The Adagrad lane independently checks the accumulated-square recurrence and
interrupted-versus-uninterrupted state resume. FortML's trainer target is
recorded as `unavailable` until a release app exports the same state, with no
other optimizer timing substituted:

```bash
.venv/bin/python -B scripts/bench_adagrad.py \
  --fortml ../fortml --output results/adagrad.csv
```

See [`results/ADAGRAD.md`](results/ADAGRAD.md).

The RMSprop lane checks both the canonical FortOpt state recurrence and the
centered, momentum-enabled MLP trainer path against independent NumPy updates.
The release app exports separate direct-optimizer and MLP rows; a missing
target is retained as an explicit refusal rather than substituting an Adam or
Adagrad timing:

```bash
.venv/bin/python -B scripts/bench_rmsprop.py \
  --fortml ../fortml --output results/rmsprop.csv
```

See [`results/RMSPROP.md`](results/RMSPROP.md).

FortML also ships a resident native-CUDA RMSprop state kernel for direct
device-resident gradients. `../fortml/test/run_cuda_rmsprop_state.sh` checks
its centered recurrence against an independent CPU oracle. The benchmark CSV
does not claim a CUDA timing for the full MLP trainer until gradient assembly,
transfer accounting, and matched device timing are included.

The resident RMSprop state gate also checks malformed creation arguments,
null gradient/download boundaries, and the explicit create/step/download/
destroy lifecycle. Its independent recurrence and typed refusal row are
recorded separately from the CPU MLP trainer lane:

```bash
python3 scripts/bench_cuda_rmsprop.py \
  --fortml ../fortml --output results/cuda_rmsprop.csv
```

See [`results/CUDA_RMSPROP.md`](results/CUDA_RMSPROP.md). This is a
no-autodiff optimizer-state contract, not end-to-end resident MLP training.

The fixed full-batch RMSprop hypergradient lane independently finite-differences
the value, all five packed hyperparameters, and a directional JVP for separate
centered and uncentered trajectories. The FortML release app is retained only after its
value, gradient, and JVP products pass the NumPy oracle. The no-autodiff
optimizer state has a separate resident CUDA oracle, while CUDA remains an
explicit refusal for the complete MLP HVP trajectory:

```bash
.venv/bin/python -B scripts/bench_rmsprop_hypergradient.py \
  --fortml ../fortml --output results/rmsprop_hypergradient.csv
```

See [`results/RMSPROP_HYPERGRADIENT.md`](results/RMSPROP_HYPERGRADIENT.md).

The fixed full-batch Adagrad hypergradient lane independently finite-differences
the validation value, all three packed log hyperparameters, and a directional
JVP for the accumulated-square trajectory. The FortML release app is retained
only after its complete value/gradient/JVP array agrees with the NumPy oracle;
the raw record includes CPU timing, repository/compiler provenance, and typed
CUDA refusals:

```bash
.venv/bin/python -B scripts/bench_adagrad_hypergradient.py \
  --fortml ../fortml --output results/adagrad_hypergradient.csv
```

See [`results/ADAGRAD_HYPERGRADIENT.md`](results/ADAGRAD_HYPERGRADIENT.md).

The fixed seeded mini-batch SGD hypergradient lane checks the complete
value/gradient/JVP array against an independent NumPy trajectory with the same
Park–Miller shuffle cursor, then retains CPU timing and typed CUDA refusals:

```bash
.venv/bin/python -B scripts/bench_mlp_minibatch_hypergradient.py \
    --fortml ../fortml --output results/mlp_minibatch_hypergradient.csv
```

See [`results/MLP_MINIBATCH_HYPERGRADIENT.md`](results/MLP_MINIBATCH_HYPERGRADIENT.md).

The fixed seeded mini-batch coupled-L2 Adam hypergradient lane independently
replays first/second moments, bias correction, and validation MSE with a NumPy
recurrence. The FortML release app is retained only after value, both packed
gradient components, and a directional JVP agree; CPU timing and typed CUDA
refusals are recorded:

```bash
python3 -B scripts/bench_mlp_minibatch_adam_hypergradient.py \
    --fortml ../fortml --output results/mlp_minibatch_adam_hypergradient.csv
```

See [`results/MLP_MINIBATCH_ADAM_HYPERGRADIENT.md`](results/MLP_MINIBATCH_ADAM_HYPERGRADIENT.md).

The calibrated neural classifier lane checks sorted labels, finite calibrated
probabilities, the probability simplex, and prediction-domain invariants on a
64-row deterministic fixture before retaining CPU fit/predict timings:

```bash
.venv/bin/python -B scripts/bench_mlp_calibrated_classifier.py \
    --fortml ../fortml --output results/mlp_calibrated_classifier.csv
```

See [`results/MLP_CALIBRATED_CLASSIFIER.md`](results/MLP_CALIBRATED_CLASSIFIER.md).

The fixed full-batch SGD momentum hypergradient lane uses the same fixture and
checks the exact classical-momentum recurrence, all three packed
`[log_learning_rate, log_l2, momentum]` gradient components, and a directional
JVP against an independent central-finite-difference NumPy oracle:

```bash
.venv/bin/python -B scripts/bench_sgd_momentum_hypergradient.py \
    --fortml ../fortml --output results/sgd_momentum_hypergradient.csv
```

See [`results/SGD_MOMENTUM_HYPERGRADIENT.md`](results/SGD_MOMENTUM_HYPERGRADIENT.md).

The fixed full-batch Lion hypergradient lane checks the piecewise-smooth
momentum/sign recurrence, all four packed `[log_learning_rate, log_l2,
logit(beta1), logit(beta2)]` components, and a directional JVP against an
independent central-finite-difference NumPy trajectory. Near-zero sign
branches are refused explicitly; CUDA is recorded as unavailable until the
complete model and optimizer state are resident:

```bash
.venv/bin/python -B scripts/bench_mlp_lion_hypergradient.py \
    --fortml ../fortml --output results/mlp_lion_hypergradient.csv
```

See [`results/MLP_LION_HYPERGRADIENT.md`](results/MLP_LION_HYPERGRADIENT.md).

The optimizer-group trajectory lane checks the production trainer's
post-SGD group scaling and all four packed `[log_learning_rate, log_l2,
log_multiplier_1, log_multiplier_2]` products against an independent NumPy
central-difference trajectory. It records CPU timing only after the complete
value/gradient/JVP array agrees; overlapping ranges and CUDA are explicit
refusals:

```bash
.venv/bin/python -B scripts/bench_mlp_optimizer_group_hypergradient.py \
    --fortml ../fortml --output results/mlp_optimizer_group_hypergradient.csv
```

See [`results/MLP_OPTIMIZER_GROUP_HYPERGRADIENT.md`](results/MLP_OPTIMIZER_GROUP_HYPERGRADIENT.md).

The deterministic mini-batch SGD hypergradient lane records a private seeded
batch cursor and checks validation MSE, both packed `[log_learning_rate,
log_l2]` gradient components, and a directional JVP against an independent
NumPy central-difference trajectory before retaining the FortML timing:

```bash
.venv/bin/python -B scripts/bench_mlp_minibatch_hypergradient.py \
    --fortml ../fortml --output results/mlp_minibatch_hypergradient.csv
```

See [`results/MLP_MINIBATCH_HYPERGRADIENT.md`](results/MLP_MINIBATCH_HYPERGRADIENT.md).

The resident CUDA contract lane independently checks the native kNN prediction
plan, the resident dense-affine inference primitive (all eight MLP activations
and their forward-mode JVP/reverse-mode VJP), its single-layer tanh MSE update,
and the no-autodiff RMSprop optimizer-state kernel. NumPy computes the expected
labels, activation and activation-JVP checksums, MSE loss/gradient/parameter
update, transfer-counter lower bounds, and five-step centered recurrence before
the native gates run. These are correctness rows, not timings for a complete
estimator or trainer; missing CUDA toolchains/devices remain explicit `skipped`
records:

```bash
.venv/bin/python -B scripts/bench_device_contracts.py \
  --fortml ../fortml --output results/device_contracts.csv
```

See [`results/DEVICE_CONTRACTS.md`](results/DEVICE_CONTRACTS.md) for the exact
fixtures, GPU/toolchain metadata, and remaining resident-workload boundaries.

The joint basis-pipeline training lane checks the packed Fourier basis and
linear coefficient objective, including value/JVP/HVP finite differences, the
exact optimized-ridge coordinate/mixed products, and the typed CUDA refusal:

```bash
python -B scripts/bench_basis_pipeline_training.py \
  --fortml ../fortml --output results/basis_pipeline_training.csv
```

See [`results/BASIS_PIPELINE_TRAINING.md`](results/BASIS_PIPELINE_TRAINING.md).

The named fan-out/fan-in basis-DAG lane checks an independent two-branch
quadratic/spectral feature construction, branch metadata and packed offsets,
JVP/VJP duality, central-difference HVPs, CPU dispatch, and the typed CUDA
refusal. Its wall time is the complete correctness gate, not a synthetic
throughput claim:

```bash
python -B scripts/bench_basis_fanout_pipeline.py \
  --fortml ../fortml --output results/basis_fanout_pipeline.csv
```

See [`results/BASIS_FANOUT_PIPELINE.md`](results/BASIS_FANOUT_PIPELINE.md).

The scheduled-Adagrad lane checks exact fixed-trajectory value/gradient/JVP/VJP
products, FortOpt L-BFGS-B integration, default schedule validation, malformed
options, and the typed CUDA refusal:

```bash
python -B scripts/bench_mlp_adagrad_schedule_hypergradient.py \
  --fortml ../fortml \
  --output results/mlp_adagrad_schedule_hypergradient.csv
```

See [`results/MLP_ADAGRAD_SCHEDULE_HYPERGRADIENT.md`](results/MLP_ADAGRAD_SCHEDULE_HYPERGRADIENT.md).

The shared objective-trainer and XGBoost additive-contribution correctness
lane runs independent Fortran behavioral gates and an independent NumPy
quadratic update oracle. Its wall times are correctness-gate timings, not
throughput measurements:

```bash
python -B scripts/bench_training_core.py \
  --fortml ../fortml --output results/training_core.csv
```

See [`results/TRAINING_CORE.md`](results/TRAINING_CORE.md).

The resident CUDA AdamW state lane wraps
`../fortml/test/run_cuda_adamw_state.sh` and independently reconstructs its
seven-step bias-corrected recurrence and decoupled weight decay in NumPy. A
native pass requires the reported maximum error to be at most `3e-13`.
Compilation/device absence is recorded as `unavailable`; the gate's
compile-inclusive subprocess duration is not presented as a kernel timing:

```bash
.venv/bin/python -B scripts/bench_cuda_adamw.py \
  --fortml ../fortml --output results/cuda_adamw.csv
```

See [`results/CUDA_ADAMW.md`](results/CUDA_ADAMW.md). This is a resident
no-autodiff optimizer-state contract, not full MLP gradient or hypergradient
GPU evidence.

The matching resident CUDA Adagrad state lane independently checks the
canonical accumulated-square recurrence and device-resident gradient ABI:

```bash
.venv/bin/python -B scripts/bench_cuda_adagrad.py \
  --fortml ../fortml --output results/cuda_adagrad.csv
```

See [`results/CUDA_ADAGRAD.md`](results/CUDA_ADAGRAD.md). Compilation or
device absence is an explicit `unavailable` row; the gate has no kernel timing
claim and does not imply resident MLP gradient or hypergradient training.

The dense k-nearest-neighbor lane checks sorted arbitrary integer classes,
stable original-row distance ties, uniform and inverse-distance weighting,
complete probability checksums, and predicted labels with an independent
NumPy oracle:

```bash
.venv/bin/python -B scripts/bench_knn.py \
  --fortml ../fortml --output results/knn.csv
```

See [`results/KNN.md`](results/KNN.md). Neighbor selection is discrete, so
input JVP/VJP refusal is recorded by the FortML unit contract rather than
inventing a derivative timing.

The GP-classification training lane checks the bounded binary and shared-kernel
one-vs-rest L-BFGS-B adapters against an independent NumPy Laplace-mode and
envelope-gradient recurrence. It records mode-log-posterior training only,
not full Laplace evidence:

```bash
.venv/bin/python -B scripts/bench_gp_classification_training.py \
  --fortml ../fortml --output results/gp_classification_training.csv
```

See [`results/GP_CLASSIFICATION_TRAINING.md`](results/GP_CLASSIFICATION_TRAINING.md).

The weighted Laplace-GP lane extends binary and one-vs-rest fitting with
nonnegative row weights, including zero-weight rows. Its independent NumPy
oracle checks the weighted mode log posterior and kernel envelope gradient
against refitted finite differences; the FortML gate covers logistic/probit,
OVR composition, malformed weights, and the explicit CUDA boundary:

```bash
python3 -B scripts/bench_gp_classification_sample_weights.py \
  --fortml ../fortml --output results/gp_classification_sample_weights.csv
```

See [`results/GP_CLASSIFICATION_SAMPLE_WEIGHTS.md`](results/GP_CLASSIFICATION_SAMPLE_WEIGHTS.md).

The inducing-point Bernoulli variational-GP lane checks a dense two-inducing
point ELBO and packed variational-parameter gradient against an independent
NumPy finite-difference oracle, then runs FortML's seeded Monte Carlo,
parameter/query JVP and VJP, minibatch, malformed-label, and bounded
L-BFGS-B optimizer tests. CUDA is an explicit typed refusal until
the inducing solve, likelihood table, and reduction are resident:

```bash
python3 -B scripts/bench_gp_variational_classification.py \
  --fortml ../fortml --output results/gp_variational_classification.csv
```

See [`results/GP_VARIATIONAL_CLASSIFICATION.md`](results/GP_VARIATIONAL_CLASSIFICATION.md).

The weighted variational-GP lane extends that gate with nonuniform sample
weights shared by binary and one-vs-rest heads. Its independent NumPy oracle
checks that uniform weights scale only the expected likelihood, verifies the
packed weighted gradient by finite differences, and records malformed-weight
refusals plus the typed CUDA boundary:

```bash
python3 -B scripts/bench_gp_variational_classification_weights.py \
  --fortml ../fortml --output results/gp_variational_classification_weights.csv
```

See [`results/GP_VARIATIONAL_CLASSIFICATION_WEIGHTS.md`](results/GP_VARIATIONAL_CLASSIFICATION_WEIGHTS.md).

The coupled categorical variational-GP lane checks the variance-corrected
shared-softmax ELBO, packed and query products, FortOpt fitting, and typed CUDA
refusal against independent NumPy and finite-difference oracles:

```bash
python3 -B scripts/bench_gp_variational_categorical.py \
  --fortml ../fortml --output results/gp_variational_categorical.csv
```

See [`results/GP_VARIATIONAL_CATEGORICAL.md`](results/GP_VARIATIONAL_CATEGORICAL.md).

The XGBoost-style lane has its own workload and raw record. It checks squared,
binary logistic, one-vs-rest multiclass, and learned-NaN default-direction
depth-two boosting against independent recursive NumPy gradient/Hessian
oracles. It also checks weighted CPU histogram growth for regression, binary
logistic, and multiclass OVR (`tree_method="hist"`, `max_bin=2`) against a
weighted-quantile NumPy oracle. Native CUDA histogram growth and LightGBM
leaf-wise policies remain explicit capability rows:

```bash
.venv/bin/python -B scripts/bench_xgboost.py \
  --fortml ../fortml --output results/xgboost_workloads.csv
```

See [`results/XGBOOST.md`](results/XGBOOST.md) for the regularisation settings,
weighted-histogram fixture, oracle boundary, and recorded timings. The
optional package row never turns a different histogram or tree-growth policy
into a bitwise comparison.

The interaction-constraint lane fits an unconstrained and a separated-group
depth-two tree. Its NumPy oracle replays the group-mean split and path mask,
checks the complete prediction vector and fitted diagnostics, and records a
typed CUDA refusal because no resident XGBoost tree kernel is linked:

```bash
python3 -B scripts/bench_xgboost_interaction.py \
  --fortml ../fortml --output results/xgboost_interaction.csv
```

See [`results/XGBOOST_INTERACTION.md`](results/XGBOOST_INTERACTION.md).

The binary AdaBoost lane uses weighted depth-one CART learners. Its independent
NumPy oracle reconstructs the one-stump error, learner weight, signed margin,
probabilities, and labels. The CPU rows are retained only after the complete
vector check, and CUDA is recorded as a typed refusal:

```bash
python3 -B scripts/bench_adaboost_classifier.py \
  --fortml ../fortml --output results/adaboost_classifier.csv
```

See [`results/ADABOOST_CLASSIFIER.md`](results/ADABOOST_CLASSIFIER.md).

The multiclass SAMME lane fits one weighted depth-one CART learner over three
sorted arbitrary integer labels. Its independent NumPy oracle reconstructs
the SAMME stage weight, weighted-vote margins, stabilized softmax, and labels;
the CPU rows are retained only after every value matches, and CUDA is recorded
as a typed refusal:

```bash
python3 -B scripts/bench_adaboost_samme.py \
  --fortml ../fortml --output results/adaboost_samme.csv
```

See [`results/ADABOOST_SAMME.md`](results/ADABOOST_SAMME.md).

The bagging lane exercises the seeded bootstrap and without-replacement CART
ensemble with three integer classes. Its independent NumPy oracle checks the
cluster labels and probability simplex, while the release app records fit and
prediction timings and an explicit CUDA refusal:

```bash
python3 -B scripts/bench_bagging_classifier.py \
  --fortml ../fortml --output results/bagging_classifier.csv
```

See [`results/BAGGING_CLASSIFIER.md`](results/BAGGING_CLASSIFIER.md).

The column-basis device lane checks a named Fourier/polynomial feature union
against an independent NumPy oracle. It records the CPU transform timing and
the explicit CUDA refusal from the typed device API; no host transform is
reported as accelerator evidence:

```bash
python3 -B scripts/bench_column_pipeline_device.py \
  --fortml ../fortml --output results/column_pipeline_device.csv
```

See [`results/COLUMN_PIPELINE_DEVICE.md`](results/COLUMN_PIPELINE_DEVICE.md).

The separately named LightGBM-style lane records weighted regression and
binary-logistic histogram boosting with deterministic globally best-leaf growth
up to `num_leaves`. The release app also gates cumulative staged predictions,
additive base-plus-tree contributions, transactional fitted-prefix slicing, and
versioned text save/load with trailing-record refusal. The release app also
continues a weighted four-tree prefix to eight trees and checks all staged
outputs against an independently fitted eight-tree model, including a typed
non-growing-target refusal.
The six-sample weighted-Newton fixture is an independent oracle; CPU timing
and the typed CUDA refusal are kept in a dedicated CSV:

```bash
.venv/bin/python -B scripts/bench_lightgbm.py \
  --fortml ../fortml --output results/lightgbm_leafwise.csv
```

See [`results/LIGHTGBM_LEAFWISE.md`](results/LIGHTGBM_LEAFWISE.md). GOSS,
EFB, categorical statistics, distributed workers, and resident GPU histograms
remain explicit follow-up gaps.

The LightGBM validation lane independently replays the one-feature Newton
recurrence for regression and binary logistic objectives. It checks patience,
best-round metadata, restore-best versus retain-all ensembles, malformed
validation refusal, and the explicit CUDA refusal:

```bash
python -B scripts/bench_lightgbm_early_stopping.py \
  --fortml ../fortml --output results/LIGHTGBM_EARLY_STOPPING.md
```

See [`results/LIGHTGBM_EARLY_STOPPING.md`](results/LIGHTGBM_EARLY_STOPPING.md).
The same release protocol now checks regression margins, binary positive-class
probabilities, multiclass simplex probabilities, staged raw margins, and
normalized gain feature importance after every boosting stage.

The Poisson count-objective lane independently reconstructs a one-tree
log-link Newton fixture and then records exact CPU fit/predict and weighted
histogram timings on a larger deterministic count workload:

```bash
.venv/bin/python -B scripts/bench_xgboost_poisson.py \
  --fortml ../fortml --output results/xgboost_poisson.csv
```

See [`results/XGBOOST_POISSON.md`](results/XGBOOST_POISSON.md). Its CUDA row
is an explicit unavailable refusal until a resident tree kernel is linked.

The fixed-shape Gamma objective lane independently reconstructs the positive
log-link one-tree Newton update, then records exact CPU fit/predict and
weighted-histogram timings on a strictly positive workload:

```bash
python -B scripts/bench_xgboost_gamma.py \
  --fortml ../fortml --output results/xgboost_gamma.csv
```

See [`results/XGBOOST_GAMMA.md`](results/XGBOOST_GAMMA.md). Its CUDA row is
an explicit unavailable refusal until a resident tree kernel is linked.

The squared-log (RMSLE) objective lane independently reconstructs a one-tree
Newton update in the `log1p(target)` coordinate, then records exact CPU
fit/predict and a weighted-histogram diagnostic on a deterministic
nonnegative workload:

```bash
python -B scripts/bench_xgboost_squared_log.py \
  --fortml ../fortml --output results/xgboost_squared_log.csv
```

See [`results/XGBOOST_SQUARED_LOG.md`](results/XGBOOST_SQUARED_LOG.md). Its
CUDA row remains an explicit unavailable refusal until a resident tree kernel
is linked.

The XGBoost validation lane independently replays every depth-one Newton
stage for squared, binary logistic, and squared-log objectives. It checks
two-round patience, `restore_best` versus retaining the stopped ensemble,
best-validation metadata, and the typed malformed-validation refusal:

```bash
python -B scripts/bench_xgboost_early_stopping.py \
  --fortml ../fortml --output results/XGBOOST_EARLY_STOPPING.md
```

See [`results/XGBOOST_EARLY_STOPPING.md`](results/XGBOOST_EARLY_STOPPING.md).
The lane is CPU correctness evidence; tree CUDA remains an explicit typed
unavailable contract.

The sampling lane checks deterministic without-replacement row and feature
subsets, exact depth-one Newton gains, and predictions against an independent
NumPy oracle. CUDA remains an explicit typed refusal until resident tree
kernels are linked:

```bash
python -B scripts/bench_xgboost_sampling.py \
  --fortml ../fortml --output results/xgboost_sampling.csv
```

See [`results/XGBOOST_SAMPLING.md`](results/XGBOOST_SAMPLING.md).

The XGBoost persistence lane fits a validation-aware four-tree ensemble,
round-trips its versioned text state, and independently parses and walks every
serialized node. Predictions, raw margins, staged outputs, missing-routing
metadata, and monotone constraints are checked before timings are retained;
resident tree CUDA remains an explicit typed refusal:

```bash
python -B scripts/bench_xgboost_serialization.py \
  --fortml ../fortml --output results/xgboost_serialization.csv
```

See [`results/XGBOOST_SERIALIZATION.md`](results/XGBOOST_SERIALIZATION.md).

The robust XGBoost lane independently reconstructs weighted one-tree Huber and
pinball/quantile objectives, including base margins and leaf corrections:

```bash
python -B scripts/bench_xgboost_robust.py \
  --fortml ../fortml --output results/xgboost_robust.csv
```

See [`results/XGBOOST_ROBUST.md`](results/XGBOOST_ROBUST.md). CUDA remains a
typed unavailable row until a resident robust-tree kernel is linked.

The absolute-deviation XGBoost lane checks the weighted-median identity-link
base margin and one-tree L1 Newton corrections against an independent NumPy
oracle. CPU fit/predict timings and the typed CUDA refusal are recorded:

```bash
python -B scripts/bench_xgboost_absolute.py \
  --fortml ../fortml --output results/xgboost_absolute.csv
```

See [`results/XGBOOST_ABSOLUTE.md`](results/XGBOOST_ABSOLUTE.md).

The XGBoost derivative lane checks query JVP and VJP products against an
independent NumPy central-difference oracle away from learned split surfaces.
It also checks the typed derivative-domain refusal on a split and the explicit
CUDA capability refusal:

```bash
.venv/bin/python -B scripts/bench_xgboost_derivatives.py \
  --fortml ../fortml --output results/xgboost_derivatives.csv
```

See [`results/XGBOOST_DERIVATIVES.md`](results/XGBOOST_DERIVATIVES.md).

The fitted-ensemble slicing lane independently replays a deterministic
three-stump fixture in NumPy and checks that a two-tree prefix equals the
source's staged prediction while remaining distinct from the full model. It
also checks the typed invalid-prefix refusal:

```bash
python -B scripts/bench_xgboost_slice.py \
  --fortml ../fortml --output results/xgboost_slice.csv
```

See [`results/XGBOOST_SLICE.md`](results/XGBOOST_SLICE.md).

The warm-start lane continues a fitted two-tree prefix to four trees. An
independent NumPy Newton-stump replay gates the fourth staged margin against a
fresh four-tree fit, and the release app records transactional refusals for
changed controls, non-increasing targets, and unfitted sources:

```bash
python -B scripts/bench_xgboost_warm_start.py \
  --fortml ../fortml --output results/xgboost_warm_start.csv
```

See [`results/XGBOOST_WARM_START.md`](results/XGBOOST_WARM_START.md). The CUDA
row is an explicit unavailable capability record because warm-start
continuation has no resident CUDA entry point.

The classifier-chain lane checks sequential logistic heads, packed-parameter
NumPy replay, integer-label predictions, fit/predict timings, and an explicit
CUDA refusal:

```bash
python -B scripts/bench_classifier_chain.py \
  --fortml ../fortml --output results/classifier_chain.csv
```

See [`results/CLASSIFIER_CHAIN.md`](results/CLASSIFIER_CHAIN.md).

The generic hyperparameter-search lane uses an independent three-parameter
quadratic oracle to gate Cartesian grid, seeded random, single-start FortOpt
L-BFGS-B, and eight-start bounded L-BFGS-B timings. The random and multistart
streams are deterministic for seed `20260807` and record their evaluation
budgets:

```bash
.venv/bin/python -B scripts/bench_hyperparameter_search.py \
  --fortml ../fortml --output results/hyperparameter_search.csv
```

See [`results/HYPERPARAMETER_SEARCH.md`](results/HYPERPARAMETER_SEARCH.md).
The CUDA row is an explicit unavailable refusal until resident objective/search
state is implemented; no host search timing is relabeled as device evidence.

The scalable-model report <!-- slop-ok --> also contains the current corrected GRBCM,
contiguous-versus-clustered expert, and multidimensional-SKI records. Older
GRBCM rows are superseded because they predate the communication-set and
enhanced-expert correction. Use the raw CSVs and reproduction commands linked
from [`results/SCALABLE_GP.md`](results/SCALABLE_GP.md).

The CPU lane defaults to physical cores and can be pinned explicitly:

    CPU_THREADS=16 ./scripts/run_suite.sh

When nvfortran is available, the CPU lane uses its host OpenMP path with
nvfortran -O3 -mp. Set CPU_FC=gfortran to select the GNU Fortran fallback.

Run the five-size scaling sweep with:

    ./scripts/run_scaling.sh

It writes one merged CSV and separate log-log CPU and GPU plots under
`results/`. Dense PyTorch capacity failures are recorded as `oom` so the
matrix-free competitors remain visible at larger sizes.

For operation-level counters, run `scripts/profile_rbf_mvm.sh`. It records
CPU `perf stat` counters and NVIDIA Nsight Systems plus `NV_ACC_TIME`
reports. The Python operation profiler records the corresponding operation
tables for dense PyTorch, KeOps, and GPyTorch-KeOps. Nsight Compute is
attempted when permissions allow it.

The standalone RBF driver selects the direct nvfortran operator build for its
CPU lane when `FC=nvfortran`. This keeps the benchmark usable while the full
fpm dependency graph remains blocked by the documented FortAD 26.5 compiler
ICE. GNU and other compiler selections continue through `benchmark/run.sh`.

Run the matched matrix-free CG workload with:

    N_SAMPLES=2048 N_FEATURES=8 ./scripts/run_cg_suite.sh

The CG suite uses float64, variance 1.4, lengthscale 0.7, diagonal shift 0.08,
relative tolerance `1e-8`, and a 500-iteration cap. The Python lanes use one
explicit CG recurrence around dense PyTorch, KeOps, and GPyTorch-KeOps MVMs.
The FortML lane uses the specialized `nvfortran`/OpenACC RBF solve. Every
result is checked with a blocked NumPy residual, and sizes up to `--oracle-n`
also use an independent dense NumPy solve. Raw records and scaling plots are
written under `results/rbf_cg*`. For the full four-point sweep, run:

    ./scripts/run_cg_scaling.sh

The fused multi-right-hand-side CG workload uses four float64 right-hand sides
and one batched kernel product per iteration. Run it with:

    N_SAMPLES=2048 N_FEATURES=8 N_RHS=4 ./scripts/run_cg_multi_suite.sh

It compares the FortML multi-RHS operator with batched dense PyTorch, KeOps,
and GPyTorch-KeOps recurrences. The blocked NumPy matmat residual and the
small dense multi-RHS solve are independent correctness oracles. Results are
written to `results/rbf_cg_multi.csv`.

The detailed record and plots are in
[`results/rbf_cg_multi_scaling.md`](results/rbf_cg_multi_scaling.md).

For sample-count scaling, run:

    N_RHS=4 ./scripts/run_cg_multi_scaling.sh

This writes multi-RHS CPU and GPU log-log plots beside the merged scaling CSV.
Pass `--include-setup` to `scripts/plot_cg_multi.py` to plot first-solve time
(setup plus one solve) when a preconditioner has a one-time build.

The experimental KeOps-style Nystrom/Woodbury path is enabled with
`NYSTROM_RANK=32`. It builds a rank-32 feature factor, applies the resulting
Woodbury preconditioner inside fused multi-RHS CG, and records both setup and
steady-state solve time. The matched evidence is in
[`results/rbf_cg_multi_nystrom.md`](results/rbf_cg_multi_nystrom.md).

The `nvfortran` drivers also expose a native CUDA variant for the fixed
eight-feature RBF kernel. Set `FORTML_NATIVE_CUDA=1` to compile the shared
neighbor-tile kernel with `nvcc` and link it into the Fortran executable. The
default lane remains OpenACC. Both variants run the direct pairwise oracle in
the MVM benchmark, and the CG lane records the selected variant in its metadata.
The matmat benchmark uses the same switch and checks every right-hand side
against the direct pairwise oracle. Run `scripts/run_matmat_suite.sh` for the
CPU, OpenACC, and native CUDA lanes over one, two, four, and eight right-hand
sides. It writes `results/rbf_matmat.csv` and resident-runtime plots.

## KeOps-style static composite operator

The composite lane measures the operator used by a common GP covariance
expression without materializing its dense matrix:

\[
 y_i = \delta v_i + c\sum_j v_j + \sigma^2
 \sum_j \exp\left(-\frac{\lVert x_i-x_j\rVert^2}{2\ell^2}\right)v_j.
\]

FortML recognizes this fixed RBF-plus-constant expression and executes one
matrix-free row map/reduce, with an optional native CUDA kernel. The KeOps
lane uses a LazyTensor reduction. The GPyTorch lane uses its KeOps RBF
operator plus the same explicit constant rank-one term. Dense PyTorch is a
materialized reference. All lanes use the same deterministic float64 inputs,
constants, output oracle, and resident-versus-transfer policy. The blocked
NumPy pairwise implementation is run before every timed competitor and is an
independent behavioral oracle.

Run the matched sweep with five repetitions:

    REPETITIONS=5 ./scripts/run_composite_scaling.sh

The default CPU build selects nvfortran with physical-core OpenMP when it is
available. The GPU build uses nvfortran/OpenACC. Set `FORTML_NATIVE_CUDA=1`
for the optional `nvcc` fixed-feature kernel. To extend an existing sweep to
larger sizes, write to a separate directory and merge the resulting CSV:

    N_VALUES='8192 16384' REPETITIONS=2 \
    RESULTS_DIR=/mnt/storage/fortml-composite-high ./scripts/run_composite_scaling.sh

The released extended record and plots are in
[`results/composite_mvm_scaling_extended.csv`](results/composite_mvm_scaling_extended.csv),
[`results/composite_mvm_scaling_extended_cpu.png`](results/composite_mvm_scaling_extended_cpu.png),
[`results/composite_mvm_scaling_extended_cuda.png`](results/composite_mvm_scaling_extended_cuda.png),
and [`results/composite_mvm_scaling.md`](results/composite_mvm_scaling.md).
The high-N dense GPU failures remain in the CSV as `oom`. They are capacity
evidence, not timing points.

## Composable MLP module tree

The `mlp_chain` lane measures a named `2 -> 4 -> 1` sequential MLP chain. An
independent NumPy fixture checks the value, packed parameter/input JVP and VJP,
and a central-difference differentiated-VJP HVP before retaining any Fortran
timing. The release app reports separate predict, JVP, VJP, and HVP costs.

Run it with:

    python -B scripts/bench_mlp_chain.py \
      --fortml ../fortml --output results/mlp_chain.csv

The CUDA capability row is explicitly `unavailable`: no resident fused chain
kernel exists yet, so a CUDA request returns `FORTNUM_NOT_IMPLEMENTED` and is
never timed through a host fallback. Details are in
[`results/MLP_CHAIN.md`](results/MLP_CHAIN.md).

## Unfactored Adafactor training

The Adafactor lane checks the deterministic unfactored vector recurrence used
by `fortml_trainer` and `mlp_train`: exponentially averaged squared gradients,
update-RMS clipping, and exact split/resume state. An independent NumPy oracle
runs before the public `test_trainer` and `test_mlp_adafactor` gates. The flat
API has no parameter-layout metadata, so this lane does not claim matrix
row/column factorization; CUDA remains a typed unavailable result until a
resident no-autodiff kernel and a layout-aware adapter are linked.

```bash
python -B scripts/bench_adafactor.py \
  --fortml ../fortml --output results/adafactor.csv
```

See [`results/ADAFACTOR.md`](results/ADAFACTOR.md).

The layout-aware Adafactor lane adds an independent matrix row/column oracle,
an unfactored vector-block fallback, MLP integration, and the explicit CUDA
boundary:

```bash
python -B scripts/bench_adafactor_factored.py \
  --fortml ../fortml --output results/adafactor_factored.csv
```

See [`results/ADAFACTOR_FACTORED.md`](results/ADAFACTOR_FACTORED.md).

## AMSGrad optimizer and MLP training

The AMSGrad lane checks the deterministic bias-corrected first moment and
elementwise maximum second-moment recurrence used by `mlp_train`. An
independent NumPy oracle covers a 4,096-parameter, 128-step state trajectory
and a one-feature linear MLP for 32 epochs before the FortML release timing is
retained. The CUDA rows are explicit `unavailable`: no resident AMSGrad state
kernel is linked, and no host fallback is relabeled as device evidence.

```bash
python -B scripts/bench_amsgrad_training.py \
  --fortml ../fortml --output results/amsgrad.csv
```

See [`results/AMSGRAD.md`](results/AMSGRAD.md). The FortML behavioral fixture
also checks in-memory and formatted checkpoint continuation, including the
maximum-second-moment state.

## RAdam optimizer and MLP training

The RAdam lane checks the independent bias-corrected first/second-moment
recurrence, including its `rho_t` rectification threshold, for a 4,096-
parameter state trajectory and a one-feature linear MLP. The FortML app is
`fortml_bench_radam_training`; the NumPy oracle runs before any CPU timing is
retained. CUDA rows are explicitly `unavailable` because no resident RAdam
state kernel is linked and host fallback is forbidden.

```bash
python -B scripts/bench_radam_training.py \
  --fortml ../fortml --output results/radam.csv
```

See [`results/RADAM.md`](results/RADAM.md). The source test also checks exact
format-8/text-schema-6 checkpoint resume and the output-preserving CUDA refusal.
The fixed full-batch trajectory products have a separate correctness gate in
[`results/MLP_RADAM_HYPERGRADIENT.md`](results/MLP_RADAM_HYPERGRADIENT.md),
including central differences, scalar adjointness, FortOpt L-BFGS-B, typed
rho-branch refusals, and the explicit CUDA boundary.

## Validity boundary

The first comparison is a kernel-product comparison. It does not claim that a
full GP training run has the same cost. The separate CG workload adds a
matched unpreconditioned matrix-free solve. Preconditioned CG, GP marginal
likelihood, and prediction remain separate workloads.

Every result records the machine, compiler, flags, package versions, source
revisions, precision, dimensions, residency mode, repetitions, setup time,
runtime, peak memory where available, and correctness error. A missing or
unsupported competitor is reported explicitly.

The RBF constants are variance 1.4, lengthscale 0.7, and diagonal shift 0.08.
All implementations use float64 and the same deterministic points and input
vector. The Fortran, KeOps, and GPyTorch adapters are compared by the same
blocked pairwise operator, with only storage, tiling, and execution backend
changing. CG rows use resident inputs. The FortML solver's internal Krylov
workspace is allocated and mapped by its call boundary.

The first regular-grid structured workload is recorded in
[`results/tensor_product_device.csv`](results/tensor_product_device.csv) with
its interpretation in
[`results/tensor_product_device.md`](results/tensor_product_device.md). It
measures the resident OpenACC tensor contraction from fortnum. It is not a
KeOps pairwise-kernel comparison. The compact-support Wendland C2 workload is
recorded in [`results/sparse_compact_support.csv`](results/sparse_compact_support.csv)
with its scaling interpretation in
[`results/sparse_compact_support.md`](results/sparse_compact_support.md). It
compares the FortML resident CSR product with dense PyTorch and KeOps at
matched float64 parameters. Dense PyTorch OOM rows are retained explicitly.

## License

MIT. See [LICENSE](LICENSE).
