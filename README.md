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

The kernel-catalog lane checks the periodic and rational-quadratic covariance
leaves, their input derivatives, and logarithmic parameter JVP/VJP/HVP products
against independent NumPy formulas. It records host timings and explicit CUDA
capability refusals until the resident postfix ABI carries the third leaf
parameter:

```bash
python -B scripts/bench_kernel_catalog.py \
  --fortml ../fortml --output results/kernel_catalog.csv
```

See [`results/KERNEL_CATALOG.md`](results/KERNEL_CATALOG.md).

The derivative-GP lane checks exact query-input JVP/VJP products for mixed
value/first-derivative periodic and rational-quadratic GPs against an
independent NumPy covariance and posterior oracle. CPU timings are retained
only after the checks pass; CUDA is an explicit typed refusal until the
resident derivative-GP graph is linked:

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

The linear-margin lane checks weighted primal squared-hinge SVM fitting,
arbitrary integer classes, labels, and signed decision margins against an
independent NumPy/SciPy L-BFGS-B oracle:

```bash
python -B scripts/bench_linear_svm.py \
    --fortml ../fortml --output results/linear_svm.csv
```

See [`results/LINEAR_SVM.md`](results/LINEAR_SVM.md). CUDA is an explicit
unavailable capability row until a resident linear-SVM kernel is linked.

The differentiable neural-loss lane checks BCE/logistic, softmax
cross-entropy, weighted MSE, and Huber HVPs against independent NumPy
curvature formulas, and checks the weighted-MSE path used by the MLP objective:

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
linear, `tanh`, ReLU, GELU, SiLU, ELU, softplus, and leaky ReLU against an
independent NumPy checksum oracle:

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

Binary probability calibration (Platt sigmoid and weighted PAVA isotonic) is
checked against an independent NumPy oracle:

```bash
.venv/bin/python -B scripts/bench_probability_calibration.py \
    --fortml ../fortml --output results/probability_calibration.csv
```

See [`results/PROBABILITY_CALIBRATION.md`](results/PROBABILITY_CALIBRATION.md).
CUDA capability rows are explicit refusals because no resident calibration
kernel is linked.

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

The resident CUDA contract lane independently checks the native kNN prediction
plan and the no-autodiff RMSprop optimizer-state kernel. NumPy computes the
expected labels, checksums, and five-step centered recurrence before the native
gates run. These are correctness rows, not timings for a complete estimator or
trainer; missing CUDA toolchains/devices remain explicit `skipped` records:

```bash
.venv/bin/python -B scripts/bench_device_contracts.py \
  --fortml ../fortml --output results/device_contracts.csv
```

See [`results/DEVICE_CONTRACTS.md`](results/DEVICE_CONTRACTS.md) for the exact
fixtures, GPU/toolchain metadata, and remaining resident-workload boundaries.

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

The generic hyperparameter-search lane uses an independent three-parameter
quadratic oracle to gate Cartesian grid and FortOpt L-BFGS-B timings:

```bash
.venv/bin/python -B scripts/bench_hyperparameter_search.py \
  --fortml ../fortml --output results/hyperparameter_search.csv
```

See [`results/HYPERPARAMETER_SEARCH.md`](results/HYPERPARAMETER_SEARCH.md).
The CUDA row is an explicit unavailable refusal until resident objective/search
state is implemented.

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
