# Benchmark results

The maintained family-level status index is
[PARITY_MATRIX.md](PARITY_MATRIX.md). It separates CPU correctness, resident
CUDA correctness, transfer-inclusive measurements, and typed refusals.

Complete-call studies:

- [Exact GP and MLP workloads](MODEL_WORKLOADS.md)
- [MLP training, basis pipeline, and tree workloads](FEATURES.md)
- [SGD/Nesterov training and differentiable imputation](TRAINING_IMPUTER.md)
- [AdamW training and MLP hypergradients](ADAMW_HYPERGRADIENT.md)
- [AdamW beta-logit hypergradients](ADAMW_BETA_HYPERGRADIENT.md)
- [Typed MLP learning-rate schedules](MLP_SCHEDULES.md)
- [Dense MLP activation products](MLP_ACTIVATIONS.md)
- [Scheduled MLP trajectory hypergradients](MLP_SCHEDULE_HYPERGRADIENT.md)
- [Composable MLP module tree](MLP_CHAIN.md)
- [Differentiable neural losses and weighted MLP objective](NEURAL_LOSSES.md)
- [Weighted ridge regression and derivative products](RIDGE.md)
- [Weighted elastic-net regression and derivative products](ELASTIC_NET.md)
- [RMSprop optimizer and MLP training](RMSPROP.md)
- [RMSprop trajectory hypergradients](RMSPROP_HYPERGRADIENT.md)
- [Resident CUDA correctness contracts](DEVICE_CONTRACTS.md)
- [Grouped MLP value, HVP, and bounded L-BFGS-B](MLP_GROUPED_TRAINING.md)
- [Physics-informed and Hamiltonian model evidence](PHYSICS_MODELS.md)
- [Resident CUDA AdamW state gate](CUDA_ADAMW.md)
- [Deterministic k-nearest-neighbor classification](KNN.md)
- [One-vs-one logistic classification](OVO_LOGISTIC.md)
- [Multilabel logistic classification](MULTILABEL_LOGISTIC.md)
- [Radius-neighbor classification](RADIUS_NEIGHBORS.md)
- [Weighted linear SVM classification](LINEAR_SVM.md)
- [Weighted linear SVR regression](LINEAR_SVR.md)
- [Weighted ordinal logistic classification](ORDINAL_LOGISTIC.md)
- [GP classification hyperparameter training](GP_CLASSIFICATION_TRAINING.md)
- [GP classification and fitted preprocessing](CLASSIFICATION_EXTENSIONS.md)
- [Binary GP likelihood value/JVP/VJP primitive](GP_LIKELIHOOD.md)
- [Probability calibration](PROBABILITY_CALIBRATION.md)
- [Monotonic XGBoost constraints](XGBOOST_MONOTONIC_CONSTRAINTS.md)
- [Multinomial and neural classification](CLASSIFICATION_MODELS.md)
- [Differentiable Multinomial Naive Bayes](MULTINOMIAL_NB.md)
- [Differentiable Complement Naive Bayes](COMPLEMENT_NB.md)
- [Categorical Naive Bayes](CATEGORICAL_NB.md)
- [Integer one-hot encoder](ONE_HOT_ENCODER.md)
- [Stochastic, derivative, multi-output, and variational GP features](GP_FEATURES.md)
- [Derivative-observation GP query products](DERIVATIVE_GP.md)
- [Multilabel classification metrics](MULTILABEL_METRICS.md)
- [ROC-AUC and PR-AUC ranking metrics](ROC_AUC.md)
- [Smooth kernel catalog](KERNEL_CATALOG.md)
- [Weighted LDA and QDA](DISCRIMINANT_ANALYSIS.md)
- [Robust Huber and quantile XGBoost](XGBOOST_ROBUST.md)
- [Hyperparameter grid, seeded random, and L-BFGS-B search](HYPERPARAMETER_SEARCH.md)
- [Large-GP review comparison](SCALABLE_GP.md)

The current large-GP addenda are
[`scalable_gp_grbcm_corrected.csv`](scalable_gp_grbcm_corrected.csv),
[`scalable_gp_clustered.csv`](scalable_gp_clustered.csv), and
[`scalable_gp_dimension_current.csv`](scalable_gp_dimension_current.csv), each
with a matching training-time plot. Historical GRBCM rows are retained only as
an audit trail and are superseded by the corrected record.

## Recorded RBF MVM result

Run date: 2026-08-06. Workload: 2048 samples, 8 features, float64, 12 MVM
repetitions. The CPU comparison uses 16 physical cores on an AMD Ryzen 9 5950X.
The GPU comparison uses an NVIDIA GeForce RTX 5060 Ti. Every recorded row
passed the independent blocked NumPy oracle. Capacity failures are recorded
as OOM rather than treated as correctness passes.

Fortran is within the 30-percent target of GPyTorch-KeOps on this workload.
The current scaling sweep uses the contiguous sample-major kernel from fortml
commit `81b0655`. Its CPU lane uses nvfortran -O3 -mp and its GPU lane uses
nvfortran -O3 -acc. The CPU and GPU compiler, package, driver, source-commit, and
numerical error fields are in rbf_mvm_scaling.csv.

Plot:

https://box.sloppy.at/8ba9a.png

The Slopbox URL is public and expires after three days. This result covers the
RBF matrix-vector product only. Matched CG is documented below. Stochastic
log-determinant evidence is in [GP_FEATURES.md](GP_FEATURES.md), and exact GP
fit/prediction evidence is in [MODEL_WORKLOADS.md](MODEL_WORKLOADS.md).

## Scaling sweep

The scaling record covers 256, 512, 1024, 2048, and 4096 samples with the same
float64 constants and deterministic inputs. It uses twelve timed repetitions per
size. Dense PyTorch reaches `oom` at 4096 on the 16 GiB GPU. KeOps,
GPyTorch-KeOps, and Fortran continue and pass the independent oracle.

CPU plot:

https://box.sloppy.at/0f460.png

GPU plot:

https://box.sloppy.at/e1f7f.png

The merged scaling data is in `rbf_mvm_scaling.csv`. At 4096 samples, Fortran
takes 4.32 ms per resident GPU MVM, compared with 6.31 ms for GPyTorch-KeOps
and 7.48 ms for KeOps. On the CPU lane, nvfortran Fortran takes 4.66 ms,
compared with 6.22 ms for GPyTorch-KeOps and 7.56 ms for KeOps. The Fortran
curve is lowest at every tested size on both devices.

## Static composite kernel lane

The static RBF-plus-constant lane uses the generic CUDA plan introduced in
fortml commit `5ff3d80`. Its postfix program is lowered once during residency
setup, and the timed CUDA MVM reuses the device-resident points and program.
The same float64 inputs, kernel constants, diagonal shift, and independent
blocked pairwise oracle are used for every backend. The base sweep has five
repetitions at 256 through 4,096 samples. The slope extension has three
repetitions at 8,192 and 16,384 samples. The merged record is
`composite_mvm_scaling_extended.csv`.

At 16,384 samples, resident CUDA FortML takes 56.68 ms per MVM, compared with
64.13 ms for GPyTorch-KeOps and 83.61 ms for KeOps. Dense PyTorch is recorded
as OOM from 8,192 samples. The last two FortML CUDA doublings have slopes 2.00
and 2.00, which is the expected quadratic matrix-free scaling once launch and
tile overheads are amortized. CPU placement noise remains visible in the
extended endpoint, so CPU slope claims use the recorded 16-thread nvfortran
run without extrapolation.

Composite CPU plot: `composite_mvm_scaling_extended_cpu.png`

https://box.sloppy.at/8b824.png

Composite GPU plot: `composite_mvm_scaling_extended_cuda.png`

https://box.sloppy.at/92846.png

The dense PyTorch CPU curve is a dense-reference measurement. Its implementation
materializes the full (N 	imes N 	imes D) difference tensor and two
(N 	imes N) intermediates on every call. At 4,096 samples it takes 349 ms,
whereas a diagnostic blocked PyTorch version still takes 327 ms. This confirms
that setup time is not causing the offset. FortML and KeOps avoid those
intermediates with fused matrix-free reductions, so dense PyTorch is not an
apples-to-apples implementation of the same memory strategy.

The operation-level comparison is in
[OPERATION_PROFILE.md](OPERATION_PROFILE.md), with raw torch.profiler
tables in operation_profile_cpu.csv and operation_profile_cuda.csv.

## High-N slope check

A follow-up sweep extended the same float64 workload to 8,192 and 16,384
samples, with three timed repetitions per size. The merged record is in
`rbf_mvm_scaling_extended.csv`. All non-OOM rows passed the independent
blocked NumPy oracle.

The resident GPU FortML timings were 4.054, 16.129, and 64.390 ms at
4,096, 8,192, and 16,384 samples. The local doubling slopes are 1.992 and
1.997, so the GPU curve is stably quadratic over the extended range. KeOps
and GPyTorch-KeOps have slopes of about 1.5 on the last doubling, while
dense PyTorch is OOM on the GPU from 4,096 onward.

The first CPU sweep shows process-placement noise at 8,192 samples. Three
additional unpinned nvfortran runs per size, recorded in
`rbf_mvm_scaling_cpu_repeats.csv`, gave median FortML timings of
5.707, 16.202, and 59.851 ms. Their local slopes are 1.505 and 1.885 and
are approaching the expected quadratic regime. The CPU result should
therefore be read with the runtime-placement caveat until the cluster's CPU
affinity policy is fixed.

Extended CPU plot:

https://box.sloppy.at/c7d09.png

Extended GPU plot:

https://box.sloppy.at/465d6.png

## Matrix-free CG

The matched CG record is `rbf_cg.csv`, generated by `scripts/run_cg_suite.sh`.
It uses 2,048 samples, 8 features, float64, variance 1.4, lengthscale 0.7,
diagonal shift 0.08, tolerance `1e-8`, and a 500-iteration cap. The same
unpreconditioned CG recurrence and true-residual check are used around dense
PyTorch, KeOps, GPyTorch-KeOps, and FortML's specialized RBF operator. The
Python lanes and Fortran lane keep the input points resident. FortML maps its
solver workspace at the call boundary.

All rows passed the independent blocked NumPy residual check. The 2,048-sample
lane additionally compares against `numpy.linalg.solve`. The nvfortran/
OpenACC FortML solve took 0.187 s per solve on the RTX 5060 Ti, compared with
0.754 s for the explicit KeOps loop and 0.597 s for the matching
GPyTorch-KeOps loop. The 16-thread CPU FortML result was 0.162 s, compared with
0.735 s for KeOps and 0.568 s for GPyTorch-KeOps. This is matrix-free solver
evidence. It does not cover preconditioned solves, stochastic log determinants,
or full GP training.

CPU plot: `rbf_cg_scaling_cpu.png`

GPU plot: `rbf_cg_scaling_cuda.png`

The primary four-point sweep reaches 2,048 samples with all rows passing. The
extended record `rbf_cg_scaling_extended.csv` adds 4,096 samples. FortML takes
828.7 ms on CPU and 872.4 ms on CUDA at 4,096 samples. The corresponding KeOps
and GPyTorch-KeOps times are 1,841.5/1,876.3 ms and 1,431.8/1,445.8 ms for
CPU/CUDA. Dense PyTorch is OOM at 4,096 on CUDA. The FortML CUDA doubling
slope from 2,048 to 4,096 is 2.22, compared with 1.31 for KeOps and 1.28 for
GPyTorch-KeOps. The two-row output tile lowers the FortML endpoint. The
optional native CUDA shared-neighbor tile is now profiled separately, with
OpenACC retained as the default because both paths are within the same timing
envelope on this GPU.

Extended CPU plot: `rbf_cg_scaling_extended_cpu.png`
https://box.sloppy.at/9cef6.png

Extended GPU plot: `rbf_cg_scaling_extended_cuda.png`
https://box.sloppy.at/4d9a5.png

## Batched matrix-matrix products

The fixed eight-feature native CUDA path also fuses up to eight right-hand
sides. At 2,048 samples and four right-hand sides, the resident native kernel
takes 1.363 ms, compared with 3.666 ms for the OpenACC matmat loop. At eight
right-hand sides, the native path takes 1.405 ms, compared with 7.327 ms for
OpenACC. Every row in `rbf_matmat.csv` passes the direct pairwise oracle for
each right-hand side. The CPU, OpenACC, and native CUDA runs are produced by
`scripts/run_matmat_suite.sh`.

CUDA plot: `rbf_matmat_rhs_cuda.png`
https://box.sloppy.at/aabb5.png

CPU plot: `rbf_matmat_rhs_cpu.png`
https://box.sloppy.at/98dcc.png
