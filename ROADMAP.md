# fortml-bench roadmap

This repository owns reproducible cross-engine evidence for fortml. A result
requires an independent oracle, matched mathematical work, recorded toolchain
metadata, and a committed raw record.

## Status and handoff

Last updated 2026-08-07. The scalable-GP study of Liu et al. (IEEE TNNLS <!-- slop-ok -->
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
- [x] Add regular-grid Toeplitz/Kronecker evidence with independent dense or
  structured oracles and resident OpenACC scaling records.
- [x] Add compact-support sparse workloads using `fortsparse`, with stored
  nonzero/storage diagnostics, independent row-wise oracles, and matched
  float64 CPU/GPU measurements against dense PyTorch and KeOps.
- [x] Add a KeOps-style static RBF-plus-constant matrix-free lane with a
  blocked independent oracle, native CUDA option, and high-N CPU/GPU scaling
  evidence against KeOps, GPyTorch-KeOps, and dense PyTorch.
- [x] Add derivative-observation and derivative-prediction workloads.
- [x] Add multi-output and variational GP workloads.

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
- [x] Add a deterministic MLP-training lane with an independent NumPy Adam
  oracle, full loss/prediction checks, and release-build fit timings.
- [x] Add a composable polynomial/Fourier basis-pipeline lane with independent
  transform, JVP, and VJP checks and timings.
- [x] Add deterministic CART-stump and residual-stump boosting lanes with an
  exhaustive NumPy split oracle and matched scikit-learn depth-1 reference.
- [x] Add fitted standard/min-max scaler and binary Laplace GP logistic/probit
  lanes with independent NumPy transform and Newton posterior oracles. The
  raw record is `results/classification_extensions.csv`.
- [x] Add an exact depth-one second-order boosting lane with independent NumPy
  gradient/Hessian, regularized leaf-weight, split-gain, squared-objective, and
  logistic-objective checks. Record a dedicated FortML workload and an explicit
  optional-XGBoost contextual/refusal row in `results/xgboost_workloads.csv`.
- [ ] Add matched multinomial softmax regression and neural classifier lanes.
  FortML now has the softmax and multiclass MLP implementations (including
  deterministic input-shape oracles), but this repository still needs the
  matched probability/metric/timing record before the benchmark lane is
  complete.
- [ ] Add multiclass and variational GP classification with GPyTorch likelihood
  references and independent dense small-data oracles. The binary Laplace lane
  is complete and remains explicitly separate from that target.
- [ ] Add the full histogram/CART feature matrix: missing values, sample and
  class weights, monotonic constraints, early stopping, feature importance,
  categorical inputs, and GPU histograms.
- [ ] Add a matched full XGBoost lane when the optional dependency and a pinned
  release are available. The current exact depth-one FortML lane is recorded in
  `results/xgboost_workloads.csv`. Histogram, deeper-tree, missing-value, and
  constraint comparisons remain separate work. A stump benchmark is not called
  XGBoost.
- [ ] Add physics-informed, Hamiltonian, Lagrangian, and symplectic workloads
  with analytic harmonic-oscillator and manufactured-PDE oracles. Record
  trajectory error, energy drift, symplectic Jacobian defect, residual norms,
  posterior calibration, optimizer evaluations, and long-horizon behavior.
- [ ] Add physics-consistent GP, Ghosttasking, Monge-GP, and GP-initialized
  finite-network lanes when their public equations and reference data are
  pinned. Private project results must be accompanied by a reproducible data
  generator and versioned artifact.

Every new lane keeps the independent behavioral oracle ahead of timing. A
missing compiler, GPU, package, equation, or reference dataset produces a
parseable refusal row rather than an omitted result.
