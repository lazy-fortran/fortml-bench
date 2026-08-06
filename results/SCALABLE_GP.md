# Scalable GPs: measured cost against the review's claimed complexity

Reference: H. Liu, Y.-S. Ong, X. Shen and J. Cai, "When Gaussian Process Meets
Big Data: A Review of Scalable GPs", *IEEE Transactions on Neural Networks and
Learning Systems* 31(11):4405-4423, November 2020,
[doi:10.1109/TNNLS.2019.2957109](https://doi.org/10.1109/TNNLS.2019.2957109).

## What "matching the paper" can and cannot mean

The review contains **no numeric result tables**. Its reproducible content is:

* the one-dimensional toy of Figs. 4 and 5 (`y(x) = sinc(x) + eps` with
  `eps ~ N(0, 0.04)` and 120 training points) and the qualitative behaviours
  reported in those captions and in Section IV-C;
* the scalability/capability sketch of Fig. 2 and the complexity statements
  scattered through Sections III to V;
* Table I, a list of libraries implementing these methods;
* Table II, a list of large regression data sets used in the literature.

So the comparison here has two halves. The **behaviours** are checked as
assertions inside `fortml`'s own test suite, on the paper's fixture, in
`test/test_review_toy.f90`. The **complexity orders** are measured here and put
beside the claimed ones. Numbers that the paper never published are not
invented.

## GRBCM record status

Every historical `grbcm` row in the `scalable_gp_*.csv` files predates the
production correction to generalized RBCM. Those runs used a small shared
expert and did not train the published disjoint communication set and enhanced
experts. The rows and all GRBCM slopes or accuracy claims derived from them are
superseded. They remain in the raw files only to preserve the audit trail and
must not be used as evidence for the corrected method. Corrected contiguous and
clustered results are recorded separately in
[`scalable_gp_grbcm_corrected.csv`](scalable_gp_grbcm_corrected.csv).

## Current corrected and multidimensional records

These focused records use float64, gfortran with `-O3 -funroll-loops`, one
pinned CPU core, and three repetitions. The harness runs the method-specific
behavioral tests before rebuilding the timed release executable. Each CSV also
stores the full affinity set, compiler version, claimed order, oracle test
names, partition policy, revisions, the FortML tracked-diff hash, and hashes of
the benchmark driver and relevant sources.

### Corrected GRBCM

The corrected communication-set and enhanced-expert implementation passes the
independent dense-formula checks in `test_local_experts`. Its current sample
sweep is:

| n | contiguous train [ms] | clustered train [ms] | contiguous SMSE | clustered SMSE |
|---:|---:|---:|---:|---:|
| 256 | 0.582 | 0.629 | 0.0483 | 0.0494 |
| 512 | 2.74 | 3.12 | 0.0285 | 0.0284 |
| 1,024 | 14.1 | 15.1 | 0.0182 | 0.0185 |

The measured training slopes are +2.30 for contiguous partitioning and +2.29
for deterministic Lloyd partitioning. They are below the cubic fixed-expert
asymptote over these small sizes. The source record and plot are
[`scalable_gp_grbcm_corrected.csv`](scalable_gp_grbcm_corrected.csv) and
[`scalable_gp_grbcm_corrected_train_seconds.png`](scalable_gp_grbcm_corrected_train_seconds.png).

### Contiguous and clustered experts

At `n = 1024`, the expert-count sweep includes the partition step in the timed
training call. Slopes are fitted over `M` in {8, 16, 32}:

| method | contiguous slope | clustered slope |
|---|---:|---:|
| NLE | -1.06 | -0.63 |
| PoE | -1.10 | -0.61 |
| GPoE | -1.07 | -0.68 |
| BCM | -1.06 | -0.63 |
| RBCM | -1.06 | -0.60 |
| corrected GRBCM | -1.23 | -1.01 |
| MoE | -1.08 | -0.65 |

Lloyd partitioning offsets part of the factorization saving as `M` grows, so
the clustered slopes are less negative. The record does not support a blanket
accuracy preference. SMSE depends on the method and expert count. See
[`scalable_gp_clustered.csv`](scalable_gp_clustered.csv) and
[`scalable_gp_clustered_train_seconds.png`](scalable_gp_clustered_train_seconds.png).

### Multidimensional SKI

Here `m = 64` is a maximum total grid budget, not a per-axis size. The target
depends only on the first coordinate. The added coordinates are nuisance
inputs that still affect isotropic RBF distance and interpolation.

| d | equal-axis grid | status | train [ms] | SMSE |
|---:|---:|:---|---:|---:|
| 1 | 64 | pass | 0.483 | 0.0161 |
| 2 | 8 x 8 | pass | 0.482 | 0.733 |
| 4 | 2 x 2 x 2 x 2 | pass | 0.702 | 1.01 |
| 8 | minimum is 2^8 = 256 | refused by grid budget | - | - |

The fixed budget exposes the expected grid limitation: accuracy deteriorates
as the grid thins, and `d = 8` is an explicit refusal rather than an omitted or
failed timing. `test_ski_gp` supplies the independent dense-tensor oracle. The
raw record and pass-row plot are
[`scalable_gp_dimension_current.csv`](scalable_gp_dimension_current.csv) and
[`scalable_gp_dimension_current_train_seconds.png`](scalable_gp_dimension_current_train_seconds.png).

## Behaviours reproduced (fortml `test_review_toy`)

| Paper's statement | Where | Check |
| --- | --- | --- |
| SoR is overconfident leaving the training data | Fig. 4 caption | SoR variance is below DTC's at every test point, and collapses below `1e-8` far from the inducing set |
| FITC captures heteroscedasticity in variance | Fig. 4 caption | FITC predictive variance varies by more than 5 % across the range |
| VFE approximates the full GP well | Fig. 4 caption | collapsed VFE predictive mean is within 0.15 of the exact GP everywhere |
| DTC variance grows to the prior away from the inducing points | Fig. 4 caption | DTC variance equals the prior to `1e-8` far away |
| PoE gives overconfident variance | Sec. IV-C | PoE aggregate is sharper than *every* individual expert at all 40 test points |
| GPoE and MoE suppress that | Sec. IV-C | neither is sharper than PoE anywhere, and GPoE returns exactly to the prior far away |
| MoE is never sharper than its sharpest expert | Sec. IV-C | mixture variance is at least the minimum expert variance |

Additional exactness identities, which the paper states structurally rather
than plots: DTC, FITC and PITC each collapse to the full GP when the inducing
set is the training set (checked to `1e-7` on mean, variance and log marginal
likelihood), and PoE, normalized GPoE and BCM each reproduce the exact GP with
a single expert.

## Measured scaling

Workload: the paper's fixture extended to `n` points, RBF kernel with unit
signal variance and unit lengthscale, noise variance 0.04, 256 held-out test
points, float64, gfortran, 16 physical cores, best of 2 repetitions. Peak
resident memory is the kernel's own `VmHWM` for the benchmark process.

Raw data: `scalable_gp_samples.csv`, `scalable_gp_inducing.csv`,
`scalable_gp_experts.csv`, `scalable_gp_dimension.csv`. Plots:
`scalable_gp_<sweep>_<metric>.png`.

### Training time against the number of training points

Fixed `m = 64`, `M = 8`, `d = 1`, over `n` in {256, 512, 1024}. The slope is a
log-log least-squares fit.

| Method | Claimed order | Measured slope in `n` | Reading |
| --- | --- | --- | --- |
| full | `O(n^3)` | +2.55 | approaching cubic |
| keops (matrix-free exact) | `O(s n^2)` | +2.18 | quadratic, as claimed |
| sod | `O(m^3)`, independent of `n` | −0.04 | flat, as claimed |
| sor | `O(n m^2)` | +0.94 | linear, as claimed |
| dtc | `O(n m^2)` | +0.93 | linear |
| fitc | `O(n m^2)` | +0.92 | linear |
| vfe | `O(n m^2)` | +0.10 | dominated by the `O(m^3)` closed-form solve at these `n` |
| pitc | `O(n m^2 + M b^3)` | +2.74 | the `b = n/M` blocks grow with `n` here, so the block term dominates |
| ski | `O(n + m log m)` | +0.81 | linear, as claimed |
| nle, poe, gpoe, bcm, rbcm, moe | `O(M m0^3)` with `m0 = n/M` | +2.00 to +2.05 | rising toward the cubic-in-`m0` regime |

The local experts are claimed to be cubic in the per-expert size `m0`, which at
fixed `M` means cubic in `n`. The measured +2.0 says these sizes are not yet
asymptotic, not that the claim is wrong. The same caveat applies to `full` at
+2.55.

### Training time against the inducing size

Fixed `n = 1024`, `d = 1`, over `m` in {16, 32, 64, 128, 256}.

| Method | Claimed order in `m` | Measured slope |
| --- | --- | --- |
| sod | `O(m^3)` | +2.25 |
| sor | `O(n m^2)` | +1.59 |
| dtc | `O(n m^2)` | +1.68 |
| fitc | `O(n m^2)` | +1.74 |
| pitc | `O(n m^2)` | +1.00 |
| vfe | `O(n m^2)` | +4.31 |
| ski | `O(m log m)` | +0.42 |

The VFE row is an honest artifact of *this* implementation, not of the method:
the benchmark forms the optimal `q(u)` in closed form with one dense solve per
inducing column, which is `O(m^4)` rather than the `O(n m^2 + m^3)` the method
allows. The paper's VFE cost is reached by the stochastic variational path
(its Section III-C2, eq. 18), which this repository does not yet implement.
This is recorded rather than hidden.

### Training time against the expert count

Fixed `n = 1024`, `m = 64`, `d = 1`, over `M` in {2, 4, 8, 16, 32}.

The six non-GRBCM local methods measure a slope of −1.04 to −1.08 in `M`. The
claim is
`O(M m0^3) = O(n^3 / M^2)`, i.e. slope −2. Again, the measured value says the
per-expert problems are too small to be asymptotic at `n = 1024`, and the
direction and the fact that more experts is cheaper are as claimed.

### Training time against the input dimension

Fixed `n = 1024`, `m = 64`, `M = 8`, over `d` in {1, 2, 4, 8}. The truth
depends only on the first coordinate. The other coordinates are nuisance
inputs.

Every method measures a slope between −0.40 and +0.17, i.e. **no method's cost
is materially driven by `d` here**. That is expected: the kernel evaluation is
`O(d)` and every method's dominant term is a factorization or a solve whose
size does not involve `d`. The review's own scalability discussion (Sec. VI-A)
concerns high-dimensional *capability*, not cost, and notes that the grid-based
methods degrade with `d` because `m` grows exponentially. That limit is reached by
construction here. This historical sweep predates multidimensional SKI and
contains no comparable SKI measurements beyond `d = 1`. The current harness
uses `m` as a maximum total grid-point budget and records an explicit refusal
when it cannot provide at least two points per dimension.

## Peak memory

Fixed `m = 64`, `M = 8`, `d = 1`, over `n` in {256, 512, 1024}.

| Method | Measured slope in `n` | Reading |
| --- | --- | --- |
| full | +0.75 | the `O(n^2)` kernel matrix, still partly under the process baseline at these sizes |
| keops, ski | +0.02 | flat: the covariance is never formed |
| sod | +0.02 | flat: the subset size is fixed |
| all sparse and local methods | +0.12 to +0.18 | near flat |

This is the clearest separation in the whole study. At `n = 1024` the exact
dense GP has already reached 12.4 MiB of peak resident memory against 4.9 MiB
for the matrix-free lane, and it is the only curve with a rising slope.

### Accuracy

Standardized mean squared error against the true `sinc`, over `n`:

* `full`, `keops`, `sor`, `dtc`, `fitc`, `pitc`, `vfe` all improve at slope
  −0.66 and agree to three digits at every `n`. The prior and posterior sparse
  approximations lose nothing measurable on this problem at `m = 64`.
* `sod` **degrades** at slope +0.60 and `ski` at +0.88, because their budget
  (subset size, grid size) is held fixed while `n` grows. That is the expected
  failure mode of both, and the review says as much for SoD in Section III-A.
* The non-GRBCM local aggregations sit between: `nle` improves fastest,
  `poe`/`bcm` track the exact GP, and `moe` is the weakest of them. The old
  GRBCM accuracy slope is superseded for the reason stated above.

## Large-n measurement (8,192 to 131,072)

The first sweep above reached only n = 1024 and used the default `fo` build
profile, which is `-O0 -fcheck=all -fbacktrace`. Both were wrong for a
performance claim: the sizes were too small for any asymptotic regime, and the
timings were debug-build timings. This section supersedes it. Everything here
is `-O3 -funroll-loops`, one core, best of one repetition, per-expert size held
at 1024 so the local methods keep a fixed sub-problem, `m = 64` inducing
points, `d = 1`. Raw data: `scalable_gp_large_merged.csv` plus the continuation
rows in `scalable_gp_large_tail.csv`.

The historical CSV uses a legacy schema with no status or refusal-reason
column. Its bare `NaN` rows are retained as an audit trail, not as successful
measurements. The file has no explicit full-GP rows for 32,768 or 131,072. The full
GP entries marked below as preflight are allocation estimates from the launch
guard, not timed runs. The current focused harness uses explicit status and
reason fields for new refusals.

| n | full GP | SoR / DTC / FITC | VFE | local experts | matrix-free product |
| --- | --- | --- | --- | --- | --- |
| 8,192 | 56.5 s / 1029 MiB | 0.040 s / 18 MiB | 0.018 s / 10 MiB | 0.66 s / 85 MiB | 0.246 s / 5.1 MiB |
| 16,384 | 364 s / 4101 MiB | 0.081 s / 30 MiB | 0.035 s / 14 MiB | 1.30 s / 150 MiB | 0.983 s / 5.4 MiB |
| 32,768 | preflight > 8 GB; not run | 0.151 s / 55 MiB | 0.064 s / 22 MiB | 2.44 s / 278 MiB | 3.81 s / 6.0 MiB |
| 65,536 | preflight > 8 GB; not run | FITC 0.325 s / 104 MiB; SoR/DTC not recorded | 0.158 s / 39 MiB | 4.93 s / 534 MiB | 40.55 s / ~7 MiB |
| 131,072 | preflight ~137 GB; not run | DTC 0.686 s / 202 MiB; FITC 0.708 s / 202 MiB; SoR 0.687 s / 202 MiB | 0.413 s / 71.6 MiB | 11.0 s / 1047 MiB | 161.08 s / ~9.5 MiB |

Measured slopes over that range:

These are log-log fits over every finite recorded point for each method; the
131,072 local and matrix-free continuation is in the tail CSV above.

| method | train | peak memory | SMSE |
| --- | --- | --- | --- |
| full | +2.69 | +2.00 | -1.17 |
| sor | +1.02 | +0.88 | -1.20 |
| dtc | +1.03 | +0.88 | -1.20 |
| fitc | +1.02 | +0.88 | **-1.15** |
| vfe | +1.12 | +0.70 | -1.10 |
| poe, gpoe, bcm, rbcm, nle | +1.00 to +1.01 | +0.91 | -0.52 to -0.39 |
| moe | +1.01 | +0.91 | +0.79 |
| keops matrix-free product | +2.41 | **+0.21** | exact |
| sod | +0.04 | +0.09 | **+0.17** |
| ski, 64-node grid | +1.54 | +0.42 | **+1.18** |

The dense GP measures the textbook `O(n^2)` memory exactly (+2.00) through its
last recorded run. Above 16,384, the launch preflight predicts allocations over
an 8 GB budget and the dense timing is not attempted. The sparse approximations measure
`+1.02` time and `+0.88` memory - linear, as claimed - and have the best
accuracy slope in the study. At n = 131,072 FITC reaches SMSE 6.76e-5 in
0.708 s and 202 MiB. The exact GP would need 137 GB.

The two positive accuracy slopes are the study's clearest negative results.
`sod` and `ski` **degrade** as data grows, because each holds a fixed budget
(subset size, grid size) while `n` rises. For SoD that is the review's own
Section III-A point. For SKI it is an artifact of holding the grid at 64 nodes.
`scalable_gp_ski_scaled.csv` repeats it with the grid at `n/8`.

### The device lane and the corrected SKI lane

Two lanes needed a second run to be worth anything.

**SKI held its grid at 64 nodes** in the sweep above, which measures a fixed
budget rather than the method: its accuracy degrades at `+0.64` while the data
grows. Rerun with the grid at `n/8`
(`scalable_gp_ski_scaled.csv`):

| n | grid | train | peak | SMSE |
| --- | --- | --- | --- | --- |
| 8,192 | 1,024 | 0.044 s | 5.0 MiB | 1.85e-3 |
| 16,384 | 2,048 | 0.099 s | 6.2 MiB | 8.27e-4 |
| 32,768 | 4,096 | 0.231 s | 8.0 MiB | 3.71e-4 |
| 65,536 | 8,192 | 0.559 s | 11.2 MiB | 2.13e-4 |
| 131,072 | 16,384 | 1.375 s | 19.0 MiB | 6.97e-5 |

That is linear time, near-flat memory, and accuracy matching FITC's 6.76e-5 at
**one tenth of its memory** - 19.0 MiB against 201.9 MiB. Held at a fixed grid SKI
looks like the worst method in the study. Scaled with the data it has the best
memory-accuracy trade in it.

**The device matrix-free lane** (`scalable_gp_device.csv`, nvfortran 26.5,
`-O3 -acc -gpu=cc89`, RTX 5060 Ti, resident points, OpenACC Krylov products):

| n | device solve | CPU solve | speed-up | peak | SMSE |
| --- | --- | --- | --- | --- | --- |
| 8,192 | 1.12 s | 53.4 s | 48x | 158 MiB | 1.869056e-3 |
| 16,384 | 3.91 s | - | - | 158 MiB | 8.325399e-4 |
| 32,768 | 17.0 s | - | - | 159 MiB | 3.788e-4 |
| 65,536 | 82.7 s | - | - | 160 MiB | 2.138e-4 |
| 131,072 | 374.0 s | ~1.7 h (est.) | - | 162 MiB | 6.995e-5 |

The memory is flat: the 158 MiB is the CUDA runtime, not the problem. The
accuracy is the exact GP's to seven digits where the exact GP can still be run
(8.325399e-4 against the dense 8.325399e-4 at n = 16,384), which is the point -
this lane is not an approximation.

The first attempt at this lane produced numbers that were 48 times too slow,
because `fo` shares `build/fo/bin` across compilers: a later gfortran build had
silently replaced the nvfortran binary, so `FO_FC=nvfortran fo exec` ran the
CPU one. `fo` now clears the native tree when the compiler changes
(`fo` commit `e3cff00`). The device script rebuilds immediately before
measuring and nothing else builds in between.

### Two defects that only appear above 32k

Both were found by this sweep and are fixed on `fortml` main.

* FITC and PITC could produce a slightly negative Nystrom residual
  `k_ii - q_ii` once the two terms agree to working precision, which makes the
  noise matrix indefinite and fails the whole factorization. The residual is
  now clamped at zero, as GPflow and GPy do. The observation noise keeps the
  matrix strictly positive.
* The sparse posterior jitter was scaled to `K_mm` alone, but the matrix
  actually factorized is `K_mm + K_mn L^-1 K_nm`, whose scale grows with `n`.
  At n = 65,536 the data term reached `1e6` and a `1e-10` jitter became
  meaningless, so FITC failed there while passing at 32,768. The jitter is now
  relative to the matrix being factorized.

A third defect, upstream in `fortnum`, blocked the device lane entirely: the
generated FortAD derivative kernels carried no `!$acc routine seq`, so
`nvfortran` refused every compute region that reaches them. Fixed in the
generator and its 39 generated files.

## The answer at n = 131,072

This is the row that settles it. All five methods on the same problem, same
data, same kernel:

| method | train | peak | SMSE |
| --- | --- | --- | --- |
| VFE | **0.413 s** | 71.6 MiB | 8.08e-5 |
| DTC | 0.686 s | 201.8 MiB | 6.78e-5 |
| FITC | 0.708 s | 201.9 MiB | **6.76e-5** |
| SKI (grid `n/8`) | 1.375 s | **19.0 MiB** | 6.97e-5 |
| exact, matrix-free on GPU | 374.0 s | 162.5 MiB | 6.99e-5 |
| exact, dense | preflight ~137 GB; not run | - | - |

The exact solve is **529 times slower than FITC and 272 times slower than SKI,
and its accuracy is no better**: 6.99e-5 against FITC's 6.76e-5 and SKI's
6.97e-5. At this sample count the error is set by the observation noise and the
model, not by the solver, so the approximations have already reached the
accuracy ceiling that exactness would buy. Paying 529x for it returns nothing.

That is the conclusion for this problem. It is not a general law: it holds
because 64 inducing points already summarize a one-dimensional `sinc` with
131,072 samples on it. A target with structure that a rank-64 summary cannot
capture would separate the methods again, and that is the experiment that would
change the answer.

## Is the KeOps-style matrix-free lane good enough on its own?

On this evidence, **accuracy passes and cost does not. The crossover is what
matters**.

* Accuracy: the matrix-free lane is *exact*. Its SMSE matches the dense exact
  GP to eight digits (`1.61255278E-02` against the same value at `n = 1024`),
  and with the LOVE Lanczos variance its mean negative log predictive density
  (−1.92) matches the dense GP's. No approximation can beat it, because it is
  not an approximation.
* Memory: it is the best in the study, flat in `n` where the dense GP is
  quadratic. This is the reason to use it.
* Training time: it is the *worst* of every method at these sizes, at 0.571 s at
  `n = 1024` against 0.0136 s for the dense Cholesky and 0.002 s for the sparse
  approximations. Its slope is +2.18 against the dense +2.55, so it wins
  eventually, but not before `n` is far larger than anything measured here.
* Prediction: 51.8 s at `n = 1024`, three orders of magnitude worse than every
  other method, because the LOVE variance runs 20 Lanczos steps per test point
  and each step is a full `O(n^2)` product. Predictive variance is where the
  matrix-free lane is the bottleneck.

The result is that the matrix-free lane replaces the *dense exact* GP
and nothing else: it removes the memory wall while keeping exactness. It does
not replace the sparse or local approximations, which are one to three orders
of magnitude faster here at accuracy that is indistinguishable on this problem.
A system that has to answer many predictive-variance queries needs either the
approximations or a better variance path than per-point Lanczos.

The measurement that would change this conclusion is a sweep at `n` in the
`10^5`-`10^6` range, where the dense lane cannot run at all and the sparse
methods' fixed `m` starts to cost accuracy. That sweep is not yet recorded.

## Where these methods are implemented elsewhere

From the review's Table I, plus the repositories behind them:

| Package | Language | Methods | Source |
| --- | --- | --- | --- |
| GPML | MATLAB | FITC, VFE, SPEP, SKI | <http://www.gaussianprocess.org/gpml/> |
| GPy | Python | VFE, SPEP, SKI, SVGP | <https://github.com/SheffieldML/GPy> |
| GPstuff | MATLAB, R | SoR, DTC, FITC, VFE, SVGP, CS, PIC, FIC | <https://github.com/gpstuff-dev/gpstuff> |
| GPflow | Python | FITC, VFE, SVGP, NN+SVGP | <https://github.com/GPflow/GPflow> |
| pyMC3 | Python | DTC, FITC, VFE | <https://github.com/pymc-devs/pymc3> |
| GPyTorch | Python | SKI, DKL (NN+SKI) | <https://github.com/cornellius-gp/gpytorch> |
| pyGPs | Python | FITC | <https://github.com/PMBio/pygp> |
| AugGP | Julia | VFE, SVGP | <https://github.com/theogf/AugmentedGaussianProcesses.jl> |
| laGP | R | NeNe | <http://bobby.gramacy.com/r_packages/laGP/> |
| GPLP | MATLAB | NeNe, PoE, DDM, PIC | <http://www.jmlr.org/mloss/> |
| KeOps / PyKeOps | C++, Python | matrix-free kernel reductions | <https://github.com/getkeops/keops> |

`scripts/fetch_reference_implementations.sh` records and downloads these, with
checksums, into the ignored `.provenance/` tree.

## Final comparison plots (8k to 131k, release build)

These supersede the small-n plots below. `ski_scaled` is SKI with the grid at
`n/8`. `keops_gpu` is the resident OpenACC matrix-free solve.

| Plot | URL |
| --- | --- |
| training time against `n` | https://box.sloppy.at/9fdc1.png |
| peak memory against `n` | https://box.sloppy.at/945a4.png |
| accuracy against `n` | https://box.sloppy.at/15ceb.png |

Measured slopes over 8,192 to 131,072:

The table uses every finite recorded point for each method. Plot labels use the
plotting script's three-point tail fit, so their displayed slope can differ
slightly.

| method | train | peak memory | SMSE |
| --- | --- | --- | --- |
| full | +2.69 | +2.00 | -1.17 |
| sor / dtc / fitc | +1.02 / +1.03 / +1.02 | +0.88 / +0.88 / +0.88 | -1.20 / -1.20 / **-1.15** |
| vfe | +1.12 | +0.70 | -1.10 |
| ski_scaled | +1.24 | +0.47 | -1.14 |
| poe, gpoe, bcm, rbcm, nle | +1.00 to +1.01 | +0.91 | -0.52 to -0.39 |
| moe | +1.01 | +0.91 | +0.79 |
| keops_gpu (device, exact) | +2.12 | **+0.01** | -1.14 |
| keops matrix-free product | +2.41 | +0.21 | exact |
| sod | +0.04 | +0.09 | **+0.17** |

## Published plots (superseded, small-n debug build)

| Plot | URL |
| --- | --- |
| training time against `n` | https://box.sloppy.at/058c3.png |
| peak memory against `n` | https://box.sloppy.at/c70d3.png |
| accuracy against `n` | https://box.sloppy.at/24744.png |
| prediction time against `n` | https://box.sloppy.at/7c8d5.png |
| training time against `m` | https://box.sloppy.at/6ca51.png |
| training time against `M` | https://box.sloppy.at/55811.png |

## Reproduce

```sh
python3 scripts/bench_scalable_gp.py --output results/scalable_gp_samples.csv \
    --sweep samples --values 256 512 1024 --repetitions 2 --threads 16
python3 scripts/plot_scalable_gp.py --input results/scalable_gp_samples.csv \
    --prefix results/scalable_gp_samples --metric train_seconds
```

This command preserves the historical record's 16-thread policy while running
the current implementation. New focused sweeps use `--threads 1`. The CSV
records both the thread count and the full CPU affinity set. `--cpu-affinity`
accepts an explicit list when the automatic first-N-allowed-CPUs choice is
unsuitable.

Sweeps `inducing`, `experts` and `dimension` take `--sweep` with the matching
name. `--methods` restricts the set. For SKI, `m` is the maximum total grid
budget. With the default `m = 64`, dimensions 1, 2, and 4 use grids of 64,
8-by-8, and 2-by-2-by-2-by-2 points. The `d = 8` row is recorded as refused.
Use `--m 256` to provide the minimum two points per axis at `d = 8`.

Reproduce the current focused records and plots with:

```sh
python3 scripts/bench_scalable_gp.py \
    --output results/scalable_gp_grbcm_corrected.csv \
    --sweep samples --values 256 512 1024 \
    --methods grbcm grbcm_clustered --experts 8 \
    --repetitions 3 --threads 1 --compiler gfortran \
    --flags='-O3 -funroll-loops'
python3 scripts/plot_scalable_gp.py \
    --input results/scalable_gp_grbcm_corrected.csv \
    --prefix results/scalable_gp_grbcm_corrected --metric train_seconds

python3 scripts/bench_scalable_gp.py \
    --output results/scalable_gp_clustered.csv \
    --sweep experts --values 2 4 8 16 32 --n 1024 --m 64 --d 1 \
    --methods nle nle_clustered poe poe_clustered gpoe gpoe_clustered \
    bcm bcm_clustered rbcm rbcm_clustered grbcm grbcm_clustered \
    moe moe_clustered --repetitions 3 --threads 1 --compiler gfortran \
    --flags='-O3 -funroll-loops'
python3 scripts/plot_scalable_gp.py \
    --input results/scalable_gp_clustered.csv \
    --prefix results/scalable_gp_clustered --metric train_seconds

python3 scripts/bench_scalable_gp.py \
    --output results/scalable_gp_dimension_current.csv \
    --sweep dimension --values 1 2 4 8 --n 1024 --m 64 --methods ski \
    --repetitions 3 --threads 1 --compiler gfortran \
    --flags='-O3 -funroll-loops'
python3 scripts/plot_scalable_gp.py \
    --input results/scalable_gp_dimension_current.csv \
    --prefix results/scalable_gp_dimension_current --metric train_seconds
```
