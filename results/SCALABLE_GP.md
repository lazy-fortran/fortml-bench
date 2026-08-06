# Scalable GPs: measured cost against the review's claimed complexity

Reference: H. Liu, Y.-S. Ong, X. Shen and J. Cai, "When Gaussian Process Meets
Big Data: A Review of Scalable GPs", *IEEE Transactions on Neural Networks and
Learning Systems* 31(11):4405-4423, November 2020,
[doi:10.1109/TNNLS.2019.2957109](https://doi.org/10.1109/TNNLS.2019.2957109).

## What "matching the paper" can and cannot mean

The review contains **no numeric result tables**. Its reproducible content is:

* the one-dimensional toy of Figs. 4 and 5 — `y(x) = sinc(x) + eps` with
  `eps ~ N(0, 0.04)` and 120 training points — and the qualitative behaviours
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

## Behaviours reproduced (fortml `test_review_toy`)

| Paper's statement | Where | Check |
| --- | --- | --- |
| SoR is overconfident leaving the training data | Fig. 4 caption | SoR variance is below DTC's at every test point, and collapses below `1e-8` far from the inducing set |
| FITC captures heteroscedasticity in variance | Fig. 4 caption | FITC predictive variance varies by more than 5 % across the range |
| VFE approximates the full GP well | Fig. 4 caption | collapsed VFE predictive mean is within 0.15 of the exact GP everywhere |
| DTC variance grows to the prior away from the inducing points | Fig. 4 caption | DTC variance equals the prior to `1e-8` far away |
| PoE gives overconfident variance | Sec. IV-C | PoE aggregate is sharper than *every* individual expert at all 40 test points |
| GPoE and MoE suppress that | Sec. IV-C | neither is sharper than PoE anywhere; GPoE returns exactly to the prior far away |
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
| nle, poe, gpoe, bcm, rbcm, grbcm, moe | `O(M m0^3)` with `m0 = n/M` | +2.00 to +2.05 | rising toward the cubic-in-`m0` regime |

The local experts are claimed to be cubic in the per-expert size `m0`, which at
fixed `M` means cubic in `n`; the measured +2.0 says these sizes are not yet
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

All seven local methods measure a slope of −1.04 to −1.08 in `M`. The claim is
`O(M m0^3) = O(n^3 / M^2)`, i.e. slope −2; again the measured value says the
per-expert problems are too small to be asymptotic at `n = 1024`, and the
direction and the fact that more experts is cheaper are as claimed.

#### Training time against the input dimension

Fixed `n = 1024`, `m = 64`, `M = 8`, over `d` in {1, 2, 4, 8}, with the extra
coordinates carrying no signal so accuracy stays comparable.

Every method measures a slope between −0.40 and +0.17, i.e. **no method's cost
is materially driven by `d` here**. That is expected: the kernel evaluation is
`O(d)` and every method's dominant term is a factorization or a solve whose
size does not involve `d`. The review's own scalability discussion (Sec. VI-A)
concerns high-dimensional *capability*, not cost, and notes that the grid-based
methods degrade with `d` because `m` grows exponentially — a limit reached by
construction here, since the SKI path in this repository is one-dimensional and
is excluded from this sweep.

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
* The local aggregations sit between: `grbcm` and `nle` improve fastest
  (−0.86, −0.87), `poe`/`bcm` track the exact GP, and `moe` is the weakest of
  them.

## Is the KeOps-style matrix-free lane good enough on its own?

On this evidence, **for accuracy yes, for cost no — and the crossover is what
matters**.

* Accuracy: the matrix-free lane is *exact*. Its SMSE matches the dense exact
  GP to eight digits (`1.61255278E-02` against the same value at `n = 1024`),
  and with the LOVE Lanczos variance its mean negative log predictive density
  (−1.92) matches the dense GP's. No approximation can beat it, because it is
  not an approximation.
* Memory: it is the best in the study, flat in `n` where the dense GP is
  quadratic. This is the reason to use it.
* Training time: it is the *worst* of every method at these sizes — 0.571 s at
  `n = 1024` against 0.0136 s for the dense Cholesky and 0.002 s for the sparse
  approximations. Its slope is +2.18 against the dense +2.55, so it wins
  eventually, but not before `n` is far larger than anything measured here.
* Prediction: 51.8 s at `n = 1024`, three orders of magnitude worse than every
  other method, because the LOVE variance runs 20 Lanczos steps per test point
  and each step is a full `O(n^2)` product. Predictive variance is where the
  matrix-free lane is genuinely weak.

So the honest answer is that the matrix-free lane replaces the *dense exact* GP
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

## Reproduce

```sh
python3 scripts/bench_scalable_gp.py --output results/scalable_gp_samples.csv \
    --sweep samples --values 256 512 1024 --repetitions 2
python3 scripts/plot_scalable_gp.py --input results/scalable_gp_samples.csv \
    --prefix results/scalable_gp_samples --metric train_seconds
```

Sweeps `inducing`, `experts` and `dimension` take `--sweep` with the matching
name. `--methods` restricts the set.
