# Benchmark provenance

The benchmark compares source revisions, not moving development trees. Each
run records the `fortml` and `fortnum` commit, compiler versions, Python
package versions, CUDA runtime, driver, GPU model, precision, problem shape,
residency mode, warm-up count, repetitions, and correctness error.

The independent RBF oracle evaluates the stated pairwise formula in blocked
NumPy loops. It is separate from the dense PyTorch, KeOps, GPyTorch, and
Fortran implementations.

The classifier lanes use the same provenance rule. The softmax and MLP
classification record is gated by independent NumPy damped-Newton and
full-batch Adam implementations before any FortML, scikit-learn, or PyTorch
timing is retained. Missing optional packages and release targets are written
as explicit refusal rows, and a FortML pass requires complete probabilities
and labels in addition to a successful process exit.

The relaxed Bernoulli Naive Bayes record applies the same gate to a direct
NumPy Bernoulli likelihood, stable log-softmax, and analytic input JVP.  Its
scikit-learn context uses `binarize=None` so the relaxed `[0.1,0.9]` fixture is
not silently thresholded.  The FortML classifier source is pinned separately
from the release-app target: until that target is present, the raw record
retains explicit `unavailable` rows and makes no timing or device claim.

The KeOps and GPyTorch adapters follow their public PyTorch interfaces. The
Fortran adapter invokes the pinned `fortml` benchmark entry point and records
the source revision it used. No competitor source is linked into the MIT
Fortran libraries.

The first recorded run uses 2048 samples, 8 features, float64, and 12 timed
MVM repetitions. Its CPU lane uses 16 physical CPU cores and its GPU lane uses an
RTX 5060 Ti with resident and transfer-inclusive timings. The direct oracle
must pass before a timing is written to the CSV.

## Composite run (2026-08-06)

The KeOps-style static composite run uses fortml `5ff3d80` and fortnum
`c5bea47`, eight features, float64, variance 1.4, lengthscale 0.7, constant
variance 0.2, and diagonal shift 0.08. It evaluates the RBF-plus-constant
formula with a direct blocked NumPy pairwise oracle before timing. Sizes are
256 through 4096 with five repetitions, extended to 8192 and 16384 with three
repetitions. The CPU lane uses nvfortran 26.5 with `-O3 -mp=multicore` on 16
physical cores. The GPU lane uses nvfortran 26.5 with `-O3 -acc` on an NVIDIA
GeForce RTX 5060 Ti. Both resident and transfer-inclusive rows are recorded.
Composite GPU rows use fortml's resident generic CUDA kernel plan through the
opaque C ABI. The plan is created once per benchmark and reused across timed
MVMs.

The Python lanes use PyTorch 2.13.0+cu130, GPyTorch 1.15.2, and pykeops 2.3.
The GPyTorch adapter is its KeOps RBF operator plus the same explicit constant
rank-one term, so the mathematical work matches the Fortran and KeOps lanes.
All non-OOM rows pass the independent oracle. Dense PyTorch CUDA is retained
as `oom` at the capacity boundary rather than being silently omitted. The
generic CUDA plan was separately checked by a direct C++ pairwise oracle for
matvec and two-RHS matmat before this sweep.

## Scalable-GP review comparison

The reference for the scalable-GP study is H. Liu, Y.-S. Ong, X. Shen and
J. Cai, "When Gaussian Process Meets Big Data: A Review of Scalable GPs", IEEE
Transactions on Neural Networks and Learning Systems 31(11):4405-4423, 2020,
doi:10.1109/TNNLS.2019.2957109. The article is not redistributable and is
recorded by citation and DOI only.

That review publishes no numeric result tables. Its reproducible content is the
one-dimensional toy of Figs. 4 and 5, the qualitative behaviours stated in
those captions and in Section IV-C, the complexity claims of Fig. 2 and
Sections III to V, the library list of Table I, and the data set list of
Table II. `results/SCALABLE_GP.md` states which half of the comparison each
result belongs to. No number is attributed to the paper that the paper does not
contain.

`reference_revisions.tsv` pins each public repository from Table I by commit
and by the SHA-256 of `git archive` at that commit.
`scripts/fetch_reference_implementations.sh` fetches exactly those commits into
the ignored `.provenance/reference_implementations` tree, verifies the archive
checksums, and writes the local verification manifest. It exits unsuccessfully
if any fetch, revision, or checksum differs. Archive-distributed packages are
recorded by location rather than downloaded. No third-party source is linked
into the MIT Fortran libraries.

The fixture itself lives in `fortml` as `fortml_review_toy`, so the benchmark
and the correctness tests share one definition of the paper's problem.
