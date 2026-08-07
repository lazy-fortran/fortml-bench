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
scikit-learn `BernoulliNB` context row and an explicit FortML target refusal
until the matching release app is shipped:

```bash
.venv/bin/python -B scripts/bench_bernoulli_nb.py \
    --fortml ../fortml --output results/bernoulli_naive_bayes.csv
```

See [`results/BERNOULLI_NB.md`](results/BERNOULLI_NB.md) for the fixture,
oracle boundary, and release-app protocol.

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

The exact second-order XGBoost-style lane has its own workload and raw record.
It checks squared, binary logistic, and one-vs-rest multiclass depth-two
boosting against independent recursive NumPy gradient/Hessian oracles, then
records an explicit optional-XGBoost contextual row:

```bash
.venv/bin/python -B scripts/bench_xgboost.py \
  --fortml ../fortml --output results/xgboost_workloads.csv
```

See [`results/XGBOOST.md`](results/XGBOOST.md) for the regularisation settings,
oracle boundary, and recorded timings. The optional package row never turns a
different histogram or tree-growth policy into a bitwise comparison.

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
