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

## License

MIT. See [LICENSE](LICENSE).
