# GP feature workloads

Run date: 2026-08-06. The raw record is
[`gp_features.csv`](gp_features.csv). Times are milliseconds per complete
timed call. Deterministic input generation is outside the timer. Each timed
call contains the complete model work defined below.

| workload | FortML CPU | PyTorch/GPyTorch CPU | PyTorch/GPyTorch CUDA | FortML / CPU reference | within 30 percent |
|---|---:|---:|---:|---:|:---:|
| stochastic log determinant | 35.4 | 7.32 | 15.9 | 4.84 | no |
| predictive variance | 8.23 | 2.66 | 6.03 | 3.09 | no |
| derivative GP fit and predict | 0.526 | 0.289 | 0.845 | 1.82 | no |
| multi-output GP fit and predict | 0.0527 | 0.163 | 0.468 | 0.324 | yes |
| variational ELBO and predict | 0.0476 | 0.262 | 0.907 | 0.182 | yes |

The 30-percent gate passes for the multi-output and variational calls. It does
not pass for log determinant, predictive variance, or the derivative call.
Each verdict is confined to this CPU workload and timed-call definition.

FortML has no device-resident public path for these calls. The raw CSV retains
one `unsupported` CUDA row per workload. The CUDA reference timings therefore
describe PyTorch and GPyTorch only.

## Workload definitions

The stochastic log-determinant case uses a 64-by-64 RBF covariance plus a
0.35 diagonal shift, 64 trace probes, 48 Lanczos steps, and a fixed seed.
FortML and GPyTorch both execute an SLQ estimate. Their relative errors against
the dense NumPy log determinant are 0.42 percent and at most 2.33 percent,
respectively.

The predictive-variance case evaluates 16 test points for the same covariance
with a rank/iteration cap of 48. FortML currently performs a Lanczos product
for each cross-covariance column. GPyTorch LOVE constructs one inverse-root
approximation and reuses it for the 16 columns. The result, size, and
approximation budget match, but the batching strategy does not. The timing is
evidence about the two public implementation paths, not a kernel-level speedup
claim.

The derivative workload fits 48 mixed value/first-derivative observations with
two outputs and predicts 24 mixed queries. The multi-output workload fits 24
points and three outputs with a rank-two coregionalization matrix, then
predicts 12 points. The variational workload evaluates one ELBO on 64 points
with 12 inducing points and predicts 20 points. Its variational parameters are
fixed. No optimizer step is timed. The dense PyTorch paths assemble the same
covariances and perform the same complete products as the FortML calls.

Every backend is checked before timing against full arrays assembled by a
separate NumPy implementation. This is the external behavioral oracle. The
FortML program also retains direct formula checks, but the CSV does not treat
its shared FortNum solve as the independent oracle. Maximum relative errors
for derivative and multi-output results are below `1.7e-15`. The variational
maximum is below `1.8e-11`. Approximate spectral results use the stated
tolerances.

## Run conditions and reproduction

The CPU lane is pinned to one core of an AMD Ryzen 9 5950X. The GPU reference
lane uses an NVIDIA GeForce RTX 5060 Ti with driver 610.43.03. The run uses GNU
Fortran 16.1.1 with `-O3 -funroll-loops`, Python 3.14.6, NumPy 2.5.1, PyTorch
2.13.0+cu130, GPyTorch 1.15.2, and CUDA 13.0. The CSV records setup and build
times, memory, compiler and package versions, source revisions, the FortML
dirty-tree diff hash, and SHA-256 hashes of the driver, feature app, and
Lanczos source. Each Python workload/device pair runs in a fresh child process,
so its CPU peak is that process's maximum RSS rather than a cumulative driver
value. CUDA rows reset and record PyTorch's peak device allocation.

Reproduce the CSV and plot with:

```sh
.venv/bin/python -B scripts/bench_gp_features.py \
    --output results/gp_features.csv --plot results/gp_features.png \
    --repetitions 3
```

Do not run another `fo` build against the same FortML checkout while the
harness is active. The rendered comparison is
[`gp_features.png`](gp_features.png).
