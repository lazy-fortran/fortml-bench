# Recorded RBF MVM result

Run date: 2026-08-05. Workload: 2048 samples, 8 features, float64, 12 MVM
repetitions. The CPU comparison uses 16 physical cores on an AMD Ryzen 9 5950X.
The GPU comparison uses an NVIDIA GeForce RTX 5060 Ti. Every recorded row
passed the independent blocked NumPy oracle.

Fortran is within the 30-percent target of GPyTorch-KeOps on this workload.
The current scaling sweep uses the contiguous sample-major kernel from fortml
commit a205898. The CPU and GPU compiler, package, driver, source-commit, and
numerical error fields are in rbf_mvm_scaling.csv.

Plot:

https://box.sloppy.at/8ba9a.png

The Slopbox URL is public and expires after three days. This result covers the
RBF matrix-vector product only. Matched CG, log-determinant, and full GP
training workloads remain roadmap items.

## Scaling sweep

The scaling record covers 256, 512, 1024, 2048, and 4096 samples with the same
float64 constants and deterministic inputs. It uses twelve timed repetitions per
size. Dense PyTorch reaches `oom` at 4096 on the 16 GiB GPU. KeOps,
GPyTorch-KeOps, and Fortran continue and pass the independent oracle.

CPU plot:

https://box.sloppy.at/d69f0.png

GPU plot:

https://box.sloppy.at/076a0.png

The merged scaling data is in `rbf_mvm_scaling.csv`. At 4096 samples, Fortran
takes 4.37 ms per resident GPU MVM, compared with 6.63 ms for GPyTorch-KeOps
and 7.55 ms for KeOps. On the CPU lane, the same run takes 7.85 ms for
Fortran, 6.27 ms for GPyTorch-KeOps, and 7.70 ms for KeOps. Thus the GPU
curve is lowest throughout this sweep. The CPU endpoint remains within the
30-percent target but is not yet the lowest point.

The operation-level comparison is in
[OPERATION_PROFILE.md](OPERATION_PROFILE.md), with raw torch.profiler
tables in operation_profile_cpu.csv and operation_profile_cuda.csv.
