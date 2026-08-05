# Recorded RBF MVM result

Run date: 2026-08-05. Workload: 2048 samples, 8 features, float64, 12 MVM
repetitions. The CPU comparison uses 16 physical cores on an AMD Ryzen 9 5950X.
The GPU comparison uses an NVIDIA GeForce RTX 5060 Ti. Every recorded row
passed the independent blocked NumPy oracle.

Fortran is within the 30-percent target of GPyTorch-KeOps on this workload:
about 9 percent slower on CPU, and about 68 percent faster on the resident GPU
lane. The CPU and GPU compiler, package, driver, source-commit, and numerical
error fields are in rbf_mvm.csv.

Plot:

https://box.sloppy.at/8ba9a.png

The Slopbox URL is public and expires after three days. This result covers the
RBF matrix-vector product only. Matched CG, log-determinant, and full GP
training workloads remain roadmap items.
