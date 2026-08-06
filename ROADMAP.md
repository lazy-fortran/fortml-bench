# fortml-bench roadmap

This repository owns reproducible cross-engine evidence for fortml. A result
requires an independent oracle, matched mathematical work, recorded toolchain
metadata, and a committed raw record.

## Workloads

- [x] Complete matched RBF MVM runs on CPU and GPU for Fortran, dense PyTorch,
  KeOps, and GPyTorch with KeOps.
- [x] Add size sweeps that report runtime, the dense OOM boundary, and
  CPU/GPU scaling plots.
- [x] Add matched matrix-free CG solves with the same float64 tolerance,
  iteration cap, diagonal shift, unpreconditioned recurrence, and true-residual
  stopping check for dense PyTorch, KeOps, GPyTorch-KeOps, and nvfortran
  FortML.
- [ ] Add stochastic log-determinant and predictive-variance products.
- [ ] Add exact small-GP training and prediction comparisons.
- [x] Add regular-grid Toeplitz/Kronecker evidence with independent dense or
  structured oracles and resident OpenACC scaling records.
- [ ] Add compact-support sparse workloads using `fortsparse`, with fill-in,
  memory, and matched CPU/GPU diagnostics.
- [ ] Add derivative-observation and derivative-prediction workloads.
- [ ] Add multi-output and variational GP workloads.

## Evidence

- [x] Record gfortran and nvfortran compiler reports for the Fortran kernels.
- [x] Record PyTorch, GPyTorch, KeOps, CUDA, driver, and GPU revisions.
- [x] Generate comparison plots from committed CSV data.
- [x] Upload the first released plot to Slopbox and record its URL.
- [x] Establish the within-30-percent comparison against the best matched
  competitor for the first RBF MVM workload, precision, and device.
- [x] Publish a machine-readable CSV record and reproducibility script.

The within-30-percent target is a measurement gate. It is never inferred from
a different workload, precision, device, or residency policy.
