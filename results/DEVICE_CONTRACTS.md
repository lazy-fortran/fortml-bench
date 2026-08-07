# CUDA correctness contracts

This lane is a correctness gate for two small resident CUDA paths and one
transfer-inclusive metric reduction that are not represented by the CPU
release-app benchmarks. It deliberately records no device timing: a
correctness-only gate must not be read as a performance claim for the full
estimator or trainer.

Run it with:

```bash
python -B scripts/bench_device_contracts.py \
  --fortml ../fortml --output results/device_contracts.csv
```

The harness computes independent NumPy fixtures before invoking the native
tests. For kNN, it lexicographically orders squared distances by distance and
original row, giving labels `[-7, 11]` and checksum `4`. The native
`run_knn_classifier_cuda.sh` gate then checks those labels through the resident
CUDA plan. For RMSprop, the oracle independently performs five centered
updates with learning rate `0.08`, decay `0.8`, epsilon `1e-5`, and momentum
`0.2`; the expected final parameter norm is
`0.5731622547095053` and the state checksum is
`-1.8508517188642806`. The native
`run_cuda_rmsprop_state.sh` gate checks every downloaded parameter, square,
mean, and momentum-buffer entry against its own CPU recurrence.

For weighted multi-output MSE, the independent oracle applies row weights to
the squared residuals and divides by weight mass times output count; its
fixture value is `1.6923076923076923`. The native `run_cuda_metric.sh` gate
compiles the CUDA block reduction, exercises the Fortran binding with explicit
host/device transfers, and checks the complete scalar against that oracle. A
missing CUDA object is recorded as a typed `skipped` row rather than a host
fallback.

The recorded run used an NVIDIA GeForce RTX 5060 Ti (driver 610.43.03,
16,311 MiB), CUDA 13.3, nvfortran 26.5, and gfortran as the host compiler.
All three rows passed; the RMSprop native maximum error was `1.11e-16`, the
kNN label checksum matched exactly, and the CUDA MSE scalar matched the
independent value above. The CSV keeps the FortML and benchmark revisions,
compiler flags, device, and oracle boundary. Empty timing fields are
intentional. If `nvcc`, `nvfortran`, or a CUDA device is unavailable, the same
rows become explicit `skipped` records instead of being relabeled as CPU
measurements.

This gate covers resident kNN prediction, the no-autodiff RMSprop state
kernel, and the transfer-inclusive weighted MSE reduction only. It does not
establish CUDA support for MLP gradient assembly, RMSprop hypergradients,
staged XGBoost, or GP classification training; those remain separate workload
contracts.
