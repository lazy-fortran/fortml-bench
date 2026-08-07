# CUDA correctness contracts

This lane is a correctness gate for four small resident CUDA paths and one
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

The AdamW row uses the same seven-step, five-parameter fixture as the
dedicated `cuda_adamw` lane. Its independent oracle includes bias-corrected
first and second moments and decoupled weight decay; the expected parameter
norm is `0.3149703604198818` and the state checksum is
`1.6129140383844665e-01`. The `run_cuda_adamw_state.sh` gate checks every
resident parameter and moment entry with a maximum error threshold of
`3e-13`.

The resident weighted-MSE row exercises `run_cuda_mse_plan.sh`. Creation copies
the target, prediction, and optional weights once; five execute calls reuse
those device buffers and return the same scalar as the independent NumPy
oracle. The native maximum error threshold is `3e-13`. This is a resident
no-autodiff reduction contract, distinct from the transfer-inclusive MSE row.

The recorded run used an NVIDIA GeForce RTX 5060 Ti (driver 610.43.03,
16,311 MiB), CUDA 13.3, nvfortran 26.5, and gfortran as the host compiler.
All five rows passed; the RMSprop and AdamW native maximum errors were
`1.11e-16`, the kNN label checksum matched exactly, and the CUDA MSE scalar
and the five resident-plan executions matched the independent value above. The CSV keeps the FortML and benchmark revisions,
compiler flags, device, and oracle boundary. Empty timing fields are
intentional. If `nvcc`, `nvfortran`, or a CUDA device is unavailable, the same
rows become explicit `skipped` records instead of being relabeled as CPU
measurements.

This gate covers resident kNN prediction, the no-autodiff RMSprop and AdamW
state kernels, and the transfer-inclusive weighted MSE reduction. It does not
establish CUDA support for MLP gradient assembly, RMSprop hypergradients,
staged XGBoost, or GP classification training; those remain separate workload
contracts.
