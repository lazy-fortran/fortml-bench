# Sequential basis device dispatch

The sequential basis pipeline now has the same explicit device contract as the
fan-out and residual pipeline types.  CPU dispatch evaluates the composed
polynomial/Fourier chain and its input/parameter products.  CUDA remains a
typed `FORTNUM_NOT_IMPLEMENTED` boundary until a resident sequential basis
executor is linked.  Refusal leaves caller-owned output buffers untouched.

The independent NumPy oracle checks the mixed input/log-frequency JVP against a
central finite difference and checks the VJP adjoint identity.  The Fortran
gate exercises CPU transform, JVP, VJP, HVP dispatch and all four CUDA refusal
paths.

Run the lane with:

```sh
python3 scripts/bench_basis_sequential_device.py \
  --fortml ../fortml \
  --output results/basis_sequential_device.csv
```

Raw results are in [`basis_sequential_device.csv`](basis_sequential_device.csv).
