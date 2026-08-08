# Resident CUDA Adagrad state gate

This lane wraps `fortml/test/run_cuda_adagrad_state.sh`. The native plan keeps
parameters and accumulated-square state resident on the selected CUDA device;
each optimizer step accepts a device-resident gradient. The CUDA test owns
compilation and launch and checks eight updates of a five-parameter fixture.

Before accepting the native result, the benchmark independently reconstructs
the canonical Adagrad recurrence in NumPy. The reported native maximum error
must be at most `2e-13`. The test subprocess does not export a resident kernel
timer, so compile-inclusive wall time is deliberately not recorded as
`seconds_per_operation`.

Run:

```bash
python3 scripts/bench_cuda_adagrad.py \
  --fortml ../fortml --output results/cuda_adagrad.csv
```

The CSV contains one independent CPU-oracle row and one native CUDA gate row.
A missing compiler or device is written as `unavailable`, never as a CPU
timing or a CUDA pass. This is a no-autodiff optimizer-state contract, not
full MLP gradient or hypergradient GPU evidence.
