# Resident CUDA AdamW state gate

This lane wraps `fortml/test/run_cuda_adamw_state.sh`.  The native plan keeps
parameters, first and second moments, and the update counter resident on the
selected CUDA device; each step accepts a device-resident gradient.  The
CUDA test owns compilation and launch and checks seven updates of a
five-parameter fixture.

Before accepting the native result, the benchmark independently reconstructs
the bias-corrected AdamW moments and decoupled weight decay in NumPy.  The
reported native maximum error must be at most `3e-13`.  The test subprocess
does not export a resident kernel timer, so its wall time (which includes
compilation) is deliberately not recorded as `seconds_per_operation`.

Run:

```bash
python3 scripts/bench_cuda_adamw.py \
  --fortml ../fortml --output results/cuda_adamw.csv
```

The CSV has one independent CPU-oracle row and one native CUDA gate row.  A
missing compiler or device is written as `unavailable`, never as a CPU timing
or a CUDA pass.  Full MLP gradient assembly and hypergradient GPU timing are
separate contracts and are not implied by this state-kernel gate.
