# Resident CUDA RMSprop state gate

This lane wraps `fortml/test/run_cuda_rmsprop_state.sh`. The native plan keeps
parameters, running square averages, centered means, and optional momentum
state resident on the selected CUDA device; each optimizer step accepts a
device-resident gradient. The gate checks five updates of a four-parameter
fixture and refuses malformed creation, null gradients, and null download
destinations without changing the plan state.

Before accepting the native result, the benchmark independently reconstructs
the centered, momentum-enabled RMSprop recurrence in NumPy. The reported
native maximum error must be at most `2e-12`. The test subprocess does not
export a resident kernel timer, so compile-inclusive wall time is deliberately
not recorded as `seconds_per_operation`.

Run:

```bash
python3 scripts/bench_cuda_rmsprop.py \
  --fortml ../fortml --output results/cuda_rmsprop.csv
```

The CSV contains one independent CPU-oracle row and one native CUDA gate row.
A missing compiler or device is written as `unavailable`, never as a CPU
timing or a CUDA pass. This is a no-autodiff optimizer-state contract, not
full MLP gradient or hypergradient GPU evidence.
