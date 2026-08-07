# Scheduled MLP hypergradient benchmark

This lane checks the complete scheduled full-batch MLP trajectory product
against an independent NumPy tanh network.  The fixture has 72 training rows,
24 validation rows, a `3-8-1` network, eight updates, and a warm-up-plus-cosine
schedule (`warmup=2`, `total=12`, minimum fraction `0.1`).  The NumPy oracle
implements the Fortran parameter packing and backpropagation independently and
uses central differences only for the outer four-parameter check.

Run it from this repository with:

```bash
.venv/bin/python -B scripts/bench_mlp_schedule_hypergradient.py \
    --fortml ../fortml \
    --output results/mlp_schedule_hypergradient.csv
```

The release app emits a complete-array CSV before timing.  Rows are retained
only when the validation value, all four packed reverse-gradient components,
and a directional JVP agree with the NumPy oracle (`max_abs_error <= 5e-8`).
The checked-in run has six CPU correctness/timing rows with maximum error
`1.53e-12`; the CPU value/JVP timing is approximately `1.25e-3 s` per app
operation on the recorded GNU Fortran host.

CUDA rows are explicit `unavailable` capability records.  The current MLP
trajectory has no resident CUDA kernel, so the benchmark never relabels a
host run as GPU work or hides a transfer.  This boundary is independently
tested by the FortML release test.

The outer objective is wrapped in FortOpt L-BFGS-B in the FortML API.  A
second-order hyper-HVP would require third derivatives of the nonlinear MLP;
this lane intentionally reports the exact JVP/VJP contract rather than a
finite-difference approximation.
