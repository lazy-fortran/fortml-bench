# Random Fourier feature basis

This release lane checks the deterministic random Fourier feature transform
`sqrt(2/m) cos(w_k dot x + b_k)` for a three-component, two-input fixture.
The frequency matrix and phases are explicit, so the independent Fortran test
does not depend on compiler or RNG behavior. It checks the direct value
formula, intercept placement, central-difference input JVP and HVP products,
the scalar VJP adjoint identity, and the fixed-state zero-parameter contract.

Run the gate with:

```bash
python3 -B scripts/bench_random_fourier.py \
  --fortml ../fortml --output results/random_fourier.csv
```

The CSV records four CPU correctness rows and one explicit CUDA capability row.
The CPU rows are correctness records rather than throughput timings because the
subprocess includes a clean Fortran build. CUDA remains `unavailable` until a
resident basis executor is linked; there is no hidden host fallback. The
frequency sampler and seed are intentionally outside this fixed-state map and
must be recorded by the caller in a model manifest.
