# Multi-output GP products

This lane covers the exact intrinsic-coregionalization implementation in
`fortml_multi_output_gp`. It uses `B = W W^T + diag(independent)`, a shared
RBF input kernel, output-major internal stacking, and the public
`(sample,output)` arrays. The independent NumPy oracle assembles the dense
`B ⊗ K + noise I` covariance and solves it directly; it does not call FortML.

The gate checks the posterior-mean sum and query-input JVP norm against that
oracle. The packed parameter JVP is timed through the differentiated dense
solve, and the parameter and prior-covariance VJPs must satisfy the scalar
adjoint identity to `2e-10`. The prior covariance JVP is checked against an
independent Kronecker derivative oracle. The benchmark app is a small
correctness-gated release protocol,
not a claim of large-scale GPU parity.

| phase | backend | device | result | seconds/op | metric | value | max error |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| predict | FortML | CPU | pass | 0 | mean sum | 13.867292706800631 | 1.42e-14 |
| input JVP | FortML | CPU | pass | 1.25e-4 | JVP L2 | 0.16567441154738527 | 2.78e-17 |
| parameter JVP | FortML | CPU | pass | 2.50e-4 | JVP L2 | 0.028853299556143262 | oracle-independent product |
| parameter VJP | FortML | CPU | pass | 5.00e-4 | adjoint error | 1.46e-15 | 1.46e-15 |
| prior covariance parameter JVP | FortML | CPU | pass | 2.50e-4 | JVP L2 | 6.095470309385365 | independent NumPy product |
| prior covariance parameter VJP | FortML | CPU | pass | 2.50e-4 | adjoint error | 1.01e-13 | 1.01e-13 |
| products | FortML | CUDA | unavailable | — | typed refusal | — | — |

The exact timings are machine-dependent; the checked CSV records Python,
NumPy, compiler, FortML, and benchmark revisions. Re-run from this repository
with:

```bash
python -B scripts/bench_multi_output_gp_products.py \
  --fortml ../fortml --output results/multi_output_gp_products.csv
```

CUDA is deliberately reported as unavailable (`FORTNUM_NOT_IMPLEMENTED`) until
resident coregionalization, factorization, and derivative kernels are linked;
the CPU path is never relabelled as an accelerator measurement.
