# Canonical symplectic-form residual evidence

This lane checks the map certificate used by structure-preserving models. For
canonical `[q,p]` coordinates, a map Jacobian `A` must satisfy
`A^T Omega A = Omega`, with `Omega=[0,I;-I,0]`. FortML exposes the packed
defect and exact residual and weighted-value JVP/VJP products through
`fortml_symplectic`. `symplectic_constraint_t` adapts the same term to the
composable physics objective.

The independent NumPy fixture is the unit harmonic-oscillator velocity-Verlet
map. It checks the form defect, the analytic map tangent against a central
difference, the residual JVP/VJP dot-product identity, and the value tangent.
The Fortran gate checks the same fixture, the diagnostic object, the
`physics_constraint_t` bridge, and transactional CUDA refusal. The benchmark
reports correctness-gate time, not end-to-end training throughput.

Run from this repository:

```bash
python3 scripts/bench_symplectic_residual.py \
  --fortml ../fortml --output results/symplectic_residual.csv
```

## Results

| Phase | Backend | Device | Result | Evidence |
| --- | --- | --- | --- | --- |
| Independent Verlet form and products | NumPy | CPU | Pass | `symplectic_residual.csv`, defect `0`, tangent error `3.397e-11`, adjoint error `2.776e-17` |
| Public residual and constraint contract | FortML | CPU | Pass | `test_symplectic` |
| Resident symplectic derivative graph | FortML | CUDA | Unavailable | Typed `FORTNUM_NOT_IMPLEMENTED`, no host fallback |

The rows record source and benchmark revisions, compiler, Python/NumPy
versions, and the refusal reason. This lane certifies the canonical map term.
It does not claim Lagrangian, Poisson, implicit general-Hamiltonian, learned
SympNet, or resident GPU coverage.
