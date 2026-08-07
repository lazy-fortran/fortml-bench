# Physics-informed and structure-preserving model evidence

This is a bounded evidence lane for the current FortML API. It covers the
separable `hamiltonian_mlp_t` prototype and its independent structure checks.
The dedicated [general nonseparable Hamiltonian lane](HAMILTONIAN_GENERAL.md)
adds full-state HNN products and a typed split-integrator refusal. This report
does not claim PINN training, a symplectic GP, an HNN/LNN catalog, or a
finite-width network initialized exactly from an infinite-width GP.

The script runs the existing Fortran test, which checks energy and vector-field
products by central differences, the energy VJP by the adjoint identity, and
the velocity-Verlet map by a finite-difference symplectic-form defect and a
reverse-step check. Before that gate, it evaluates the same split-map algebra
for a unit harmonic oscillator using an independent NumPy oracle. A CUDA row
is retained as `unavailable`: the current physics prototype has no resident
model, residual, derivative, or optimizer graph.

Run from this repository:

```bash
python3 scripts/bench_physics_models.py \
  --fortml ../fortml --output results/physics_models.csv
```

The checked CSV records the source and benchmark revisions, compiler, timing
of the correctness gate, oracle description, and the explicit device boundary.
The timing is a test/build gate rather than a model-training throughput claim.

## Current contract

| Lane | Evidence | State |
| --- | --- | --- |
| Harmonic split-map algebra | Independent NumPy symplectic and reversibility oracle | Pass |
| FortML separable Hamiltonian MLP | `test_hamiltonian_mlp` finite-difference, adjoint, and structure checks | CPU pass when the gate passes |
| Physics graph on CUDA/OpenACC | No resident implementation; no host fallback | Explicit unavailable row |

The next scientific workloads are analytic oscillator, pendulum, Kepler, and
Hénon--Heiles trajectories, followed by manufactured PDE residuals. They need
matched seeds, integrators, collocation points, residual tolerances, and
long-horizon energy/symplectic diagnostics before entering the release matrix.
Physics-informed GP, symplectic-GP, Ghosttasking/Monge-GP, and GP/NN-limit
initialization rows remain planned until public equations and reference
implementations are pinned.
