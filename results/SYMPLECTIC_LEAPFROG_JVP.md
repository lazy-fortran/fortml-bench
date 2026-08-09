# Separable leapfrog tangent evidence

This lane records the bounded differentiable-integrator contract added for a
separable Hamiltonian MLP. `hamiltonian_mlp_t%leapfrog_jvp` propagates packed
potential/kinetic parameter and initial-state directions through one exact
velocity-Verlet step. The implementation uses the MLP reverse-over-forward
HVP product at each split stage and does not finite-difference production
derivatives. The integration step is fixed in this product. General
nonseparable Hamiltonians retain a typed implicit-integrator refusal.

The NumPy row is an independent harmonic-oscillator map and tangent oracle. It
checks the analytic state tangent against a central difference and checks the
canonical symplectic form defect. The Fortran row runs
`test_hamiltonian_mlp`, including a joint parameter/state central-difference
oracle for the new tangent, primal-map equivalence, and the typed general-mode
refusal. The CUDA row is `unavailable`: no resident symplectic model or
derivative kernel is exposed, and there is no hidden host fallback.

Run from this repository:

```bash
python -B scripts/bench_symplectic_leapfrog_jvp.py \
  --fortml ../fortml \
  --output results/symplectic_leapfrog_jvp.csv
```

The CSV stores source and benchmark revisions, toolchain metadata, independent
oracle error, gate status, and the explicit device boundary. It is a
correctness gate, not a long-horizon throughput claim.

| Lane | Evidence | State |
| --- | --- | --- |
| Harmonic state tangent | Independent NumPy velocity-Verlet map | Pass |
| FortML separable tangent | `test_hamiltonian_mlp` parameter/state JVP gate | CPU pass when the gate passes |
| General HNN integrator | Typed `FORTNUM_NOT_IMPLEMENTED` refusal | Explicitly unavailable |
| CUDA/OpenACC tangent | No resident implementation | Explicitly unavailable |
