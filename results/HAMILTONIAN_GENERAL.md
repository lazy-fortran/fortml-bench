# General nonseparable Hamiltonian evidence

This lane records the first general (H(q,p)) Hamiltonian MLP contract in
FortML. It is a correctness gate, not an end-to-end training or GPU timing
claim.

Run it from this repository with:

```bash
python3 -B scripts/bench_hamiltonian_general.py \
  --fortml ../fortml --output results/hamiltonian_general.csv
```

The NumPy rows are an independent analytic oracle for

```text
H(q,p) = (q² + p²)/2 + 0.2 q p
f(q,p) = (p + 0.2 q, -(q + 0.2 p)).
```

They check the canonical vector field and the Hamiltonian Jacobian identity
(A^T\Omega + \Omega A = 0) without importing FortML implementation details.
The FortML row runs `test_hamiltonian_mlp`, which checks general state/parameter
JVPs by central differences, the energy VJP adjoint identity, and the
separable map's independent symplectic-form and reversibility checks. It also
checks that `leapfrog` returns `FORTNUM_NOT_IMPLEMENTED` for a general model;
the explicit split map is not silently applied to a nonseparable Hamiltonian.

The CSV records source and benchmark revisions, compiler metadata, and the
explicit CUDA boundary. CUDA/OpenACC is `unavailable` because no resident
general-HNN model, derivative, or implicit-integrator graph is linked; no host
fallback or relabeled CPU timing is permitted.

| Lane | Evidence | State |
| --- | --- | --- |
| Analytic nonseparable field | Independent NumPy field and canonical-structure oracle | Pass |
| FortML general HNN | `test_hamiltonian_mlp` derivative, adjoint, and refusal gate | CPU pass when the gate passes |
| General HNN on CUDA/OpenACC | No resident model/integrator graph | Explicit unavailable row |
