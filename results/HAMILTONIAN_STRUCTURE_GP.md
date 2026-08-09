# Separable Hamiltonian structure-aware GP initializer

This lane compares two independent finite-feature kernel-ridge solves for the
separable Hamiltonian components `V(q)` and `T(p)` with the FortML release app.
The fixture has 256 samples, two canonical coordinates, 16 `tanh` hidden units,
and regularization `0.1`. NumPy replays FortML's deterministic seeded hidden
layers and solves `(ZᵀZ + λI)C = ZᵀY` independently for each component.

Run it from the benchmark repository with:

```bash
python3 -B scripts/bench_hamiltonian_structure_gp.py \
  --fortml ../fortml --output results/hamiltonian_structure_gp.csv
```

The FortML row is retained only when both potential and kinetic fit RMSEs are
within `2e-12` of the NumPy oracle and the reported structure defect is below
`2e-13`. A successful apply changes only the two final affine layers, so the
defect certificate is zero; this is a finite-width topology certificate, not an
infinite-width GP claim. The CUDA row is typed `unavailable`: resident
structure-aware Hamiltonian GP/MLP kernels are not implemented and no host
timing is relabeled as a GPU result.
