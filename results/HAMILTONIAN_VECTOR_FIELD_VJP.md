# Hamiltonian vector-field VJP

This lane checks the canonical Hamiltonian vector-field reverse product for a
separable Hamiltonian MLP. An independent NumPy quadratic Hamiltonian verifies
the JVP/VJP adjoint identity and a finite-difference state product. The FortML
release app repeats the packed parameter/state identity on the production MLP
HVP path. The benchmark records the CPU app timing and a typed CUDA-unavailable
row because no resident HNN derivative graph is linked.

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_hamiltonian_vector_field_vjp.py \
  --fortml ../fortml --output results/hamiltonian_vector_field_vjp.csv
```

The raw six-column record includes the source and benchmark revisions and keeps
the independent oracle separate from the release-app timing.
