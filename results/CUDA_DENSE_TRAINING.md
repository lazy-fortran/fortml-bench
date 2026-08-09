# Resident CUDA dense training

This lane covers a single dense linear layer with an explicitly uploaded resident batch. Gradients, Adam moments, and model parameters remain on the selected device across four updates. The native oracle covers SGD, Adam, and AdamW. Compute-sanitizer runs the same independent fixture.

- FortML revision: `6cf9c5b5efb71d2091c2cc1959528712e0e5605b`
- Benchmark revision: `20d3d4551e5ed25f6c66c24e2bef1e54c2dfb5e1`
- Native gate: `pass` (max error `2.429e-17`)
- Compute-sanitizer: `pass`
- Oracle parameter norms: `{'sgd': 0.4503899472104731, 'adam': 0.3163764026199537, 'adamw': 0.3986765926980302}`

The gate subprocess includes compilation and is not reported as a per-step performance number. See `fortml/docs/CUDA_DENSE_TRAINING.md` for the typed Fortran API and transfer-accounting contract.
