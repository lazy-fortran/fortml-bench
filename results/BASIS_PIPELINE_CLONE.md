# Transactional basis-pipeline cloning

This lane covers the bounded host-side clone/reset contract for a fitted
`basis_pipeline_t`. The fixture contains a degree-two polynomial map and one
radial map, for six output features. `clone` deep-copies stage state, fitted
metadata, and the input schema. An invalid source is refused without changing
an existing destination. CPU device dispatch uses the same deep copy; a
selected CUDA device returns `FORTNUM_NOT_IMPLEMENTED` and leaves the
destination unchanged because a resident graph clone is not implemented.

The independent NumPy oracle reconstructs the polynomial and radial outputs,
checks exact equality for the copied centre, and checks that changing the
clone's radial centre changes only the clone. The release gate is
`test_basis_pipeline_clone`.

## Results

| Phase | Backend/device | Result | Evidence |
| --- | --- | --- | --- |
| Independent copy/mutation oracle | NumPy / CPU | Pass | Copy error `0.0`; clone-centre mutation effect `1.8951e-1` |
| Repeated deep clone | FortML / CPU | Pass | 5,000/5,000 clones; `2.2963e-7` s per clone; output error `0.0` |
| Resident graph clone | FortML / CUDA | Unavailable | Typed `FORTNUM_NOT_IMPLEMENTED`; destination preserved |

Raw rows are in [`basis_pipeline_clone.csv`](basis_pipeline_clone.csv). They
pin FortML revision
`8b5d0cea50b90f4dc32b8e005f30b1947ea94a06`, benchmark revision
`0a93ab403737bab11c8367d93ddd89be25fd0331`, GNU Fortran `-O2`, Python
`3.14.6`, and NumPy `2.5.1`.

The lane intentionally covers cloning of the horizontal dense basis pipeline
only. Sequential/fan-out/residual graph cloning, estimator-wide clone/reset,
serialized clone migration, and resident accelerator graph state remain open
contracts.
