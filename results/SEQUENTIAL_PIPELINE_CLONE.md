# Transactional sequential basis-pipeline cloning

`sequential_basis_pipeline_t%clone` now deep-copies a fitted multi-stage basis
composition, including each stage's parameters, names, fitted state, and input
schema. The copy is built transactionally; an invalid source leaves an
existing destination unchanged. CPU device dispatch uses the same deep copy,
while a selected CUDA device returns `FORTNUM_NOT_IMPLEMENTED` until a
resident graph plan exists.

The independent NumPy oracle reconstructs the polynomial-to-Fourier fixture,
checks zero copied-output error, and checks that a parameter perturbation
changes only the clone. The release gate is
`test_sequential_pipeline_clone`.

## Results

| Phase | Backend/device | Result | Evidence |
| --- | --- | --- | --- |
| Independent copy/mutation oracle | NumPy / CPU | Pass | Copy error `0.0`, mutation effect `1.1945e-1` |
| Repeated deep clone | FortML / CPU | Pass | 5,000/5,000 clones, `2.0744e-7` s per clone, output error `0.0` |
| Resident graph clone | FortML / CUDA | Unavailable | Typed `FORTNUM_NOT_IMPLEMENTED`, destination preserved |

Raw rows are in [`sequential_pipeline_clone.csv`](sequential_pipeline_clone.csv).
The clean rows pin FortML revision `082845b7f790b8ccd9e9a8995db2a8955baeab65`,
benchmark revision `31a197067d9a810546fb5c561cd5a0f7b0eb6ccb`, GNU Fortran
`-O2`, Python `3.14.6`, and NumPy `2.5.1`. The timing-refresh commit is
`691cfaf`; it changes no correctness result.

The lane covers sequential clone/reset only. Fan-out/residual graph cloning,
serialized graph migration, sparse graph layouts, and resident accelerator
execution remain open roadmap items.
