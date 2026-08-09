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
| Repeated deep clone | FortML / CPU | Pass | 5,000/5,000 clones, `1.9834e-7` s per clone, output error `0.0` |
| Resident graph clone | FortML / CUDA | Unavailable | Typed `FORTNUM_NOT_IMPLEMENTED`, destination preserved |

Raw rows are in [`sequential_pipeline_clone.csv`](sequential_pipeline_clone.csv).
The source implementation is pinned by FortML commit `b565652`; rerun the
command above after the source and benchmark commits are clean to refresh the
provenance fields.

The lane covers sequential clone/reset only. Fan-out/residual graph cloning,
serialized graph migration, sparse graph layouts, and resident accelerator
execution remain open roadmap items.
