# Differentiable Chebyshev basis

This release lane measures the parameter-free first-kind Chebyshev basis
`T_1(x), ..., T_8(x)` for 4,096 samples and three inputs, with a shared
intercept. The Fortran implementation evaluates the three-term recurrence and
its exact first and second input derivatives. The NumPy harness independently
replays the recurrence, JVP, VJP, and scalar-contraction HVP; it does not call
FortML internals.

Run from this repository:

```bash
python3 scripts/bench_chebyshev_basis.py \
  --fortml ../fortml --output results/chebyshev_basis.csv
```

## Results

| Phase | Backend/device | Seconds/op | Oracle error | Result |
| --- | --- | ---: | ---: | --- |
| Value | FortML / CPU | 1.1298e-4 | 5.32e-11 | Pass |
| Input JVP | FortML / CPU | 2.4784e-4 | 6.82e-12 | Pass |
| Input VJP | FortML / CPU | 1.1739e-4 | 3.64e-12 | Pass |
| Input HVP | FortML / CPU | 1.3925e-4 | 1.36e-11 | Pass |
| Resident basis products | FortML / CUDA | — | — | Unavailable: typed `FORTNUM_NOT_IMPLEMENTED` |

The raw rows are in [`chebyshev_basis.csv`](chebyshev_basis.csv). They pin
FortML revision `cfee513e89472e5885ad4a58a8ce5ae1ec808168`, benchmark revision
`ddd777dc272f4bb7538c1cb59a2d01ceec3f3b2a`, GNU Fortran `-O2`, Python 3.14.6,
and NumPy 2.5.1. The resident-device row is a capability record, not a host
timing mislabeled as GPU support.

The benchmark covers the orthogonal basis and its input derivative products;
it does not claim fitted quantile transforms, sparse feature views, or resident
GPU execution.
