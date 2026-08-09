# Basis-composed linear regression

This lane evaluates the same fitted multi-output linear model after three
independent feature maps: separable degree-three polynomial powers, clamped
order-four (cubic) B-splines, and Fourier sine/cosine components.  The NumPy
path fits the intercept and coefficients by least squares, then checks the
prediction JVP with a central difference and the VJP with the reverse-mode
adjoint identity.  B-spline values use an independent Cox--de Boor recursion.
the fixture stays away from knot breakpoints so the fixed-span derivative
contract is unambiguous.

The FortML gate runs `test_basis_linear_regression` and
`test_basis_cubic_spline`, and records the existing release-app CPU timing and
checksum rows for the polynomial/Fourier pipeline and cubic-spline evaluator.
Build and subprocess setup time is excluded from per-operation timings.  The
CUDA row is `unavailable` with `FORTNUM_NOT_IMPLEMENTED`: no resident
basis-linear executor is claimed.

Reproduce the record with:

```bash
python -B scripts/bench_basis_linear_regression.py \
  --fortml ../fortml --output results/basis_linear_regression.csv
```

The CSV records both clean repository revisions, the NumPy/compiler versions,
oracle tolerances, CPU timings/checksums, and the typed device boundary.
