# Cubic B-spline basis

This release lane checks the public `make_cubic_spline_basis` constructor,
which fixes the FortNum spline order to four (degree three).  The fixture has
2,048 samples, two input columns, six strictly increasing breakpoints per
column, seventeen output features (a shared intercept plus eight features per
input), and no trainable parameters.

The independent NumPy oracle builds the clamped augmented knot vector itself
and evaluates every basis value with Cox--de Boor recursion.  Central
finite-difference products are taken away from knot breakpoints for the input
JVP, VJP, and scalar-contraction HVP checks.  The release application uses the
same fixed-span contract and reports one CPU timing row per product.  The
recorded checksum errors are below the release tolerances; a CUDA row is
`unavailable` with `FORTNUM_NOT_IMPLEMENTED`, preserving the typed host-only
boundary instead of claiming a transfer or resident-device implementation.

Reproduce the record with:

```bash
python -B scripts/bench_cubic_spline_basis.py \
  --fortml ../fortml --output results/cubic_spline_basis.csv
```

The raw CSV pins both repository revisions, compiler, flags, NumPy version,
oracle description, and the CPU timing/checksum rows.
