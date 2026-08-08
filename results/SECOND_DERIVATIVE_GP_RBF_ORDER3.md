# RBF order-three derivative-observation GP

This lane exercises the new exact-GP derivative contract for mixed value and
derivative observations through order three.  The RBF covariance blocks are
assembled independently in NumPy through total derivative order six.  The
oracle checks posterior mean and variance, a query input JVP by central
finite difference, likelihood-gradient/HVP finite differences in packed
log-variance/log-lengthscale/log-noise coordinates, and the minimum posterior
variance.  It does not import FortML or reuse its covariance implementation.

After those checks pass, the release app records CPU timings for prediction,
input JVP, input VJP, and the analytic likelihood HVP.  The device row is
explicitly `refused` with `FORTNUM_NOT_IMPLEMENTED` for CUDA: resident
derivative covariance/factorization kernels are not linked, so the benchmark
does not relabel a host fallback as GPU evidence.

The recorded release rows (gfortran, `-O3`, 24 observations and 16 queries)
are:

| phase | seconds/op | checksum | oracle error |
| --- | ---: | ---: | ---: |
| prediction | 6.779125e-6 | 3.2428472467273153 | 4.31e-14 |
| input JVP | 1.1605625e-5 | 0.1436540642153164 | 4.77e-10 |
| input VJP | 1.051725e-5 | 0.0460498030639650 | 3.45e-8 |
| hyperparameter HVP | 1.08145125e-4 | 1.0581959673351906 | 5.25e-7 |

These are one local run, not a hardware ranking.  The raw CSV also includes
the six independent-oracle metrics, the two passing Fortran tests, and the
typed CUDA boundary.

Run it from this checkout with:

```bash
python -B scripts/bench_second_derivative_gp_rbf_order3.py \
  --fortml ../fortml --output results/second_derivative_gp_rbf_order3.csv
```

The raw CSV records source and benchmark revisions, compiler flags, Python and
NumPy versions, oracle metrics, timing rows, and the typed device boundary.
The source release is `fortml` commit `7f944e8`; the script revision recorded
in this run is `7c542ed`.  The benchmark checkout was clean apart from the
ignored output CSV when the provenance was captured.

Limitations are deliberate: the current lane is one-dimensional, dense, and
CPU-only for the supported path; Matérn-5/2 remains supported through order
two, and order-three derivative observations are currently an RBF-only
extension.  Large-scale GPU parity and higher-order Matérn kernels require
resident generated covariance/factorization kernels.
