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

Run it from this checkout with:

```bash
python -B scripts/bench_second_derivative_gp_rbf_order3.py \
  --fortml ../fortml --output results/second_derivative_gp_rbf_order3.csv
```

The raw CSV records source and benchmark revisions, compiler flags, Python and
NumPy versions, oracle metrics, timing rows, and the typed device boundary.
The source release is `fortml` commit `7f944e8`; the benchmark revision is
filled by the script from the clean benchmark checkout at run time.

Limitations are deliberate: the current lane is one-dimensional, dense, and
CPU-only for the supported path; Matérn-5/2 remains supported through order
two, and order-three derivative observations are currently an RBF-only
extension.  Large-scale GPU parity and higher-order Matérn kernels require
resident generated covariance/factorization kernels.
