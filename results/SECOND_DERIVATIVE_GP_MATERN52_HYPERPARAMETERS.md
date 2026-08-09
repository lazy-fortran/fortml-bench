# Matérn-5/2 second-derivative GP hyperproducts

FortML revision: `3d688e70d6a35bb10e9e01e45cb260f91e2e7a7a`  
Benchmark revision: `6c2a2ed928952a83d46d045a3ea0473c0523c4f5+dirty`  

The independent NumPy fixture assembles value, first-derivative, and second-derivative covariance blocks through order four. It central differences the dense prediction functional, likelihood gradient, and query functional, then compares every checksum with the CPU release app. The largest CPU checksum error is 1.092e-05. Timings are recorded in the CSV.

The production Matérn-5/2 parameter jet uses the exact scaling identity for log lengthscale and a finite coincidence limit. No finite-difference fallback is used in the Fortran path. CUDA is recorded as the typed `FORTNUM_NOT_IMPLEMENTED` refusal until resident derivative covariance and factorization kernels are linked.
