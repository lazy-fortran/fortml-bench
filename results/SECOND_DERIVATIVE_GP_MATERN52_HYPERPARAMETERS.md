# Matérn-5/2 second-derivative GP hyperproducts

FortML revision: `1618c8db7ae3511e9114ead0874f4cdffd44f92b`  
Benchmark revision: `e398cd35de6e008532315c0f5b8626d37baa1b87`  

The independent NumPy fixture assembles value, first-derivative, and second-derivative covariance blocks through order four. It central differences the dense prediction functional, likelihood gradient, and query functional, then compares every checksum with the CPU release app. The largest CPU checksum error is 1.092e-05. Timings are recorded in the CSV.

The production Matérn-5/2 parameter jet uses the exact scaling identity for log lengthscale and a finite coincidence limit. No finite-difference fallback is used in the Fortran path. CUDA is recorded as the typed `FORTNUM_NOT_IMPLEMENTED` refusal until resident derivative covariance and factorization kernels are linked.
