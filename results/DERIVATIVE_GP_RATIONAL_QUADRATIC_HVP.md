# Rational-quadratic mixed-observation GP HVP

FortML revision: `d0e3a68e2f55f557a7aee29647aeb90bc2baa9a7`  
Benchmark revision: `4c5ce15fc0c6f2f3ed37c5fdb64ee473a6c98ba6+dirty`  

The independent NumPy dense covariance oracle checks the packed `[log variance, log lengthscale, log alpha, log noise]` HVP by central differences of the likelihood gradient. The Fortran CPU checksum is 2.798221193678e-01, with absolute error 3.236e-07; the measured mean time is 1.938e-05 s/HVP over 32 repetitions.

CUDA is recorded as the typed refusal code `3`; no host fallback is hidden behind the device row.
