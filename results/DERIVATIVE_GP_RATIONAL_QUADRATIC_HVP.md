# Rational-quadratic mixed-observation GP HVP

FortML revision: `c652b7ddde6586965ae4f5552f089bf02e8e0080`  
Benchmark revision: `575e152afde8c9378299039c6039c54a89d8952a`  

The independent NumPy dense covariance oracle checks the packed `[log variance, log lengthscale, log alpha, log noise]` HVP by central differences of the likelihood gradient. The Fortran CPU checksum is 2.798221193678e-01, with absolute error 3.236e-07; the measured mean time is 1.840e-05 s/HVP over 32 repetitions.

CUDA is recorded as the typed refusal code `3`; no host fallback is hidden behind the device row.
