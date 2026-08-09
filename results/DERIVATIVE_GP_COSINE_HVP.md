# Cosine mixed-observation GP HVP

FortML revision: `f74554dfde8b033216e319c7fd4511b7f9b61b4f`  
Benchmark revision: `5636617aba7e5237125e23d070e64e28020be03e`  

The independent NumPy dense covariance oracle checks the packed `[log variance, log lengthscale, log noise]` HVP by central differences of the likelihood gradient. The Fortran CPU checksum is 2.655504954147e-02, with absolute error 5.415e-07; the measured mean time is 6.539e-06 s/HVP over 32 repetitions.

CUDA is recorded as the typed refusal code `3`; no host fallback is hidden behind the device row.
