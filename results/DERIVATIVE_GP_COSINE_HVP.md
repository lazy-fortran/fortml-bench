# Cosine mixed-observation GP HVP

FortML revision: `47dce7e36c1bb1531a99dcaf366fff60accb5473+dirty`  
Benchmark revision: `8149aca8afb9c2923bb51778df43a35c4088265c+dirty`  

The independent NumPy dense covariance oracle checks the packed `[log variance, log lengthscale, log noise]` HVP by central differences of the likelihood gradient. The Fortran CPU checksum is 2.655504954147e-02, with absolute error 5.415e-07; the measured mean time is 8.422e-06 s/HVP over 32 repetitions.

CUDA is recorded as the typed refusal code `3`; no host fallback is hidden behind the device row.
