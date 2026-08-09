# Uniform empirical quantile transformer

FortML revision: b468ab4161c7267e832455f091075532b4587ba9  
Benchmark revision: 63b99bf061c51e2d4df39d597265111ae2dfd1b1  

The independent NumPy oracle uses 64 linearly interpolated order statistics per feature. It checks the piecewise-linear uniform CDF, inverse interpolation, endpoint policy, and the fixed-segment input JVP. The release checksum error is 1.421e-14 and the inverse error is 8.882e-16.

Normal-output quantiles, knot derivatives, power transforms, and resident CUDA execution remain explicit roadmap boundaries.
