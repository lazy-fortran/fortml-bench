# Uniform empirical quantile transformer

FortML revision: 7f6de2cb73fc29c0d94e1b7d43680ab5e1d208b9  
Benchmark revision: bacffbffdd81d0fe0e6a1e2410692d65e639faa4  

The independent NumPy oracle uses 64 linearly interpolated order statistics per feature. It checks the piecewise-linear uniform CDF, inverse interpolation, endpoint policy, and the fixed-segment input JVP. The release checksum error is 1.421e-14 and the inverse error is 8.882e-16.

Normal-output quantiles, knot derivatives, power transforms, and resident CUDA execution remain explicit roadmap boundaries.
