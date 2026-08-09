# Uniform empirical quantile transformer

FortML revision: ba72eb13cdf15962ad593a7c5ca196767a6938dc  
Benchmark revision: 896ffa19bc0186350bae3ef708863ef272ee7a37  

The independent NumPy oracle uses 64 linearly interpolated order statistics per feature. It checks the piecewise-linear uniform CDF, inverse interpolation, endpoint policy, and the fixed-segment input JVP. The release checksum error is 1.421e-14 and the inverse error is 8.882e-16.

Normal-output quantiles, knot derivatives, power transforms, and resident CUDA execution remain explicit roadmap boundaries.
