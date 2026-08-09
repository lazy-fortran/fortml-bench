# GP classifier prediction through fit

This lane differentiates weighted binary Laplace-GP predictions with respect
to the packed RBF kernel coordinates `[log variance, log lengthscale]`.  The
training rows, labels, weights, likelihood kind, and converged Newton branch
remain fixed.  The product includes the implicit mode tangent, likelihood
curvature, prior and posterior solves, latent mean and variance, and the final
probability map.

The independent NumPy oracle reimplements the weighted Laplace iteration and
refits the classifier at both central parameter probes.  It checks three query
means, three variances, and three positive-class probabilities for logistic
and probit likelihoods.  The largest absolute FortML errors are `3.58e-10`
for logistic prediction and `9.61e-10` for probit prediction.

The recorded CPU times are `5.30 us` per logistic JVP and `5.62 us` per
probit JVP over 32 repetitions.  CUDA requests return the typed
`FORTNUM_NOT_IMPLEMENTED` boundary without changing the output arrays or
performing a hidden host fallback.

The machine-readable evidence is
`results/gp_classification_implicit_prediction.csv`.  It pins FortML
`9c1dbdf7e1eb54dd550c7ffc976bf38deb604c6d` and benchmark revision
`4a931bae2c8fa4bd2c134704c0f50687f22b8ef2`, both without dirty markers.
