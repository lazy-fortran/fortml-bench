# Categorical variational-GP likelihood benchmark

This release lane checks the scalar positive softmax likelihood temperature
and stable categorical log-probability products of
`gp_variational_categorical_classification_t`.  The source probe uses
six one-dimensional observations, three sorted classes, and deterministic
nonzero inducing means.  FortOpt optimizes only the log-temperature while the
variational inducing state remains fixed.

The independent NumPy oracle reads the emitted latent means and variances and
recomputes the variance-corrected categorical logits,

```
z = exp(log_temperature) * mean / sqrt(1 + pi * variance / 8),
```

then evaluates a stable row-wise softmax. It computes the exact
log-temperature JVP and weighted categorical ELBO derivative. The release probe also emits
`predict_log_proba`, packed-parameter JVP, and input JVP values.  The
independent oracle checks the log values against the reconstructed
probabilities and checks both log-JVPs against the identity
`d log(p) = d p / p`. The Fortran test separately checks finite-difference and
adjoint products for the reverse paths. The HVP extension independently
replays the fixed-cotangent probability VJP and the fixed-state ELBO curvature
in the same log-temperature direction.  The clean run recorded zero maximum
absolute error for probabilities, probability JVPs, log probabilities and both
log-JVP identities, probability VJPs/HVPs, and the ELBO derivative/HVP.
the fitted scale was `3.3546262790251185e-4` after 14 iterations.  The ELBO
JVP was `-1.2492061083302581e-4`.

Run it with:

```bash
python -B scripts/bench_gp_categorical_likelihood.py \
  --fortml ../fortml --output results/gp_categorical_likelihood.csv
```

The CSV has eight rows: the independent oracle, FortML likelihood-only fit,
probability products plus probability HVP, log-probability products, ELBO
products plus ELBO HVP, and the CUDA device contract.  CUDA is reported as unavailable with
`FORTNUM_NOT_IMPLEMENTED` (status code `3`) for JVP and both HVP wrappers: the
inducing solve and categorical reduction are not resident.  The three
log-probability device wrappers return the same typed status and no host
fallback is counted as GPU support.
