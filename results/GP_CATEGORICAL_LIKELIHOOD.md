# Categorical variational-GP likelihood benchmark

This release lane checks the scalar positive softmax likelihood temperature
added to `gp_variational_categorical_classification_t`.  The source probe uses
six one-dimensional observations, three sorted classes, and deterministic
nonzero inducing means.  FortOpt optimizes only the log-temperature while the
variational inducing state remains fixed.

The independent NumPy oracle reads the emitted latent means and variances and
recomputes the variance-corrected categorical logits,

```
z = exp(log_temperature) * mean / sqrt(1 + pi * variance / 8),
```

then evaluates a stable row-wise softmax, its exact log-temperature JVP, and
the weighted categorical ELBO derivative.  The clean run recorded zero maximum
absolute error for probabilities, probability JVPs, and the ELBO derivative;
the fitted scale was `3.3546262790251185e-4` after 14 iterations.  The ELBO
JVP was `-1.2492061083302581e-4`.

Run it with:

```bash
python -B scripts/bench_gp_categorical_likelihood.py \
  --fortml ../fortml --output results/gp_categorical_likelihood.csv
```

The CSV has five rows: the independent oracle, FortML likelihood-only fit,
probability products, ELBO products, and the CUDA device contract.  CUDA is
reported as unavailable with `FORTNUM_NOT_IMPLEMENTED` (status code `3`): the
inducing solve and categorical reduction are not resident, and no host
fallback is counted as GPU support.
