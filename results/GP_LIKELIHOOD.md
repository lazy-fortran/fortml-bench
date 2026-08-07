# GP classification likelihood

This lane measures the shared binary signed-margin likelihood primitive used
by FortML's Laplace classifier. The fixture has 4,096 deterministic margins
and a directional tangent. Both the logistic and probit links are evaluated.
Before timing, the independent NumPy oracle checks the summed log likelihood,
the directional JVP against a central difference, and the VJP dot-product
identity. The probit oracle uses the same Mills-ratio tail expansion as the
documented stable reference, so large negative margins remain finite.

Run:

```bash
python3 scripts/bench_gp_likelihood.py \
  --fortml ../fortml --output results/gp_likelihood.csv
```

The current release has no complete-array FortML release app for this scalar
primitive. The CSV therefore records NumPy oracle timings and explicit
`unavailable` FortML rows. Those rows are not GPU evidence: end-to-end GPU GP
classification still requires resident covariance, Laplace mode solve, and
derivative buffers.
