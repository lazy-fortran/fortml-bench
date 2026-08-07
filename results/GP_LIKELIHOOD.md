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

The FortML release app emits the strict scalar protocol and is retained only
after the same independent NumPy checks pass. A checkout without the app or a
working compiler receives explicit `unavailable` rows. These host rows are not
GPU evidence: end-to-end GPU GP classification still requires resident
covariance, Laplace mode solve, and derivative buffers. The current CSV has 12
passing CPU rows and six explicit CUDA `unavailable` capability rows; CUDA
rows never contain a host timing.
