# Robust Poisson and Student-t GP benchmark

`bench_robust_gp.py` checks the Laplace robust-GP observation models against an
independent NumPy latent-mode Newton oracle. The Poisson rows check mode
stationarity and positive rates on a rising-count fixture. The Student-t row
adds one large outlier and checks that the robust posterior moves less than a
Gaussian GP fit. The FortML test supplies the full behavioral and malformed-
input refusal gate.

Run the lane against the pinned FortML source:

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_robust_gp.py \
  --fortml ../fortml --output results/robust_gp.csv
```

The CSV records independent stationarity/rate errors, the robust-to-Gaussian
outlier error ratio, the FortML gate timing, and one typed refusal row covering
negative Poisson counts, unknown likelihoods, and non-positive Student-t
degrees of freedom. No GPU timing is reported because this robust Laplace
state and its non-Gaussian likelihood kernels have no resident device path.
