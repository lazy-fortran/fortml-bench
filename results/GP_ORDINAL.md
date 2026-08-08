# Latent-Gaussian ordinal GP benchmark

`bench_gp_ordinal.py` checks the ordered latent-Gaussian GP contract against an
independent NumPy oracle. The oracle evaluates adjacent normal-CDF
probabilities and finite-differences their directional derivative. The FortML
test adds a fitted RBF posterior, sorted integer labels, packed kernel/noise
parameter products, query-input JVP/VJP duality, and typed CUDA refusals.

Run the lane from `fortml-bench`:

```bash
FO_SCAN_FALLBACK=regex python3 -B scripts/bench_gp_ordinal.py \
  --fortml ../fortml --output results/gp_ordinal.csv
```

The CSV records the FortML and benchmark revisions, NumPy oracle error, and
the CPU contract-gate time. CUDA is recorded as `unavailable` because resident
ordinal covariance, normal-CDF, and reverse-product kernels are not linked;
the Fortran API returns `FORTNUM_NOT_IMPLEMENTED` rather than staging through
the host. The model is a latent-Gaussian ordered surrogate, so native
cumulative-likelihood and optimized-cut-point timings are intentionally not
reported by this lane.
