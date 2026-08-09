# Ordinal-GP log-probability products

This lane covers the ordered latent-Gaussian GP log-probability API.  The
independent NumPy oracle evaluates adjacent normal-CDF differences, applies a
finite log floor, and finite-differences the latent mean/variance direction.
The FortML behavioral gate checks the fitted RBF posterior, packed
kernel/noise parameter JVP/VJP duality, query-input JVP/VJP duality, and the
output-clearing CUDA refusal.

Run it from `fortml-bench`:

```bash
FO_SCAN_FALLBACK=regex python3 -B scripts/bench_gp_ordinal_log_proba.py \
  --fortml ../fortml --output results/gp_ordinal_log_proba.csv
```

The output columns retain `classes()` order and are finite even in normal-CDF
tails.  The derivative rows use the exact chain rule for unclipped
probabilities. A value at the finite floor has zero derivative. CUDA is
reported as `unavailable` because the ordinal covariance and normal-CDF graph
is not resident.  The API returns `FORTNUM_NOT_IMPLEMENTED` and never stages
through the host.

The CSV records source and benchmark revisions, independent-oracle error, and
the release-test timing.  The lane is a correctness gate, not a claim of
CUDA parity or cumulative-likelihood inference, which remain explicit
roadmap work.
