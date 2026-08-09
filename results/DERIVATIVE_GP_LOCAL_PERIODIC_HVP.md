# Local-periodic derivative-GP mixed HVP

`bench_derivative_gp_local_periodic_hvp.py` independently assembles the
five-row, two-feature mixed value/first-derivative covariance for

```
k(s) = v exp(-s/(2 ell_envelope^2)
             - 2 sin(pi sqrt(s)/period)^2 / ell_periodic^2).
```

It central-differences the dense Cholesky likelihood gradient along a packed
four-kernel-coordinate plus log-noise direction. The FortML release app uses
the same fixture and is accepted only when its HVP checksum agrees with the
independent NumPy oracle.

The clean run produced the following records in
`results/derivative_gp_local_periodic_hvp.csv`:

| Backend | Device | Status | HVP checksum | Max absolute error | Time/op |
| --- | --- | --- | ---: | ---: | ---: |
| NumPy oracle | CPU | pass | -5.816532440878347e-02 | 0 | — |
| FortML | CPU | pass | -5.816479144864704e-02 | 5.3296e-07 | 4.7832e-05 s |
| FortML | CUDA | unavailable | — | — | — |

The CUDA row is intentionally `unavailable`: derivative-GP resident covariance
and factorization kernels are not linked, so the public API returns a typed
`FORTNUM_NOT_IMPLEMENTED` refusal rather than silently copying to the host.

Reproduce with:

```text
python -B scripts/bench_derivative_gp_local_periodic_hvp.py \
  --fortml ../fortml \
  --output results/derivative_gp_local_periodic_hvp.csv
```
