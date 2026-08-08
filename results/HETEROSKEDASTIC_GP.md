# Supplied-noise heteroskedastic GP benchmark

`bench_heteroskedastic_gp.py` checks the supplied-noise heteroskedastic GP
against an independent NumPy diagonal-noise Cholesky oracle. It records three
behavioral contrasts: constant supplied noise reduces exactly to an ordinary
GP, a quiet/noisy split yields a tighter posterior in the quiet region, and
log-noise interpolation stays positive and reverts to the geometric mean far
from data. The FortML test also checks malformed shapes and non-positive noise
refusals.

Run the lane against the pinned FortML source:

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_heteroskedastic_gp.py \
  --fortml ../fortml --output results/heteroskedastic_gp.csv
```

The CSV records independent oracle errors, the FortML behavioral-gate timing,
and the explicit refusal for zero observation variance. The model takes
per-observation variances as given; jointly inferring the latent log-noise
process, its parameter derivatives, and resident CUDA execution remain open
and are not represented by CPU timing.
