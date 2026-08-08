# Chronological time-series validation

This lane exercises `time_series_splitter_t` on 29 rows, four chronological
folds, three-row test windows, a two-row gap, and a seven-row rolling training
window. The independent NumPy oracle derives every one-based train/test index
from the blocked-window formula. It checks all rows before retaining timing;
the FortML scorer metadata is also checked by verifying that log-loss `0.2` is
oriented to the maximize value `-0.2`.

The recorded CPU run reports zero index error and zero metadata error. NumPy
took `1.76935e-05 s` per complete four-fold oracle generation; FortML took
`2.95371e-07 s` for the release-app split loop. These are control-plane index
timings on a tiny fixture, not end-to-end estimator training measurements. The
CSV is [`time_series_split.csv`](time_series_split.csv).

The source revision is `1982744ecd5331fedb5a3396d49d26b27fb9f70d`; the
benchmark revision is `0c05bc5005f1e26a4d2271e89b65030e0574c3d0`. Reproduce
with:

```bash
python3 -B scripts/bench_time_series_split.py \
  --fortml ../fortml --output results/time_series_split.csv
```

The CUDA row is `unavailable` by contract. Splitters and scorer/clone metadata
own CPU validation control-plane state; no host fallback is reported as
accelerator execution. Repeated K-fold, Monte Carlo scoring, and a generic
estimator clone implementation remain separate roadmap items.
