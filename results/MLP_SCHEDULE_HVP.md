# Scheduled MLP outer-HVP benchmark

This release lane checks the fixed full-batch affine trajectory exposed by
`mlp_schedule_hypergradient_objective_t` with a two-update linear warm-up and
cosine tail.  The packed outer coordinates are
`[log(base_rate), log(l2), logit(min_rate_fraction),
logit(decay_factor)]`; the decay-factor coordinate is intentionally inactive
for this schedule family and remains an exact zero.

The NumPy implementation in `scripts/bench_mlp_schedule_hvp.py` is an
independent affine recurrence.  It central-differences the complete packed
validation gradient to obtain the HVP oracle, then gates the release app on
value, gradient, JVP, and HVP agreement before recording CPU timings.  The
latest CSV records a maximum absolute error of `2.31e-8`, with value/gradient
and HVP timings of approximately `7.79e-5 s` and `4.71e-4 s` per operation.

Run it with:

```bash
.venv/bin/python -B scripts/bench_mlp_schedule_hvp.py \
    --fortml ../fortml --output results/mlp_schedule_hvp.csv
```

Cosine, warm-up, and one-cycle rate fields now expose analytic raw rate second
products, transformed log/logit outer HVPs, and FortOpt-compatible value and
gradient callbacks on the CPU affine slice.  Nonlinear networks, metric
plateau branches, and CUDA remain typed refusals; no host trajectory is
reported as resident-device work.
