# Scheduled optimizer-group hypergradient benchmark

This release lane checks the fixed full-batch CPU SGD trajectory with two
contiguous parameter groups and a cosine learning-rate schedule. The packed
coordinates are

```text
[log(lr), log(l2), logit(min_rate_fraction), logit(decay_factor),
 log(multiplier_1), log(multiplier_2)]
```

The schedule shape is fixed (`total_updates=6`, four inner updates); the
decay-factor coordinate is intentionally inactive for cosine and must report an
exact zero product. The independent NumPy oracle replays the cosine recurrence,
global norm clipping, and group scaling, then central-differences all six
coordinates and a directional JVP. The FortML release app is accepted only
when value, every gradient component, and the JVP agree within `5e-9`.

Run:

```bash
.venv/bin/python -B scripts/bench_mlp_optimizer_group_schedule_hypergradient.py \
  --fortml ../fortml --output results/mlp_optimizer_group_schedule_hypergradient.csv
```

The CSV records provenance pins, CPU timing, and explicit CUDA `unavailable`
rows. Plateau, stochastic, validation-policy, and resident-device optimizer
state are outside this bounded benchmark.
