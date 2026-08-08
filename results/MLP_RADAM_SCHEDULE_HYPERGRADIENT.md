# Scheduled RAdam hypergradient gate

This lane runs `test_mlp_radam_schedule_hypergradient`, an independent
Fortran behavioral oracle for the fixed full-batch scheduled RAdam objective.
The seven packed coordinates are
`[log(base_rate), log(l2), logit(beta1), logit(beta2), log(epsilon),
logit(min_rate_fraction), logit(decay_factor)]`. The fixture checks central
differences for cosine minimum-rate and exponential decay-factor schedules,
directional JVPs, scalar VJP adjointness, bounded FortOpt L-BFGS-B, malformed
optimizer/device refusals, RAdam branch-boundary refusal, and the typed outer
hyper-HVP refusal.

Run it with:

```bash
python -B scripts/bench_mlp_radam_schedule_hypergradient.py \
  --fortml ../fortml \
  --output results/mlp_radam_schedule_hypergradient.csv
```

The CSV separates CPU trajectory products, the outer-HVP refusal, and CUDA
refusal. No host fallback or approximate derivative is reported as a GPU
result; resident RAdam state and third network derivatives remain explicit
future capabilities.
