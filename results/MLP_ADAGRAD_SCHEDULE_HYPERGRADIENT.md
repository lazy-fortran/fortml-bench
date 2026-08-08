# Scheduled Adagrad hypergradient gate

This lane checks the fixed full-batch `fortml_mlp_adagrad_schedule_hypergradient`
objective. The packed outer vector is
`[log(base_rate), log(l2), log(epsilon), logit(min_rate_fraction),
logit(decay_factor)]`. The Fortran fixture independently checks central
differences, directional JVPs, scalar VJP adjointness, FortOpt L-BFGS-B
integration, default schedule state, malformed options, and the typed CUDA
refusal. The benchmark records no throughput timing because the subprocess
includes the build and behavioral gate.

Run it with:

```bash
python -B scripts/bench_mlp_adagrad_schedule_hypergradient.py \
  --fortml ../fortml \
  --output results/mlp_adagrad_schedule_hypergradient.csv
```

The two CSV rows separate CPU trajectory products from the CUDA refusal. A
future resident optimizer kernel needs its own independent device oracle and
transfer counters before it can replace the refusal row.
