# Scheduled AdamW trajectory hypergradient gate

This lane checks the fixed full-batch `fortml_mlp_adamw_schedule_hypergradient`
objective. Its eight packed coordinates are
`[log(base_rate), log(l2), log(weight_decay), logit(beta1), logit(beta2),
log(epsilon), logit(min_rate_fraction), logit(decay_factor)]`. The release
fixture checks cosine and exponential schedule branches, all packed central
differences, a directional JVP, scalar VJP adjointness, the FortOpt projected
L-BFGS-B callback, malformed optimizer/device requests, and the typed outer-HVP
boundary.

The benchmark script independently reconstructs the affine AdamW recurrence in
NumPy, including bias correction and decoupled shrinkage. It compares the
complete release-app value, eight gradients, and JVP before writing the CSV.
The recorded CPU row has maximum absolute error `1.227e-11`; CUDA, mixed
precision, and outer HVP remain explicit typed-refusal rows rather than host
timings.

Run it with:

```bash
python -B scripts/bench_mlp_adamw_schedule_hypergradient.py \
  --fortml ../fortml \
  --output results/mlp_adamw_schedule_hypergradient.csv
```
