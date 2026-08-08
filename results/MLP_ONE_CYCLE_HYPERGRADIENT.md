# One-cycle MLP schedule hypergradient benchmark

This lane checks the fixed full-batch affine MLP trajectory exposed by
`mlp_schedule_hypergradient_objective_t` when its schedule is
`MLP_SCHEDULE_ONE_CYCLE`. The fixture has five training rows, three validation
rows, five updates, two warm-up updates, and a total schedule shape of eight
updates. Its packed outer coordinates are
`[log(base_rate), log(l2), log(peak_rate_fraction),
log(final_rate_fraction)]`.

Run it from this repository with:

```bash
.venv/bin/python -B scripts/bench_mlp_one_cycle_hypergradient.py \
    --fortml ../fortml \
    --output results/mlp_one_cycle_hypergradient.csv
```

The independent NumPy oracle replays the affine MSE+L2 updates and uses
central differences only for the four outer coordinates. Rows are retained
only when the complete value, all four reverse-gradient components, and the
directional JVP agree within `5e-8`. The release app emits its complete array
before timing, so compilation or a partial output can only produce an explicit
unavailable row.

CUDA rows are explicit `unavailable` capability records: no resident MLP
trajectory kernel is linked and no host path is relabeled as device work. The
outer hyper-HVP row is likewise a typed refusal because nonlinear third
network derivatives are outside this bounded contract.
