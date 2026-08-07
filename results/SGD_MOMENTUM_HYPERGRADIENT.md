# SGD momentum trajectory hypergradient benchmark

This lane checks the exact fixed full-batch SGD momentum trajectory against an
independent NumPy implementation. The packed outer variable is

```text
[ log_learning_rate, log_l2, momentum ]
```

The fixture uses the two-parameter linear MLP release workload: five training
rows, three validation rows, initial parameters `[0.15, -0.1]`, four updates,
learning rate `0.12`, L2 `0.07`, and momentum `0.31`. Each update is

```text
velocity <- momentum * velocity + gradient
theta    <- theta - learning_rate * velocity
```

The NumPy oracle obtains all three gradient components and a directional JVP by
central finite differences with `h=2e-6`. The complete FortML value/gradient/JVP
array must agree within `3e-10` before its repeated CPU timing is retained.

The release target is `fortml_bench_sgd_momentum_hypergradient`. It writes the
complete checked array under
`FORTML_BENCH_SGD_MOMENTUM_HYPERGRADIENT_ORACLE` and emits the
`sgd_momentum_hypergradient_value_gradient` CPU timing. CUDA rows are explicit
`unavailable` capability refusals until resident optimizer-state derivatives
exist; no host timing is relabeled as device evidence.

Run against a FortML checkout that contains the release app:

```bash
.venv/bin/python -B scripts/bench_sgd_momentum_hypergradient.py \
  --fortml ../fortml --output results/sgd_momentum_hypergradient.csv
```

Use `--skip-fortml` to regenerate only the independent NumPy rows while
retaining explicit skipped FortML rows.
