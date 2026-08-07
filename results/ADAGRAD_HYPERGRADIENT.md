# Adagrad trajectory hypergradient benchmark

This lane checks the exact fixed full-batch Adagrad trajectory objective against
an independent NumPy implementation. The packed outer variable is

```text
[ log_learning_rate, log_l2, log_epsilon ]
```

The NumPy oracle uses the two-parameter linear MLP fixture from the FortML
release app: five training rows, three validation rows, initial parameters
`[0.15, -0.1]`, four updates, learning rate `0.12`, L2 `0.07`, and epsilon
`0.03`. It evaluates

```text
G <- G + gradient**2
theta <- theta - learning_rate * gradient / (sqrt(G) + epsilon)
```

and obtains all three gradient components and the directional JVP by central
finite differences with `h=2e-6`. CPU oracle work is timed only after the
products are finite. The complete FortML value/gradient/JVP array must agree
within `3e-10` before its repeated CPU timing is retained.

The release target is `fortml_bench_adagrad_hypergradient`. It writes the
complete checked array under
`FORTML_BENCH_ADAGRAD_HYPERGRADIENT_ORACLE` and then emits the
`adagrad_hypergradient_value_gradient` CPU timing. The CSV records Python and
NumPy versions, compiler/flags, and both the FortML and benchmark revisions.
CUDA rows are explicit `unavailable` capability refusals: no host timing is
relabeled as device evidence while the complete MLP state-derivative and
residency contract is still missing.

Run against a FortML checkout that contains the release app:

```bash
.venv/bin/python -B scripts/bench_adagrad_hypergradient.py \
  --fortml ../fortml --output results/adagrad_hypergradient.csv
```

Use `--skip-fortml` to regenerate only the independent NumPy rows while
retaining explicit skipped FortML rows.
