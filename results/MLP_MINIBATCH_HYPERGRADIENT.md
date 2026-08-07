# Deterministic mini-batch MLP hypergradient benchmark

This lane checks the fixed seeded mini-batch SGD trajectory against an
independent NumPy linear-MSE implementation. The fixture has six training
rows, three validation rows, two parameters, three epochs, batch size two, and
Park-Miller shuffle seed 31. The packed outer variable is

```text
[ log_learning_rate, log_l2 ]
```

The NumPy oracle central-differences both components and a directional JVP
with `h=2e-6`. The FortML release app must emit the complete value, two
gradient components, and JVP array before its repeated CPU timing is retained.
The current run agrees within `2.0e-11` absolute error. CUDA rows are explicit
`unavailable` capability refusals because the complete mini-batch trajectory
and derivative state are not resident.

Run the lane with:

```bash
.venv/bin/python -B scripts/bench_mlp_minibatch_hypergradient.py \
  --fortml ../fortml --output results/mlp_minibatch_hypergradient.csv
```

The raw 12-row record includes NumPy oracle rows, FortML CPU value/gradient/JVP
rows, and typed CUDA refusals with repository and compiler provenance.
