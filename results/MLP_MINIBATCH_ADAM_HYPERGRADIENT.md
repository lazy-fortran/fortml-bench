# Deterministic mini-batch Adam hypergradient benchmark

This lane checks a fixed seeded mini-batch, coupled-L2 Adam trajectory against
an independent NumPy recurrence. The fixture has seven training rows, three
validation rows, five packed hyperparameters, three epochs, batch size three,
beta1 `0.84`, beta2 `0.93`, epsilon `0.025`, and Park-Miller shuffle seed 43.
The packed outer variable is

```text
[ log_learning_rate, log_l2, logit_beta1, logit_beta2, log_epsilon ]
```

The NumPy oracle directly replays the Adam first/second moments and computes
central finite differences for all five components and the directional
product. The FortML release app emits the complete value, gradient, and JVP
records before its repeated CPU timing is retained. The five-component fixture
agrees within `9e-12` absolute error. CUDA rows are explicit `unavailable`
capability refusals because the complete mini-batch Adam state and derivative
graph are not resident.

Run the lane with:

```bash
python3 -B scripts/bench_mlp_minibatch_adam_hypergradient.py \
  --fortml ../fortml \
  --output results/mlp_minibatch_adam_hypergradient.csv
```

The raw 21-row record contains NumPy oracle rows, FortML CPU
value/gradient/JVP rows, and typed CUDA refusals with repository and compiler
provenance.
