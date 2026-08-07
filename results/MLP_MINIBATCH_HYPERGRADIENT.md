# Mini-batch SGD trajectory hypergradient benchmark

This lane checks a deterministic three-epoch, batch-size-two SGD trajectory
against an independent NumPy implementation. The packed outer variables are
`[log_learning_rate, log_l2]`; each epoch uses the same Park–Miller shuffle
cursor (seed 31) and the release app's fixed full-batch validation objective.
All value, gradient, and directional-JVP products must agree within `3e-10`
before CPU timing is retained. CUDA is an explicit refusal until the complete
batch-cursor and optimizer-state derivative graph is resident.

```bash
.venv/bin/python -B scripts/bench_mlp_minibatch_hypergradient.py \
  --fortml ../fortml --output results/mlp_minibatch_hypergradient.csv
```
