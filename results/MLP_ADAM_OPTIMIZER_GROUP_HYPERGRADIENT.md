# Grouped coupled-L2 Adam trajectory benchmark

This lane compares FortML's fixed full-batch grouped Adam trajectory with an
independent NumPy two-parameter recurrence.  Adam moment state is updated
before each group's post-update multiplier, matching `mlp_train`.

Run it with:

```bash
python -B scripts/bench_mlp_adam_optimizer_group_hypergradient.py \
  --fortml ../fortml --output results/mlp_adam_optimizer_group_hypergradient.csv \
  --report results/MLP_ADAM_OPTIMIZER_GROUP_HYPERGRADIENT.md
```

The release fixture records 18 rows.  The maximum FortML-versus-NumPy
discrepancy was `2.558e-12` (gate `3.0e-08`).  CUDA is recorded as
an explicit typed-unavailable boundary because the complete resident Adam
trajectory and derivative state are not implemented.

Source revision: `41d0509f674cff5b89902c190a843a142e463ece`  
Benchmark revision: `e8c6b4065fc446fb427ed06952501d102e9b4b53+dirty`
