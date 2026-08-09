# Deterministic accumulated SGD momentum hypergradients

This lane exercises the production `fortml_mlp_sgd_momentum_hypergradient`
adapter with `microbatch_size=2` and `accumulation_steps=3` on five training
rows. The final microbatch has one row; both the NumPy oracle and FortML weight
each microbatch by its row mass before applying the momentum state. The packed
outer coordinates are

```text
[log(learning_rate), log(l2), momentum]
```

The independent NumPy implementation checks the validation value, all three
outer gradient components, a directional JVP, and the affine one-layer HVP by
central finite differences. The release app is checked before timing and the
CSV records source and benchmark revisions. The CPU rows pass with maximum
absolute oracle error below `2e-8` in the checked value/JVP/HVP products; the
CUDA rows are explicit typed-unavailable records because the optimizer state
and its derivatives are not resident.

Run it with:

```bash
.venv/bin/python -B scripts/bench_sgd_momentum_hypergradient.py \
    --fortml ../fortml \
    --output results/sgd_momentum_hypergradient_accumulation.csv
```

The independent oracle is implemented in
`scripts/bench_sgd_momentum_hypergradient.py` (`accumulated_loss_gradient`),
and the corresponding Fortran behavioral checks are in
`test/test_mlp_sgd_momentum_hypergradient.f90`.
