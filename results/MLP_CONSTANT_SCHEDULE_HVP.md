# Affine constant-schedule outer-HVP benchmark

This lane checks the exact outer HVP for a one-layer affine MLP trained by a
constant typed learning-rate schedule.  The fixture has 96 training rows, 32
validation rows, eight full-batch updates, initial parameters `[0.15, -0.1]`,
base rate `0.08`, and L2 coefficient `0.03`.  The two schedule-fraction slots
are inactive and must be exact zeros.

The NumPy implementation in
`scripts/bench_mlp_constant_schedule_hvp.py` is independent of FortML.  It
replays the affine recurrence, central-differences the complete packed
validation gradient, and central-differences that gradient along the requested
direction to obtain the HVP oracle.  The FortML release app writes the complete
value/gradient/JVP/HVP array before any timing; a run is retained only when the
maximum absolute error is at most `5e-8`.

Run it with:

```bash
.venv/bin/python -B scripts/bench_mlp_constant_schedule_hvp.py \
    --fortml ../fortml --output results/mlp_constant_schedule_hvp.csv
```

The CPU rows in the companion CSV are the exact affine product.  CUDA remains
an explicit unavailable capability until a resident trajectory kernel exists;
nonconstant schedules and nonlinear MLPs retain typed HVP refusals because
their rate second products or third network derivatives are not implemented.
