# Unfactored Adafactor trajectory hypergradient benchmark

This lane checks the fixed full-batch vector Adafactor objective against an
independent NumPy recurrence. The five packed outer coordinates are

```text
[ log_learning_rate, log_l2, decay, log_epsilon, log_clip_threshold ]
```

The fixture is the two-parameter affine MLP (`[0.15, -0.1]`) with five
training rows, three validation rows, four updates, learning rate `0.12`, L2
`0.07`, decay `0.75`, epsilon `0.03`, and clip threshold `0.2`. The second
moment is unfactored and the active RMS clip branch is differentiated
piecewise. All five gradient components and one directional JVP are central
finite-differenced with `h=2e-6`.

The release script retains a FortML CPU timing only after the app's complete
value/gradient/JVP array agrees with NumPy within `3e-6`. Oracle files use
`/mnt/storage`; the app exports the checked array through
`FORTML_BENCH_ADAFACTOR_HYPERGRADIENT_ORACLE` and emits a repeated
`value_gradient` timing marker. CUDA is an explicit `unavailable` row because
the full Adafactor state and its derivatives are not resident on a device.

```bash
python -B scripts/bench_adafactor_hypergradient.py \
  --fortml ../fortml --output results/adafactor_hypergradient.csv
```

`--skip-fortml` regenerates independent NumPy rows plus explicit CPU/CUDA
capability rows. The CSV records source and benchmark revisions, toolchain
metadata, oracle status, and the maximum FortML-vs-NumPy error.
