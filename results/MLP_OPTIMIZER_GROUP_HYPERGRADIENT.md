# MLP optimizer-group trajectory hypergradient benchmark

This lane checks the fixed full-batch optimizer-group adapter against an
independent NumPy linear-MSE trajectory. The packed coordinates are
`[log_learning_rate, log_l2, log_multiplier_1, log_multiplier_2]`; each group
scales its post-SGD parameter delta exactly as `mlp_train` does. Every value,
gradient component, and a directional JVP must agree with central finite
differences before a FortML timing row is retained.

Run it with:

```bash
python -B scripts/bench_mlp_optimizer_group_hypergradient.py \
  --fortml ../fortml --output results/mlp_optimizer_group_hypergradient.csv
```

The recorded run has 18 rows: six NumPy oracle rows, six passing FortML CPU
rows, and six explicit CUDA-unavailable rows. The FortML value-gradient timing
was `3.99665e-05` seconds per operation on the recorded host. The largest CPU
oracle discrepancy was `1.53e-11`, below the `4e-10` gate. Overlapping group
ranges and unsupported devices remain typed validation/refusal paths.
