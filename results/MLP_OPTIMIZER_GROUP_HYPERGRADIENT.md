# MLP optimizer-group trajectory hypergradient benchmark

This lane checks the fixed full-batch optimizer-group adapter against an
independent NumPy linear-MSE trajectory. The packed coordinates are
`[log_learning_rate, log_l2, log_multiplier_1, log_multiplier_2]`; each group
scales its post-SGD parameter delta exactly as `mlp_train` does. The release
fixture sets a fixed global `gradient_clip_norm=0.1`; clipping is applied after
the L2 term and before group scaling. Every value, gradient component, and a
directional JVP must agree with an independent clipped NumPy recurrence and
central finite differences before a FortML timing row is retained. The public
outer HVP boundary is also exercised: FortML returns a zero product and
`FORTNUM_NOT_IMPLEMENTED` (status code 3), while malformed products are
covered by the release test.

Run it with:

```bash
python -B scripts/bench_mlp_optimizer_group_hypergradient.py \
  --fortml ../fortml --output results/mlp_optimizer_group_hypergradient.csv
```

The recorded run has 21 rows: seven NumPy-oracle rows (the HVP row is
not-applicable), seven passing FortML CPU rows, and seven explicit
CUDA-unavailable rows. The FortML value-gradient timing was
`3.461025e-05` seconds per operation on the recorded host. The largest CPU
oracle discrepancy was `1.24e-11`, below the `4e-10` gate. The clipping
active-set boundary, overlapping group ranges, malformed HVP products, and
unsupported devices remain typed validation/refusal paths.
