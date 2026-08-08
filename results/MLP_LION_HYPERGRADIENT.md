# MLP Lion trajectory hypergradient benchmark

This release lane checks the fixed full-batch Lion trajectory in
`fortml_mlp_lion_hypergradient`. The packed search coordinates are
`[log(learning_rate), log(l2), logit(beta1), logit(beta2)]`; the test fixture is
a two-parameter linear MLP with four updates. The NumPy oracle implements the
same MSE, momentum interpolation, sign update, and second momentum recurrence
independently, then central-differences every packed coordinate and one
directional product. A timing row is retained only after the complete FortML
value, four gradients, and JVP agree with that oracle.

Run it with:

```bash
python -B scripts/bench_mlp_lion_hypergradient.py \
  --fortml ../fortml --output results/mlp_lion_hypergradient.csv
```

The recorded run has 18 rows: six NumPy-oracle rows, six passing FortML CPU
rows, and six explicit CUDA-unavailable rows. FortML's value-gradient timing
was `3.47955625e-05` seconds per operation on the recorded host. The largest
CPU oracle discrepancy was `5.69e-11`, below the `4e-10` gate. Lion is
piecewise smooth; both the NumPy and FortML contracts refuse a configured
near-zero sign branch rather than treating it as differentiable. CUDA remains
unavailable until the complete model, optimizer state, and branch metadata are
resident.
