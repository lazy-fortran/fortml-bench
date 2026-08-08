# AMSGrad trajectory hypergradient gate

This lane checks `fortml_mlp_amsgrad_hypergradient`, a fixed full-batch
validation objective over
`[log(learning_rate), log(l2), logit(beta1), logit(beta2), log(epsilon)]`.
The production path propagates exact tangents through the first and second
moments, the elementwise max-second-moment state, bias correction, and the
epsilon denominator. FortOpt L-BFGS-B consumes the same value and gradient
callback.

The NumPy path is an independent two-parameter linear-MSE recurrence. It
central-differences all five packed coordinates and one directional product.
The release app emits the complete value, gradient, and JVP array. The
benchmark retains its CPU row only when every value agrees within the recorded
tolerance. Max active-set ties, zero square-root or update denominators, and
CUDA are explicit typed refusal rows.

Run it with:

```bash
python3 -B scripts/bench_amsgrad_hypergradient.py \
  --fortml ../fortml --output results/amsgrad_hypergradient.csv
```

The CSV records source and benchmark revisions, NumPy and compiler versions,
the independent value and derivative products, the release timing, and the
CUDA capability boundary.
