# Multiclass MLP probability parameter products

This lane exercises `mlp_classifier_t` on a deterministic nine-row, two-feature
fixture with a `2 -> 3 -> 3` tanh network. The fixed-input probability JVP and
VJP cover all 21 packed network parameters. The NumPy oracle reconstructs the
Fortran column-major parameter layout, replays the tanh/softmax graph, checks
the JVP with a central perturbation, checks the VJP by perturbing every packed
coordinate, and verifies the Euclidean duality contraction.

Run it with:

```bash
python3 -B scripts/bench_mlp_classifier_parameter_products.py \
  --fortml ../fortml --output results/mlp_classifier_parameter_products.csv
```

The CPU rows are correctness-gated; the CUDA row is `unavailable` with a typed
`FORTNUM_NOT_IMPLEMENTED` contract because a resident multiclass MLP graph is
not yet linked. The CSV records source and benchmark revisions, compiler flags,
NumPy versions, timing, and raw oracle errors.

Current release rows are in
[`mlp_classifier_parameter_products.csv`](mlp_classifier_parameter_products.csv).
