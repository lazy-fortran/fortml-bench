# RAdam trajectory hypergradient gate

This lane checks `fortml_mlp_radam_hypergradient`, a fixed full-batch
validation objective over
`[log(learning_rate), log(l2), logit(beta1), logit(beta2), log(epsilon)]`.
The production path propagates exact tangents through the RAdam first and
second moments, bias corrections, `rho_t` rectification, and epsilon
denominator.  FortOpt L-BFGS-B consumes the same value/gradient callback.

The Fortran fixture is an independent behavioral oracle: it compares every
packed coordinate with central differences, checks a directional JVP, checks
the scalar VJP adjoint identity, exercises the L-BFGS-B adapter, and verifies
typed optimizer/device/refusal boundaries.  Products at the `rho_t = 4`
branch or a zero second-moment square root return `FORTNUM_NOT_IMPLEMENTED`
rather than silently selecting a subgradient.  CUDA is an explicit unavailable
row until a resident RAdam trajectory kernel is linked.

Run it with:

```bash
python3 -B scripts/bench_mlp_radam_hypergradient.py \
  --fortml ../fortml \
  --output results/mlp_radam_hypergradient.csv
```

The CSV records the FortML source and benchmark revisions.  It intentionally
leaves timing empty because the subprocess includes compilation and the
behavioral gate; the existing `RADAM.md` lane carries flat-state and trainer
throughput measurements.
