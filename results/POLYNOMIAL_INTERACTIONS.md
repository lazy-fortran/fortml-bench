# Polynomial interaction basis

This lane checks the total-degree polynomial interaction basis exposed by
`make_polynomial_interaction_basis`.  For two inputs and degree two the map
returns an optional intercept followed by `x1`, `x2`, `x1²`, `x1*x2`, and
`x2²`.  The independent fixture checks ordering and compares the analytic
value/JVP/VJP/HVP products with central differences and the scalar adjoint
identity.

Run it with:

```bash
python3 -B scripts/bench_polynomial_interactions.py \
  --fortml ../fortml --output results/polynomial_interactions.csv
```

The CSV records the exact FortML and benchmark revisions.  Timing columns are
intentionally empty because the subprocess includes the Fortran build and
behavioral gate; this is a correctness record, not a resident throughput
claim.
