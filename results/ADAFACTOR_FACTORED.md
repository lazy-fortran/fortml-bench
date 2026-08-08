# Matrix-factored Adafactor

`bench_adafactor_factored.py` checks the layout-aware Adafactor recurrence
before recording the public FortML gate. The independent NumPy fixture uses a
2-by-2 matrix block with row/column state and a two-element vector block with
unfactored state. It checks an uninterrupted trajectory against a split and
resumed trajectory, then runs `test_mlp_adafactor_factored` and the existing
vector-Adafactor regression fixture.

The benchmark records three explicit phases:

- `independent_oracle`: NumPy row/column and vector recurrence;
- `public_contract_gate`: focused Fortran behavioral tests and MLP trainer
  integration;
- `device_contract`: CUDA is `unavailable` because no resident row/column
  kernel is linked. No host fallback is relabeled as GPU evidence.

Run from this repository with:

```bash
python -B scripts/bench_adafactor_factored.py \
  --fortml ../fortml --output results/adafactor_factored.csv
```

The CSV stores both source revisions, compiler metadata, the independent
oracle error, and the typed device boundary.
