# Multilabel Laplace-GP log-probability gate

This lane closes the sklearn-style `predict_log_proba` surface for independent
multilabel Laplace-GP heads.  The independent NumPy oracle checks the positive
label logarithm, central-difference input/product chain, and the adjoint rule
used when one shared kernel direction is applied to every label.  The FortML
behavioral gate additionally checks weighted fitting, threshold metadata,
input and packed per-label products, packed shared-kernel JVP/VJP products,
CPU dispatch, and output-preserving typed CUDA refusals.

Reproduce from the benchmark checkout:

```bash
python3 -B scripts/bench_gp_multilabel_log_proba.py \
  --fortml ../fortml \
  --output results/gp_multilabel_log_proba.csv
```

The NumPy oracle must report a maximum error below `2e-8`.  The CSV records
the independent oracle, the Fortran public-contract gate, and an explicit
`unavailable` CUDA row.  CUDA is not counted as CPU fallback: resident binary
Laplace states and the multilabel reduction are not linked, and every log
probability/shared-product refusal leaves caller output buffers unchanged.
