# Polynomial-kernel binary SVM benchmark

The `polynomial_svm` lane uses eight deterministic two-feature rows and labels
`[-12,37]`. FortML fits the finite training-basis kernel
`(0.4*dot(x,z)+1.0)**2` with weighted squared-hinge `C=3`, using FortOpt
L-BFGS-B. The independent oracle is a separate SciPy L-BFGS-B solve of the
Same objective, with no FortML coefficients or scores imported into the oracle.

The release gate checks sorted class metadata, labels, degree/gamma/coef0,
intercept and dense scores. The score checksum error is `1.9984e-15` on the
verified run, with prediction accuracy `1.0`. The Fortran test independently
checks finite-difference JVPs, the VJP adjoint identity, sigmoid probability
products, malformed fit transactionality, and CPU/CUDA dispatch. CUDA is
recorded as `unavailable` with `FORTNUM_NOT_IMPLEMENTED` because no resident
polynomial-SVM kernel is linked.

## Reproduction

```bash
python -B scripts/bench_polynomial_svm.py \
  --fortml ../fortml --output results/polynomial_svm.csv
```

The pinned release run used source `fortml` revision `6a47c36` and benchmark
revision `f121fa4`, with GNU Fortran `-O3`, Python `3.14.6`, NumPy `2.5.1`, and
SciPy `1.18.0`. The CSV contains independent-oracle, FortML CPU fit/predict,
and typed CUDA capability rows.
