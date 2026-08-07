# Weighted ridge regression benchmark

This lane covers the weighted, multi-output ridge estimator added to FortML.
The fixture has 96 samples, six features, three outputs, an intercept, and
`alpha=0.37`.  Weights are positive and nonuniform.  The independent oracle
forms

\[
 (X^T W X + \alpha P)^{-1}X^T W Y,
\]

with the intercept row excluded from `P`.  It checks the vector and matrix
fits, matrix and vector predictions, and the packed-coefficient/input JVP and
VJP products.  A centered finite-difference check and the full VJP adjoint
identity run before any NumPy timing is retained.

Run:

```bash
python3 scripts/bench_ridge.py --fortml ../fortml --output results/ridge.csv
```

The NumPy rows are complete and timed after the independent checks.  FortML
rows require a future `app/fortml_bench_ridge_regression.f90` release target
using the strict complete-array protocol:

```text
ridge,<workload>,<one-based-index>,<value>,<seconds>
```

The harness rejects checksum-only or incomplete output.  Until that target is
present, every FortML operation is recorded as `unavailable`; no CPU oracle
timing is relabeled as FortML timing or CUDA evidence.  The current CSV is
therefore an explicit interface/provenance record as well as a NumPy baseline.
