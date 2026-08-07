# Weighted linear SVM

`scripts/bench_linear_svm.py` fits the dense primal
`linear_svm_classifier_t` on a deterministic 192-by-4 fixture with arbitrary
integer labels `[-12, 37]`, nonnegative sample weights, and feature-only L2
regularization. Its independent SciPy/NumPy oracle minimizes the identical
weighted squared-hinge objective and checks the complete class metadata,
predicted labels, and signed decision margins before retaining timings.

The release app uses FortOpt L-BFGS-B with the same stopping contract. The
recorded CPU rows are fit and fixed-state affine prediction; the CUDA row is an
explicit unavailable capability record because no resident linear-SVM kernel
is linked. This is not a host fallback or a GPU performance claim.

The ordinary hinge objective is covered by the FortML unit test's exact-margin
boundary refusal. Squared hinge has a continuous first derivative and a
piecewise second derivative at margin one.

Run:

```bash
python -B scripts/bench_linear_svm.py \
    --fortml ../fortml --output results/linear_svm.csv
```

The checked-in CSV records the source and benchmark revisions, compiler, NumPy
and SciPy versions, correctness error, and explicit CUDA status.
