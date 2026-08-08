# Dense RBF SVM classification

`scripts/bench_rbf_svm.py` compares the FortML dense training-set RBF SVM
against an independent SciPy L-BFGS-B solve of the same weighted squared-hinge
RKHS objective. The 36-by-2 fixture uses arbitrary integer labels `[-12, 37]`
and positive sample weights. The gate checks every score, label, class order,
intercept, and fixed gamma before retaining fit and prediction timings.

The finite RBF Gram matrix can be mildly ill-conditioned, so equivalent
coefficient vectors are not compared coordinate by coordinate; the behavioral
score map and labels are the oracle. The Fortran behavioral test additionally
checks fixed-state JVP/VJP CPU dispatch and all four typed CUDA derivative
refusals. The CUDA row is an explicit unavailable capability record until
resident RBF-SVM value/derivative kernels are linked.

Run:

```bash
python -B scripts/bench_rbf_svm.py \
  --fortml ../fortml --output results/rbf_svm.csv
```

The checked-in CSV records the source and benchmark revisions, compiler,
NumPy/SciPy versions, complete-output error, CPU timings, and typed CUDA
refusal. It is correctness evidence, not a claim of GPU parity.
