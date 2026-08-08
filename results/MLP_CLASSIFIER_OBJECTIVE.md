# Multiclass MLP objective and L-BFGS-B benchmark

`bench_mlp_classifier_objective.py` gates weighted multiclass softmax
cross-entropy with an independent NumPy affine-logits value/JVP/HVP oracle.
The FortML fixture additionally checks the nonlinear MLP HVP, VJP duality,
optional L2 coordinate, FortOpt callback, and bounded L-BFGS-B convergence.
Rows are recorded only after both correctness gates pass.

Run:

```bash
python3 -B scripts/bench_mlp_classifier_objective.py \
  --fortml ../fortml --output results/mlp_classifier_objective.csv
```

The CUDA row is an explicit `unavailable` contract until resident multiclass
MLP objective state and its derivative graph are linked; no host fallback is
counted as GPU support.
