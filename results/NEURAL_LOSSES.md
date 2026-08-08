# Differentiable neural-loss benchmark

`bench_neural_losses.py` checks the shared FortML loss derivatives against
independent NumPy formulas before retaining timings. The fixture covers BCE
logits curvature, weighted multilabel BCE curvature, ordered cumulative-logit
ordinal curvature, vector softmax and log-softmax HVPs, weighted softmax
cross-entropy curvature, weighted-MSE curvature, a piecewise Huber HVP,
weighted MAE JVP, focal BCE-with-logits JVP and HVP, Gaussian NLL HVP,
Poisson/count NLL HVP, multiclass focal-softmax HVP with class factors, and the
weighted-MSE path used by the MLP objective. The multilabel, ordinal, softmax,
and focal rows use positive row weights where applicable and are checked
against independent NumPy formulas. The multiclass focal row uses the exact
true-class scalar-composition Hessian and the same weighted reduction as the
FortML MLP `focal_gamma` option.
Checksums are compared to `3e-12`; the MLP objective checksum is required to be
finite. The Huber fixture stays away from its transition so the analytic HVP is
defined; production calls refuse an exact transition kink.

```bash
python -B scripts/bench_neural_losses.py \
  --fortml ../fortml --output results/neural_losses.csv
```

The release app reports CPU seconds per operation and checksum rows. CUDA is
an explicit `unavailable` capability record until resident loss and MLP
objective kernels exist; no host fallback is timed or relabeled as GPU work.
The source API also exposes typed CUDA refusals for stable softmax,
log-softmax, weighted cross-entropy, focal-BCE, and focal-softmax value
dispatch.
