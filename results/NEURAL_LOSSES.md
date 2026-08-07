# Differentiable neural-loss benchmark

`bench_neural_losses.py` checks the shared FortML loss derivatives against
independent NumPy formulas before retaining timings. The fixture covers BCE
logits curvature, softmax cross-entropy curvature, weighted-MSE curvature, a
piecewise Huber HVP, and the weighted-MSE path used by the MLP objective.
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
