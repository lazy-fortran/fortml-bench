# Calibrated neural classifier benchmark

This lane runs the deterministic temperature-calibrated MLP release workload
on a 64-row, two-feature fixture. Its independent NumPy oracle checks the
fixture labels, sorted class policy, finite probability bounds, probability
simplex, and prediction domain. It deliberately does not reimplement the
FortOpt/MLP training trajectory; the fit and prediction timings are retained
only after the complete emitted contract array passes.

```bash
.venv/bin/python -B scripts/bench_mlp_calibrated_classifier.py \
  --fortml ../fortml --output results/mlp_calibrated_classifier.csv
```

The CUDA row is an explicit `unavailable` capability refusal until resident
MLP and calibration kernels are linked. Raw data are in
[`mlp_calibrated_classifier.csv`](mlp_calibrated_classifier.csv).
