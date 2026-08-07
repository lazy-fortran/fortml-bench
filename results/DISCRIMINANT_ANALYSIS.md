# Weighted LDA and QDA

`bench_discriminant_analysis.py` fits weighted linear and quadratic
discriminants on 120 three-feature samples with arbitrary integer labels
`[-17, 4, 23]`. The independent NumPy oracle forms weighted class means,
pooled or per-class covariances, diagonal shrinkage (`reg_param=0.03`), and
stabilized Gaussian log probabilities. Every emitted class, probability, and
prediction is checked before timing. The release app additionally emits the
fitted means/covariance diagnostics and times fixed-state probability JVPs.

Run:

```bash
python -B scripts/bench_discriminant_analysis.py \
  --fortml ../fortml --output results/discriminant_analysis.csv
```

The CSV contains NumPy oracle rows, FortML CPU fit/predict/input-JVP timings,
and explicit CUDA `unavailable` rows. LDA/QDA currently return
`FORTNUM_NOT_IMPLEMENTED` for CUDA requests; no host fallback is counted as
GPU evidence.
