# Spectral-mixture kernel

`bench_spectral_mixture.py` compares the GPyTorch-compatible FortML spectral
mixture leaf (the exponential uses the squared positive frequency standard
deviations) with an independent NumPy oracle.  The fixture has 256 points,
three features, and two mixtures.  The packed block is
`[log_weight, log_scale(1:3), mean(1:3)]` per component.  The oracle checks
dense values, parameter JVP/VJP/HVP products, and input derivatives before
retaining release-app timings.  The HVP and input products use independent
central-product checks and are not inferred from repository state.

Run:

```bash
python -B scripts/bench_spectral_mixture.py \
  --fortml ../fortml --output results/spectral_mixture.csv
```

The CSV records NumPy checksums, FortML CPU timings, and explicit CUDA
`unavailable` rows.  CUDA is unavailable until a resident spectral-mixture
kernel is linked; no host timing is relabeled as GPU evidence.
