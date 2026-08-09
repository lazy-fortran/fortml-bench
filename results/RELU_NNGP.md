# Analytic ReLU NNGP covariance

This lane benchmarks the exact infinite-width covariance of a three-hidden-layer
ReLU MLP with `sigma_w^2=2` and zero bias variance on fixed `192 x 8` and
`160 x 8` batches.

Run:

```bash
python3 -B scripts/bench_relu_nngp.py \
  --fortml ../fortml --output results/relu_nngp.csv
```

The NumPy oracle independently applies the arc-cosine recurrence and compares
the full covariance checksum with the FortML release app before retaining the
CPU row. The record has a typed unavailable CUDA row: no resident NNGP kernel
exists, so no host result is reported as a device measurement.

This is a kernel-limit benchmark, not a finite-width ensemble calibration or a
claim of a deterministic finite-MLP weight map.
