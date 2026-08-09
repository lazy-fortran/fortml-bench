# Multiclass GP hyperparameter-HVP benchmark

`bench_gp_multiclass_hyper_hvp.py` independently refits three one-vs-rest
latent Laplace-GP modes at central packed kernel probes and differences the
envelope gradient. This checks the class-block parameter layout and the
implicit HVP without replaying FortML's factorization implementation. The
release app measures the fitted CPU product and records the resident-CUDA
capability boundary.

Run:

```bash
python3 -B scripts/bench_gp_multiclass_hyper_hvp.py \
  --fortml ../fortml \
  --output results/gp_multiclass_hyper_hvp.csv
```

The independent oracle and `test_gp_multiclass_classification` pass with the
three sorted labels `[-7, 10, 42]`. The CPU release checksum agrees with the
NumPy refit-gradient oracle within the recorded tolerance. Resident
multiclass Laplace factorization/HVP kernels are not linked, so the CUDA row is
an explicit typed refusal (`FORTNUM_NOT_IMPLEMENTED`, status `3`).
