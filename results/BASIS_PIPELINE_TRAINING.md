# Joint basis-pipeline training

This lane checks the CPU `basis_pipeline_training_objective_t` composition.
The packed vector contains Fourier basis log-frequency parameters followed by
multi-output linear coefficients. The Fortran gate checks the mean
half-squared-error plus ridge value, the scalar JVP, the VJP relation, a
coordinate value-gradient finite difference, a directional HVP finite
difference, and the typed CUDA refusal.

Run it with:

```bash
python -B scripts/bench_basis_pipeline_training.py \
  --fortml ../fortml --output results/basis_pipeline_training.csv
```

The fixture has seven rows and four packed parameters. Its exact fitted
residual is zero and its ridge contribution is `0.0312`. The CSV is a
correctness record. Empty timing fields are intentional because the subprocess
includes the Fortran build and test harness rather than a resident objective
timing. A failed or unavailable test is never relabeled as a CPU performance
row.
