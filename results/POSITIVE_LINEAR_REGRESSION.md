# Positive-constrained linear regression

This correctness-gated lane compares weighted multi-output least squares with nonnegative feature coefficients against an independent NumPy projected-gradient oracle. It checks complete fitted, prediction, JVP, and VJP arrays, then records the typed CUDA refusal.

- FortML revision: `7bad83097eb101b1a7cd18ca44b53c5c0b4c0d90`
- Benchmark revision: `601237b9a10307b569c6f593a0eafd26943c16c7`
- Compiler/flags: `gfortran -O3`
- Release wall time: `0.0065605` s
- Maximum oracle error: `3.55271e-15`
- Raw record: [`results/positive_linear_regression.csv`](positive_linear_regression.csv)
