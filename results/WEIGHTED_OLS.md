# Weighted ordinary least squares

This lane compares deterministic weighted multi-output ordinary least squares with an independent NumPy weighted normal-equation oracle. It checks the packed fitted state and records the typed CUDA refusal; positive constraints, derivative-through-fit, and resident GPU solves are not claimed.

- FortML revision: `492831ec1dcd0639147966a28f58e3312033c0c4`
- Benchmark revision: `0cd94c46a341f610bdc39ec3552b6b09fbfe2ef6+dirty`
- Compiler/flags: `gfortran -O3`
- Release wall time: `0.00446359` s
- Prediction-mean error: `1.9984e-15`
- Packed-checksum error: `1.25167e-12`
- Raw record: [`results/weighted_ols.csv`](weighted_ols.csv)
