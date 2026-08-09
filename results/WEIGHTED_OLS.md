# Weighted ordinary least squares

This lane compares deterministic weighted multi-output ordinary least squares with an independent NumPy weighted normal-equation oracle. It checks the packed fitted state and records the typed CUDA refusal; positive constraints, derivative-through-fit, and resident GPU solves are not claimed.

- FortML revision: `492831ec1dcd0639147966a28f58e3312033c0c4`
- Benchmark revision: `9f7ec4de60dd5bfb37c93d5c0cc3e7deba40682c`
- Compiler/flags: `gfortran -O3`
- Release wall time: `0.0039891` s
- Prediction-mean error: `1.9984e-15`
- Packed-checksum error: `1.25167e-12`
- Raw record: [`results/weighted_ols.csv`](weighted_ols.csv)
