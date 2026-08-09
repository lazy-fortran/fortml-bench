# Weighted ordinary least squares

This lane compares deterministic weighted multi-output ordinary least squares with an independent NumPy weighted normal-equation oracle. It checks the packed fitted state and records the typed CUDA refusal; positive constraints, derivative-through-fit, and resident GPU solves are not claimed.

- FortML revision: `20d522163a7b229492ddff6b59a9631996f7678b`
- Benchmark revision: `6087055e91e90b181e916d4612c380fc68f7cd0a`
- Compiler/flags: `gfortran -O3`
- Release wall time: `0.00445093` s
- Prediction-mean error: `1.9984e-15`
- Packed-checksum error: `1.25167e-12`
- Raw record: [`results/weighted_ols.csv`](weighted_ols.csv)
