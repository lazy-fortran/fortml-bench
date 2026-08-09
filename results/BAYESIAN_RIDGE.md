# Weighted Bayesian ridge

This lane compares the dense fixed-hyperparameter Bayesian-ridge posterior and prediction with an independent NumPy conjugate-Gaussian oracle. It records posterior evidence metadata and the typed CUDA refusal; evidence maximisation/ARD and resident GPU execution are not claimed.

- FortML revision: `6051b2bc1092fa69ef008fce2d0c0aa3f76fe070`
- Benchmark revision: `5febb81a44f07bbafdb9a1060e2c350bafae52e7+dirty`
- Compiler/flags: `gfortran -O3`
- Release wall time: `0.00510306` s
- Evidence error: `1.42109e-14`
- Prediction-mean error: `8.88178e-16`
- Raw record: [`results/bayesian_ridge.csv`](bayesian_ridge.csv)
