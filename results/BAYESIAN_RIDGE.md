# Weighted Bayesian ridge

This lane compares the dense fixed-hyperparameter Bayesian-ridge posterior and prediction with an independent NumPy conjugate-Gaussian oracle. It records posterior evidence metadata and the typed CUDA refusal; evidence maximisation/ARD and resident GPU execution are not claimed.

- FortML revision: `b158ef667362ede3802d3c87d981fe4421042a17`
- Benchmark revision: `7c27bb5a6c660a7a762688dff1971ac179304e95`
- Compiler/flags: `gfortran -O3`
- Release wall time: `0.0041836` s
- Evidence error: `1.42109e-14`
- Prediction-mean error: `8.88178e-16`
- Raw record: [`results/bayesian_ridge.csv`](bayesian_ridge.csv)
