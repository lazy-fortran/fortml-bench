# Weighted Bayesian ridge

This lane compares the dense fixed-hyperparameter Bayesian-ridge posterior and
prediction with an independent NumPy conjugate-Gaussian oracle. It records
posterior evidence metadata and the typed CUDA refusal; evidence
maximisation/ARD and resident GPU execution are not claimed.

- FortML revision: `54f8ea004ea9f7fe24bdb5906f9b3b52abc66a3c`
- Benchmark revision: `d01d9b106bb78269629dc098d044b27119e2a754`
- Compiler/flags: `gfortran -O3`
- Release wall time: `9.8996e-05` s (manual smoke build; rerun the script for a host-specific timing)
- Evidence error: `1.42109e-14`
- Prediction-mean error: `8.88178e-16`
- Raw record: [`results/bayesian_ridge.csv`](bayesian_ridge.csv)
