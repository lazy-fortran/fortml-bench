# Weighted linear quantile regression

The release application fits a weighted multi-output affine model through FortOpt L-BFGS-B. An independent NumPy pinball oracle checks the packed probe value and gradient; the fit reports the exact post-continuation objective and the CUDA row records its typed refusal.

- FortML revision: `689230530edd3bb85aa5f2ccc657cc34ef6f1a5a+dirty`
- Benchmark revision: `f5fb100a5b2ecd18e0a0fd6d31df772749406319+dirty`
- Compiler/flags: `gfortran -O3`
- Release probe wall time: `0.00439466` s
- Probe value error: `1.11022e-16`
- Probe gradient-norm error: `0`
- Exact post-fit objective: `0.230757`
- Exact post-fit gradient norm: `0.109973`
- Raw record: [`results/quantile_regression.csv`](quantile_regression.csv)
