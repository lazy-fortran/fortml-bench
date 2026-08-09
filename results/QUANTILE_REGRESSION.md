# Weighted linear quantile regression

The release application fits a weighted multi-output affine model through FortOpt L-BFGS-B. An independent NumPy pinball oracle checks the packed probe value and gradient; the fit reports the exact post-continuation objective and the CUDA row records its typed refusal.

- FortML revision: `7bad83097eb101b1a7cd18ca44b53c5c0b4c0d90`
- Benchmark revision: `30e6ca7282c8588fae3577a1663cb6ebb806519a`
- Compiler/flags: `gfortran -O3`
- Release probe wall time: `0.00500216` s
- Probe value error: `1.11022e-16`
- Probe gradient-norm error: `0`
- Exact post-fit objective: `0.230757`
- Exact post-fit gradient norm: `0.109973`
- Raw record: [`results/quantile_regression.csv`](quantile_regression.csv)
