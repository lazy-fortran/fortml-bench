# Weighted Huber regression

The release application fits a weighted linear Huber model through FortOpt L-BFGS-B and exposes a fixed packed probe. The probe value and gradient norm are compared with an independent NumPy implementation; the CUDA row records the typed refusal rather than a host fallback.

- FortML revision: `14c7c37a466db3e52946f5578719213ff784eb44`
- Benchmark revision: `1dc5e5ceb54ff1e66dbc34a6620e172efdd83133`
- Compiler/flags: `gfortran -O3`
- Release probe wall time: `0.0041167` s
- Probe value error: `0`
- Probe gradient-norm error: `0`
- FortOpt gradient norm: `2.29759e-09`
- Raw record: [`results/huber_regression.csv`](huber_regression.csv)
