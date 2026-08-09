# Weighted Huber regression

The release application fits a weighted linear Huber model through FortOpt L-BFGS-B and exposes a fixed packed probe. The probe value and gradient norm are compared with an independent NumPy implementation; the CUDA row records the typed refusal rather than a host fallback.

- FortML revision: `f78a0966be8e237263f4dc5aa152bd4517ba09f5`
- Benchmark revision: `8e9feb53c9dc657b36373bbfbf009e5fa2261112`
- Compiler/flags: `gfortran -O3`
- Release probe wall time: `0.0038486` s
- Probe value error: `0`
- Probe gradient-norm error: `0`
- FortOpt gradient norm: `2.29759e-09`
- Raw record: [`results/huber_regression.csv`](huber_regression.csv)
