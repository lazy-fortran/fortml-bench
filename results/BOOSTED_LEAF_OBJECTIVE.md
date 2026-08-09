# Fixed boosted-tree leaf objectives

The release app checks weighted squared and binary logistic objectives for both XGBoost- and LightGBM-style fixed two-leaf trees. An independent NumPy stump design checks exact value, gradient, JVP, VJP, and HVP products. FortOpt L-BFGS-B consumes the same analytic callback. Split thresholds, categorical partitions, and missing routes are discrete state; CUDA is a typed `FORTNUM_NOT_IMPLEMENTED` refusal.

- FortML revision: `c62ee66543de67349f96a241fabd5be7c2bfb757`
- Benchmark revision: `387b378e928b67182d1da805729b28c1353bc10e+dirty`
- Compiler/flags: `gfortran -O3`
- Release probe wall time: `0.0107644` s
- Raw record: [`results/boosted_leaf_objective.csv`](boosted_leaf_objective.csv)

All CPU product errors are below `2e-11`; FortOpt converges with a three-coordinate bounded solve. The CUDA row records the explicit typed refusal rather than a host fallback.
