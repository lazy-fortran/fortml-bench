# Fixed boosted-tree leaf objectives

The release app checks weighted squared and binary logistic objectives for both XGBoost- and LightGBM-style fixed two-leaf trees. An independent NumPy stump design checks exact value, gradient, JVP, VJP, and HVP products. FortOpt L-BFGS-B consumes the same analytic callback. Split thresholds, categorical partitions, and missing routes are discrete state; CUDA is a typed `FORTNUM_NOT_IMPLEMENTED` refusal.

- FortML revision: `ed8f92cfb0fadeb5d85f63c6a30d481491354479`
- Benchmark revision: `8f7a919b74488b7d543b50edb16391fc1ee97823`
- Compiler/flags: `gfortran -O3`
- Release probe wall time: `0.00930028` s
- Raw record: [`results/boosted_leaf_objective.csv`](boosted_leaf_objective.csv)

All CPU product errors are below `2e-11`; FortOpt converges with a three-coordinate bounded solve. The CUDA row records the explicit typed refusal rather than a host fallback.
