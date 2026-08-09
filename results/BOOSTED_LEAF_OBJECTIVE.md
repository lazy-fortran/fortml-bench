# Fixed boosted-tree leaf objectives

The release app checks weighted squared and binary logistic objectives for both XGBoost- and LightGBM-style fixed two-leaf trees. An independent NumPy stump design checks exact value, gradient, JVP, VJP, and HVP products. FortOpt L-BFGS-B consumes the same analytic callback. Split thresholds, categorical partitions, and missing routes are discrete state; CUDA is a typed `FORTNUM_NOT_IMPLEMENTED` refusal.

- FortML revision: `3b6271b4a01ca218246b6bc2eaf95574c38ead63`
- Benchmark revision: `80ca869ba67be2ffab99e87db4dd9fb772d42903`
- Compiler/flags: `gfortran -O3`
- Release probe wall time: `0.00782356` s
- Raw record: [`results/boosted_leaf_objective.csv`](boosted_leaf_objective.csv)

All CPU product errors are below `2e-11`; FortOpt converges with a three-coordinate bounded solve. The CUDA row records the explicit typed refusal rather than a host fallback.
