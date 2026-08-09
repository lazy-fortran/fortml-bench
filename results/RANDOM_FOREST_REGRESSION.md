# Random-forest regression

This correctness-gated lane replays FortML's deterministic Park--Miller bootstrap stream and weighted exhaustive-split CART policy in an independent NumPy oracle. It checks scalar and multi-output predictions, staged prefixes, split-frequency feature importance, the fixed-state zero JVP, and the typed CUDA refusal.

- FortML revision: `8baedc812fcf937b0fa17f264538cf05218cf176`
- Benchmark revision: `84cbcdadb2f7360803912833093e2791c94f1950`
- Compiler/flags: `gfortran -O3`
- Release wall time: `0.00443464` s
- Maximum prediction oracle error: `0`
- Raw record: [`results/random_forest_regression.csv`](random_forest_regression.csv)
