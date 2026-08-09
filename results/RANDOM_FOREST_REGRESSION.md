# Random-forest regression

This correctness-gated lane replays FortML's deterministic Park--Miller bootstrap stream and weighted exhaustive-split CART policy in an independent NumPy oracle. It checks scalar and multi-output predictions, staged prefixes, split-frequency feature importance, the fixed-state zero JVP, and the typed CUDA refusal.

- FortML revision: `614a6b8d3e6411ef967848d3dea341a20a62c8bd`
- Benchmark revision: `f9459848c68e67291bb35b47a8a68a0c1b77e33a`
- Compiler/flags: `gfortran -O3`
- Release wall time: `0.00537838` s
- Maximum prediction oracle error: `0`
- Raw record: [`results/random_forest_regression.csv`](random_forest_regression.csv)
