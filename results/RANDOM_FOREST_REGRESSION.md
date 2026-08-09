# Random-forest regression

This correctness-gated lane replays FortML's deterministic Park--Miller bootstrap stream and weighted exhaustive-split CART policy in an independent NumPy oracle. It checks scalar and multi-output predictions, staged prefixes, split-frequency feature importance, the fixed-state zero JVP, and the typed CUDA refusal.

- FortML revision: `614a6b8d3e6411ef967848d3dea341a20a62c8bd`
- Benchmark revision: `965065bc5e50a565216f2767e75f898e001723e2+dirty`
- Compiler/flags: `gfortran -O3`
- Release wall time: `0.00449234` s
- Maximum prediction oracle error: `0`
- Raw record: [`results/random_forest_regression.csv`](random_forest_regression.csv)
