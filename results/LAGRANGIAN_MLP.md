# Scalar Lagrangian MLP

The release application is checked against an independent NumPy analytic tanh-MLP Hessian oracle for L, its state gradient, the velocity mass matrix, and the Euler--Lagrange residual. CUDA is a typed refusal; no host fallback is claimed.

- FortML revision: `e746f3886a418fc860e5ab33c86b1c1651105335`
- Benchmark revision: `98f4be8196b3f9c784d2da7458567012ecb46514`
- Compiler/flags: `gfortran -O3`
- Release wall time: `0.00385027` s
- Maximum analytic error: `0`
- Raw record: [`results/lagrangian_mlp.csv`](lagrangian_mlp.csv)
