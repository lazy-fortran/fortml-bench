# Binary MLP objective and L-BFGS-B benchmark

`bench_mlp_binary_objective.py` gates the weighted binary sigmoid objective
with an independent NumPy value/JVP/HVP finite-difference oracle.  The FortML
test additionally checks sample and class weights, VJP duality, exact HVPs,
and bounded FortOpt L-BFGS-B, including its convergence status.  A release
row is recorded only after both gates pass.

Run:

```bash
python3 -B scripts/bench_mlp_binary_objective.py \
  --fortml ../fortml --output results/mlp_binary_objective.csv
```

The CUDA row is an explicit `unavailable` contract until a resident binary
MLP objective graph is linked; there is no hidden host fallback.
