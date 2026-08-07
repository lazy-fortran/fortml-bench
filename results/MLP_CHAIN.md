# Composable MLP chain

This lane exercises a named `encoder -> head` `mlp_chain_t` with dimensions
`2 -> 4 -> 1`. NumPy independently forms the dense forward map, parameter/input
JVP, parameter/input VJP, and a central-difference differentiated VJP oracle for
the HVP. The release app must match every packed and per-sample product before
timings are retained.

The CUDA row is explicitly `unavailable`: the chain has no resident fused
forward/backward/HVP kernel and returns `FORTNUM_NOT_IMPLEMENTED` for CUDA
objective and optimizer requests. No host fallback is timed as CUDA.

Reproduce:

```bash
python -B scripts/bench_mlp_chain.py \
  --fortml ../fortml --output results/mlp_chain.csv
```

Raw data: [`mlp_chain.csv`](mlp_chain.csv).
