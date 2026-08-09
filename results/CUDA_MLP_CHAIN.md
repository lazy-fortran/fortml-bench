# Resident CUDA dense MLP-chain

This lane compares a three-layer dense chain against an independent NumPy recurrence. The oracle checks the complete value, packed input and parameter JVP, packed input and parameter VJP, a central finite difference, and the reverse-mode adjoint identity. The Fortran test checks ordinary-build typed refusal and sentinel preservation. Native CUDA is run only when both `nvcc` and `nvidia-smi` are available. An unavailable device is recorded as `typed_refusal`, never as CPU GPU evidence.

Run:

```sh
python3 scripts/bench_cuda_mlp_chain.py --fortml ../fortml \
  --output results/cuda_mlp_chain.csv \
  --report results/CUDA_MLP_CHAIN.md
```

Oracle JVP error: `1.384e-12`. Adjoint error: `5.204e-18`.
