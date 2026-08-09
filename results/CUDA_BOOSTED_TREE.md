# Resident CUDA boosted-tree plan

This lane compares a fixed two-tree additive ensemble with an independent NumPy leaf-walk oracle. The oracle covers base score, learning rate, per-tree scales, strict split routing, and a learned NaN default. The Fortran test checks ordinary-build typed refusal, invalid-device and output-preservation behavior. It also gates the numeric XGBoost device-dispatch path, which uses the same resident plan when linked and otherwise reports FORTNUM_NOT_IMPLEMENTED. Native CUDA is run only when both `nvcc` and `nvidia-smi` are available. No device is recorded as GPU timing evidence when unavailable.

Run:

```sh
python3 scripts/bench_cuda_boosted_tree.py --fortml ../fortml \
  --output results/cuda_boosted_tree.csv \
  --report results/CUDA_BOOSTED_TREE.md
```

Oracle maximum absolute error: `2.220e-16`.
