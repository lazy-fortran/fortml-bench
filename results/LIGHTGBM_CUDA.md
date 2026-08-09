# Resident numeric LightGBM prediction

The `lightgbm_t` numeric leaf-wise model now shares the resident additive-tree
CUDA execution contract with numeric XGBoost trees. The fit remains on the
CPU; prediction flattens the fitted topology once per call, keeps node arrays
in the native plan, and applies the binary sigmoid link after raw-margin
evaluation. No host fallback is relabelled as CUDA work.

The correctness-gated row is emitted by
[`scripts/bench_lightgbm.py`](../scripts/bench_lightgbm.py) in
[`lightgbm_leafwise.csv`](lightgbm_leafwise.csv), phase `predict`, backend
`fortml_cuda`. A native build records a passing prediction error against the
independent CPU tree walk; an ordinary build records `unavailable` with
`FORTNUM_NOT_IMPLEMENTED`. The independent Fortran oracle is
`fortml/test/test_lightgbm_cuda_dispatch.f90`.

This lane covers finite numeric trees only. Categorical routing, missing-value
defaults, histogram construction/training, and resident explanation kernels
remain explicit follow-up contracts.
