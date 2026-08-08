# Change-point exact-GP benchmark

`bench_change_point_gp.py` evaluates a smooth gated sum of an RBF and a
constant covariance with a separate NumPy implementation. The fixture has
four training samples in two dimensions and two queries. It checks posterior
mean and variance, a matrix JVP, a weighted parameter VJP, and the VJP HVP by
central differences over `[log_variance, log_lengthscale, log_right_variance,
log_transition_width, transition_center]`.

The script then runs `test_kernel_change_point`, which checks the scalar
covariance oracle, input gradients, mixed Hessian, parameter products, and the
exact-GP path. The CUDA row is `refused` with a typed status until a resident
change-point covariance kernel is linked. CPU host results are not labeled as
GPU evidence.

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_change_point_gp.py \
  --fortml ../fortml --output results/change_point_gp.csv
```
