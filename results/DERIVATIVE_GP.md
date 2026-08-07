# Derivative-observation GP query products

`bench_derivative_gp.py` checks periodic, rational-quadratic, cosine, and
polynomial mixed value/first-derivative GPs. The independent NumPy oracle builds the value,
gradient, and mixed-Hessian covariance blocks from scalar formulas, then
finite-differences complete posterior queries for input JVP and VJP checks and
assembles a dense joint posterior covariance. It also finite-differences the
packed log-kernel/log-noise coordinates to independently gate the new dense
posterior `joint_covariance_jvp` and `joint_covariance_vjp` products. The
FortML release app emits CPU timings only after all checks pass; the current
CSV has 60 correctness-gated CPU/CUDA-contract rows.

CUDA rows are deliberately recorded as `unavailable`: the derivative-GP
resident covariance/factorization graph is not linked and FortML returns
`FORTNUM_NOT_IMPLEMENTED` rather than copying through the host.

```bash
python -B scripts/bench_derivative_gp.py \
  --fortml ../fortml --output results/derivative_gp.csv
```
