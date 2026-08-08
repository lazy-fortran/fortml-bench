# Derivative-observation GP query products

`bench_derivative_gp.py` checks periodic, rational-quadratic, cosine, polynomial,
spectral-mixture, and anisotropic (ARD) RBF mixed value/first-derivative GPs. The independent NumPy oracle builds the value,
gradient, and mixed-Hessian covariance blocks from scalar formulas, then
finite-differences complete posterior queries for input JVP and VJP checks and
assembles a dense joint posterior covariance. It also finite-differences the
packed log-kernel/log-noise coordinates to independently gate the new dense
posterior `joint_covariance_jvp` and `joint_covariance_vjp` products. The
FortML release app emits CPU timings only after all checks pass; the current
CSV has 94 correctness-gated CPU/CUDA-contract rows, including the polynomial
and ARD-RBF mixed-observation hyperparameter HVPs. The 95th row is the
FortSym-generated Matérn-5/2 scalar value/JVP/VJP/HVP leaf, timed after an
independent central-difference-of-gradient oracle passes.

CUDA rows are deliberately recorded as `unavailable`: the derivative-GP
resident covariance/factorization graph is not linked and FortML returns
`FORTNUM_NOT_IMPLEMENTED` rather than copying through the host.

The spectral-mixture row uses the packed GPyTorch-compatible coordinates
`[log_weight, log_scale(:), mean(:)]` and an independent dense two-feature
oracle. It covers query JVP/VJP and posterior covariance parameter JVP/VJP;
mixed parameter HVP remains a typed refusal because fourth input/parameter
products are not yet generated. The CSV includes both the independent NumPy
oracle rows and the corresponding FortML CPU/CUDA-contract rows.

The polynomial HVP row differentiates the dense mixed-observation likelihood
through the Cholesky solve in packed log variance/scale/offset/degree and
log-noise coordinates. Its value is checked against an independent NumPy
central-difference likelihood-gradient oracle; the CUDA companion row records
the typed refusal because the resident derivative-GP graph is not linked.

The ARD-RBF HVP row uses one log variance and one log lengthscale per feature.
Its value, first-derivative, and mixed-Hessian blocks are assembled directly in
the NumPy oracle, and the packed likelihood HVP is checked by central
differences before timing.

```bash
python -B scripts/bench_derivative_gp.py \
  --fortml ../fortml --output results/derivative_gp.csv
```
