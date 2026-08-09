# Derivative-GP through-fit observation products

`bench_derivative_gp_fit_products.py` independently assembles one-dimensional
RBF value/first-derivative covariance blocks and central-differences complete
refits with respect to a two-output training-target matrix. The oracle checks
the posterior mean JVP, the zero posterior-variance tangent, the JVP/VJP
adjoint identity, and a scalar refit finite difference before running the
FortML behavioral gate.

The recorded CPU oracle has mean norm `1.2654019128928644`, minimum latent
variance `1.1299395073058482e-01`, JVP norm `1.7988257500883130e-01`, and
zero NumPy finite-difference error at the displayed precision. The VJP
adjoint and scalar-refit absolute errors are `2.87e-12` and `3.38e-12`.
The FortML gate passed `test_derivative_gp_fit_products` in 4.961 s including
the focused build. CUDA is recorded as a typed
`FORTNUM_NOT_IMPLEMENTED` refusal because a resident derivative-GP solve
graph is not linked; the refusal leaves output buffers untouched.

```bash
python -B scripts/bench_derivative_gp_fit_products.py \
  --fortml ../fortml --output results/derivative_gp_fit_products.csv
```

The raw ten-row record is in
[`derivative_gp_fit_products.csv`](derivative_gp_fit_products.csv).
