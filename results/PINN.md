# PINN training adapter benchmark

`bench_pinn.py` checks the bounded `pinn_training_adapter_t` facade over
`physics_objective_t`. The independent NumPy fixture uses four weighted
manufactured affine residual terms and a nonlinear one-parameter residual for
the exact reverse-over-forward HVP. The Fortran gate additionally checks the
named term diagnostic, value/gradient/JVP/VJP products, bounded FortOpt
L-BFGS-B fitting, malformed parameter shapes, and typed CUDA refusal.

Run:

```bash
python3 -B scripts/bench_pinn.py \
  --fortml ../fortml --output results/pinn.csv
```

The HVP row is an exact product only when the residual provider registers
`physics_residual_hvp_proc`; no finite-difference fallback is hidden. CUDA is
an explicit `unavailable` capability row until a resident PINN residual and
derivative graph is linked.
