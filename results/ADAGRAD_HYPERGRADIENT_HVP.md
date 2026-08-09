# Adagrad trajectory hypergradient HVP benchmark

This lane extends the fixed full-batch Adagrad products with an exact outer
Hessian-vector product on the one-layer affine MLP branch.  The packed search
vector is

```text
[ log_learning_rate, log_l2, log_epsilon ]
```

The inner fixture has five training rows, three validation rows, four Adagrad
updates, and initial model parameters `[0.15, -0.1]`.  The NumPy oracle checks
the value, all three hypergradients, a directional JVP, and all three HVP
components.  The HVP oracle is an independent central difference of the
finite-difference outer gradient (`h=2e-6` with outer direction step `2e-5`).
FortML analytic recurrence agrees within `4.54e-7` on this run.  The focused
Fortran test additionally checks the HVP with the same independent oracle and
expects a typed `FORTNUM_NOT_IMPLEMENTED` refusal for a multi-layer nonlinear
network, where a third network derivative would be required.

The CPU release app emits both value-gradient and HVP timings only after the
complete oracle array passes.  The measured run used gfortran `-O3` and 16
repetitions:

| product | FortML seconds/op | maximum oracle error |
| --- | ---: | ---: |
| value/gradient | `2.698525e-05` | `9.35e-12` |
| affine HVP | `5.456850e-05` | `4.54e-07` |

CUDA remains an explicit `unavailable` capability row because resident MLP
state derivatives are not implemented yet. No host result is relabeled as a
device measurement.

Reproduce with:

```bash
python -B scripts/bench_adagrad_hypergradient.py \
  --fortml ../fortml --output results/adagrad_hypergradient_hvp.csv
```
