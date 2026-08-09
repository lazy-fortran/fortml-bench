# RMSprop trajectory hypergradient HVP benchmark

This lane extends the fixed full-batch RMSprop trajectory objective with an
exact outer Hessian-vector product on the one-layer affine MLP branch.  The
packed search vector is

```text
[ log_learning_rate, log_l2, decay, log_epsilon, momentum ]
```

The fixture has five training rows, three validation rows, four updates, and
initial model parameters `[0.15, -0.1]`. The centered branch differentiates
the square-average, gradient-average, and momentum states. An uncentered
NumPy trajectory is retained as a second oracle variant.  The independent
NumPy oracle obtains value, all five hypergradients, the directional JVP, and
all five HVP components by central finite differences.  With `h=2e-6` for
first products and an outer HVP step of `2e-5`, the FortML affine recurrence
agrees within `2.96e-9` on the centered fixture.

The focused Fortran test checks both centered and uncentered HVPs against an
independent central-difference gradient oracle, verifies the FortOpt context
adapter, and expects a typed `FORTNUM_NOT_IMPLEMENTED` refusal for a
multilayer nonlinear network, where third network derivatives would be
required.

The CPU release target emits complete value/gradient/JVP/HVP oracle rows only
after all products pass.  The measured run used gfortran `-O3` and 16
repetitions:

| product | FortML seconds/op | maximum oracle error |
| --- | ---: | ---: |
| value/gradient | `4.8937875e-05` | `6.47e-13` |
| affine HVP | `9.60285e-05` | `2.96e-09` |

CUDA remains an explicit `unavailable` capability row because resident MLP
state derivatives are not implemented yet.  No host result is relabeled as a
device measurement.

Reproduce with:

```bash
python -B scripts/bench_rmsprop_hypergradient.py \
  --fortml ../fortml --output results/rmsprop_hypergradient_hvp.csv
```
