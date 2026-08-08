# Affine SGD momentum outer hyper-HVP

`bench_sgd_momentum_hypergradient.py` extends the fixed full-batch SGD
momentum lane with exact outer Hessian-vector products. The NumPy oracle uses
a two-parameter one-layer linear MSE trajectory and nested central differences
only as an independent behavioral oracle. FortML propagates mixed second
tangents analytically through the velocity and learning-rate/L2/momentum
recurrences.

The release app checks value, all three packed gradient components, the
directional JVP, and all three HVP components before retaining timings. Both
the classical and Nesterov state variants are covered by the source test;
the benchmark records the classical affine lane and a typed CUDA-unavailable
row. The benchmark CSV is `sgd_momentum_hypergradient_hvp.csv`.

The FortML path is deliberately bounded: one dense layer with linear output
has a constant network Hessian, while nonlinear or multilayer models return a
typed `FORTNUM_NOT_IMPLEMENTED` status until third network derivatives are
available. No host fallback is reported as GPU execution.
