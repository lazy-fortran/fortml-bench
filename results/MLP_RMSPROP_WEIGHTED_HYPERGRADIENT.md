# Weighted RMSprop trajectory hypergradients

This lane checks a four-step, full-batch, centered RMSprop trajectory with
non-uniform training and validation measures.  The training weights are
`[0.25, 1.5, 0, 2, 0.75]`, and the validation weights are `[2, 0.5, 1.25]`.
The packed outer vector is
`[log(learning_rate), log(l2), decay, log(epsilon), momentum]`.

An independent NumPy recurrence and central differences certify the held-out
weighted MSE, all five hypergradient components, the directional JVP, and the
affine outer HVP.  FortML reports a validation value of
`0.000276191120228226`.  Its largest absolute difference from the oracle is
`3.48e-9`.  The checked hypergradient is
`[-0.00178069534, 0.00114491629, -0.00116253179, 0.000170505438,
-0.00880353986]`.

On the recorded CPU run, FortML value-and-gradient evaluation took
`83.8 us` per operation and the HVP took `172.7 us`.  The NumPy
finite-difference gradient oracle took `522.5 us`, or 6.23 times the FortML
value-and-gradient time.  These figures exclude the optimized FortML build.

The CUDA row records the current typed refusal.  It does not claim a silent
CPU fallback or resident GPU trajectory support.

The machine-readable rows are in
`results/mlp_rmsprop_weighted_hypergradient.csv`.  Their provenance pins
FortML `124a9b430085eeb5fbf2343aa36bbc0a0a08adb5` and the benchmark harness
`95a4305bb0edda135add33ec31758757fc88555f`, both without dirty markers.
