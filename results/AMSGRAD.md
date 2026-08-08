# AMSGrad optimizer and MLP training benchmark

This lane checks the new AMSGrad recurrence against independent NumPy before
retaining a FortML timing. `amsgrad_training` uses 4,096 flat parameters for
128 steps with learning rate `1e-2`, `beta1=0.9`, `beta2=0.99`, and
`epsilon=1e-8`. `amsgrad_mlp` trains a one-feature linear MLP for 32 full-batch
epochs with learning rate `0.08`, `beta1=0.85`, `beta2=0.95`, and
`epsilon=1e-5`. The NumPy oracle applies the same bias correction and
elementwise maximum second-moment state, then recomputes the MLP loss from the
final parameters.

The release app is `fortml_bench_amsgrad_training`; the script is
`scripts/bench_amsgrad_training.py`. The recorded CPU rows have zero
oracle discrepancy on the complete parameter norm and MLP loss. CUDA rows are
explicit `unavailable`: AMSGrad has no resident state kernel and the harness
does not relabel host execution as device evidence.

The CSV records source revisions, compiler metadata, timings, and the typed
CUDA boundary:

```bash
python3 -B scripts/bench_amsgrad_training.py \
  --fortml ../fortml --output results/amsgrad.csv
```

Checkpoint continuation and formatted schema-5 round trips are covered by
FortML's independent `test_mlp_amsgrad` behavioral fixture. Fixed-trajectory
AMSGrad hypergradients remain open because the elementwise maximum introduces
active-set boundaries; future products must declare a smooth branch or a
typed refusal.
