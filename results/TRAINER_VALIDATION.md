# Generic trainer validation

The release lane checks the model-agnostic trainer's validation callback,
patience transition, best-state restoration, and schema-6 checkpoint
continuation. The default loss sequence is [0.4, 0.2, 0.25, 0.3]. With
`validation_patience=2`, step 1 is the best state and step 3 stops training.
The independent NumPy quadratic update restores the step-1 parameter vector
with zero error. A second sequence, [0.4, 0.6, 0.55, 0.5], exercises
`validation_higher_is_better` for score metrics and reaches the same
best-step/restoration contract.

The public `test_trainer` oracle also saves after step 1, resumes with the
same process-local callback, and compares parameters, metric history, best
step, and counters against uninterrupted execution. Loading the same
checkpoint without the callback is a transactional typed refusal. CSV rows
record the exact source and benchmark revisions used for the run.

| Workload | Result |
| --- | --- |
| Independent minimizing validation sequence | stop step 3, best step 1, best value 0.2 |
| Independent maximizing validation sequence | stop step 3, best step 1, best value 0.6 |
| Independent best-state restore | maximum absolute error 0 |
| Independent split continuation | maximum absolute error 0 |
| FortML public contract | pass |
| CUDA validation callback | unavailable, host-owned callback and data |

The measured public test lane took approximately 5.7 seconds in the recorded
environment. It includes compilation and the focused behavioral test. The
validation callback remains host-owned, so no resident GPU timing is claimed.
