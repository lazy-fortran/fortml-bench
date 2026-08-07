# Coupled-L2 Adam trajectory hypergradient benchmark

This lane checks the fixed full-batch coupled-L2 Adam trajectory against an
independent NumPy implementation. The four packed outer coordinates are

```text
[ log_learning_rate, log_l2, logit_beta1, logit_beta2 ]
```

The fixture is a two-parameter affine MLP (`[0.15, -0.1]`) with five training
rows, three validation rows, four updates, learning rate `0.12`, L2 `0.07`,
`beta1=0.82`, `beta2=0.91`, and epsilon `0.03`. The independent recurrence
feeds the regularized gradient into both moments and applies bias correction;
it does not apply AdamW's decoupled parameter shrinkage. All four outer
gradient components and a directional JVP are central-differenced with
`h=2e-6`.

The release script retains a FortML CPU timing only after the app's complete
value/gradient/JVP array agrees with the NumPy oracle within `3e-6`. The app
exports that array through `FORTML_BENCH_ADAM_HYPERGRADIENT_ORACLE` and emits a
timing marker for repeated `value_gradient` calls. It uses `/mnt/storage` for
temporary oracle files.

CUDA is an explicit `unavailable` row: the complete Adam trajectory state and
its hyperparameter sensitivities are not resident on a device yet, so no host
run is relabeled as GPU evidence.

Run from this repository with a FortML checkout containing the release app:

```bash
python -B scripts/bench_adam_hypergradient.py \
  --fortml ../fortml --output results/adam_hypergradient.csv
```

Use `--skip-fortml` to regenerate only independent NumPy rows plus explicit
CPU/CUDA capability rows. The generated CSV is the provenance artifact; it
records both repository revisions, compiler flags, Python/NumPy versions,
oracle status, and the maximum FortML-vs-NumPy error.
