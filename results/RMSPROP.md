# RMSprop optimizer and MLP training benchmark

This lane checks two contracts against independent NumPy recurrences before
retaining a timing. `rmsprop_training` uses 4,096 parameters and 128 steps of
the uncentered FortOpt update with decay `0.9`, learning rate `1e-2`, and
epsilon `1e-8`. `rmsprop_mlp` trains a one-feature linear MLP for 32 full-batch
epochs with centered decay `0.8`, momentum `0.2`, learning rate `0.08`, and
epsilon `1e-5`. Its final loss is recomputed from the independent linear
model, so a trainer timing cannot pass with only a matching scalar state norm.

The release app is `fortml_bench_rmsprop_training`. It reports the direct
FortOpt parameter norm and the MLP final loss, each with a separate timing.
Missing compiler or target support is emitted as an explicit `unavailable`
row. The benchmark is CPU-only; no CPU result is relabeled as device evidence.

Run:

```bash
.venv/bin/python -B scripts/bench_rmsprop.py \
  --fortml ../fortml --output results/rmsprop.csv
```

The CSV stores source revisions, NumPy versions, recurrence settings, and the
oracle error for every retained row. MLP checkpoint/resume equivalence and
invalid RMSprop option refusals are independently covered by FortML's
`test_mlp_rmsprop` behavioral test.
