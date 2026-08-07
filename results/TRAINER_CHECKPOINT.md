# Generic trainer checkpoint benchmark

`bench_trainer_checkpoint.py` uses a three-parameter quadratic and an
independent NumPy Adam recurrence.  It compares the uninterrupted trajectory
with a split-at-step-3 continuation, including parameters, first/second
moments, and the step counter.  `test_trainer` also checks EMA/history state
and rejects truncated or extra text records transactionally.

Run:

```bash
python3 -B scripts/bench_trainer_checkpoint.py \
  --fortml ../fortml --output results/trainer_checkpoint.csv
```

Checkpoint state is host-resident; the CSV therefore retains an explicit CUDA
`unavailable` row rather than implying accelerator support.
